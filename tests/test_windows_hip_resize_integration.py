"""CPU-only backend selection/ABI checks; never import actual Torch or PyAV."""
import ast
from contextlib import nullcontext
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import patch

REPO=Path(__file__).resolve().parents[1]

def load_file(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

class Device:
    def __init__(self,kind='cuda',index=0): self.type=kind; self.index=index
    def __eq__(self,other): return isinstance(other,Device) and (self.type,self.index)==(other.type,other.index)

class Tensor:
    def __init__(self,shape,dtype,device,strides=None):
        self.shape=tuple(shape); self.ndim=len(shape); self.dtype=dtype; self.device=device
        stride=1; computed=[]
        for size in reversed(shape): computed.insert(0,stride); stride*=size
        self.strides=tuple(computed if strides is None else strides)
    def stride(self,index=None): return self.strides if index is None else self.strides[index]
    def data_ptr(self): return 4096
    def is_contiguous(self): return self.strides==Tensor(self.shape,self.dtype,self.device).strides
    def to(self,*args,**kwargs): return self

class ResizeIntegration(unittest.TestCase):
    def setUp(self):
        self.device=Device(); self.calls=[]; self.nvidia=False
        torch=ModuleType('torch')
        torch.float16=object(); torch.float32=object(); torch.uint8=object()
        torch.version=SimpleNamespace(hip='7.16.26354')
        torch.tensor=lambda values,*,dtype,device:Tensor((len(values),),dtype,device)
        torch.empty=lambda shape,*,dtype,device:Tensor(shape,dtype,device)
        torch.cuda=SimpleNamespace(get_device_properties=lambda d:SimpleNamespace(gcnArchName='gfx1100'),
            device=lambda d:nullcontext(),current_stream=lambda d:SimpleNamespace(cuda_stream=77))
        self.torch=torch
        contract=load_file('isolated_contract',REPO/'jasna/media/windows_hip_resize_contract.py')
        contract.validate_bundle=lambda *args:self.calls.append(('validate',args))
        hip=ModuleType('jasna.media.hip_kernel')
        hip.code_object_path=lambda name:REPO/'jasna/media'/name
        hip.hip_runtime_identity=lambda:{}
        hip.resolve_function=lambda *args:self.calls.append(('resolve',args)) or 999
        hip.launch_kernel=lambda *args,**kwargs:self.calls.append(('launch',kwargs))
        media=ModuleType('jasna.media'); media.__path__=[]; media.hip_kernel=hip
        jasna=ModuleType('jasna'); jasna.__path__=[]; jasna.media=media
        accelerator=ModuleType('jasna.accelerator'); accelerator.is_nvidia_device=lambda d:self.nvidia
        cuda=ModuleType('jasna.media.cuda_kernel')
        cuda.check_cuda=lambda *a:None; cuda.cuda_driver=lambda:None; cuda.resolve_function=lambda *a:None
        self.modules=patch.dict(sys.modules,{'torch':torch,'jasna':jasna,'jasna.media':media,
            'jasna.accelerator':accelerator,'jasna.media.cuda_kernel':cuda,
            'jasna.media.hip_kernel':hip,'jasna.media.windows_hip_resize_contract':contract})
        self.modules.start()
        self.module=load_file('isolated_resize',REPO/'jasna/media/resize_normalize.py')
        self.platform=patch.object(self.module.sys,'platform','win32'); self.platform.start()
        self.environment=patch.dict(os.environ,{'JASNA_WINDOWS_HIP_RESIZE':'0'}); self.environment.start()
    def tearDown(self):
        self.environment.stop(); self.platform.stop(); self.modules.stop()
    def normalizer(self):
        return self.module.ResizeNormalizer(device=self.device,dtype=self.torch.float32,
            mean=(0,0,0),std=(1,1,1),fill=(0,0,0))
    def test_default_amd_remains_torch_without_loading_hip(self):
        self.assertFalse(self.normalizer().available); self.assertEqual(self.calls,[])
    def test_linux_and_nvidia_defaults_unchanged(self):
        os.environ['JASNA_WINDOWS_HIP_RESIZE']='1'
        with patch.object(self.module.sys,'platform','linux'):
            self.assertFalse(self.normalizer().available)
        self.nvidia=True
        normalizer=self.normalizer()
        self.assertIs(type(normalizer._kernel),self.module._ResizeNormalizeKernel)
        self.assertEqual(self.calls,[])
    def test_opt_in_loads_product_backend_once_and_current_stream_abi(self):
        os.environ['JASNA_WINDOWS_HIP_RESIZE']='1'; normalizer=self.normalizer()
        frames=Tensor((1,3,4096,4096),self.torch.uint8,self.device,(100663296,33554432,8192,1))
        for _ in range(2): normalizer.run(frames,out_hw=(576,576),content=(0,0,576,576))
        self.assertEqual([c[0] for c in self.calls],['validate','resolve','launch','launch'])
        args=self.calls[-1][1]
        self.assertEqual(args['grid'],(36,36,1)); self.assertEqual(args['block'],(16,16,1))
        self.assertEqual(args['stream'],77); self.assertEqual(len(args['params']),20)
    def test_unsupported_shapes_are_not_advertised(self):
        os.environ['JASNA_WINDOWS_HIP_RESIZE']='1'; normalizer=self.normalizer()
        frames=Tensor((5,3,32,32),self.torch.uint8,self.device)
        self.assertFalse(normalizer.supports(frames,out_hw=(576,576),content=(0,0,576,576)))
        with self.assertRaises(ValueError): normalizer.run(frames,out_hw=(576,576),content=(0,0,576,576))
        self.assertEqual([c[0] for c in self.calls],['validate'])
    def test_strided_output_supported_but_overlap_rejected(self):
        os.environ['JASNA_WINDOWS_HIP_RESIZE']='1'; normalizer=self.normalizer()
        frames=Tensor((2,3,17,47),self.torch.uint8,self.device)
        out=Tensor((2,3,21,37),self.torch.float32,self.device,(3225,1075,43,1))
        normalizer._kernel.launch(frames,out,(2,3,31,17),normalizer._mean,normalizer._std,normalizer._fill)
        out.strides=(1,1075,43,1)
        with self.assertRaises(ValueError):
            normalizer._kernel.launch(frames,out,(2,3,31,17),normalizer._mean,normalizer._std,normalizer._fill)
    def test_shared_callers_check_supported_before_dispatch(self):
        for filename in ('rfdetr.py','yolo.py'):
            tree=ast.parse((REPO/'jasna/mosaic'/filename).read_text(encoding='utf-8'))
            method=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_preprocess')
            calls=[n.func.attr for n in ast.walk(method) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
            self.assertEqual(calls.count('supports'),1)
            self.assertEqual(calls.count('run'),1)
            self.assertTrue(any(isinstance(n,ast.If) and any(isinstance(c,ast.Call)
                and isinstance(c.func,ast.Attribute) and c.func.attr=='supports' for c in ast.walk(n.test))
                and any(isinstance(c,ast.Call) and isinstance(c.func,ast.Attribute) and c.func.attr=='run'
                    for child in n.body for c in ast.walk(child)) for n in ast.walk(method)))

    def test_actual_shared_preprocess_falls_back_without_running_unsupported_kernel(self):
        class Input(Tensor):
            def div_(self,*a): return self
            def __sub__(self,other): return self
            def __truediv__(self,other): return self
        self.torch.Tensor=Tensor
        for filename in ('rfdetr.py','yolo.py'):
            tree=ast.parse((REPO/'jasna/mosaic'/filename).read_text(encoding='utf-8'))
            method=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_preprocess')
            namespace={'torch':self.torch,'F':SimpleNamespace(interpolate=lambda x,**kw:x),
                '_letterbox_geometry':lambda *a:(1,0,0,640,640),
                '_letterbox_normalized_bchw':lambda x,**kw:'fallback'}
            exec(compile(ast.Module(body=[method],type_ignores=[]),filename,'exec'),namespace)
            called=[]
            resizer=SimpleNamespace(supports=lambda *a,**k:False,run=lambda *a,**k:called.append('run') or 'fused')
            model=SimpleNamespace(_resizer=resizer,resolution=576,imgsz=640,stride=32,
                device=self.device,input_dtype=self.torch.float32,_normalization=lambda x:(0,1))
            frames=Input((5,3,32,32),self.torch.uint8,self.device)
            got=namespace['_preprocess'](model,frames)
            self.assertEqual(called,[])
            if filename=='rfdetr.py': self.assertIs(got,frames)
            else: self.assertEqual(got,'fallback')
            resizer.supports=lambda *a,**k:True
            got=namespace['_preprocess'](model,frames)
            self.assertEqual(called,['run'])
            self.assertEqual(got if filename=='rfdetr.py' else got[0],'fused')

class BuildSourceContract(unittest.TestCase):
    def test_build_derives_only_accepted_four_tap_change(self):
        build=load_file('build_source_contract',REPO/'scripts/build_windows_hip_resize.py')
        source=build.SOURCE.read_text(encoding='utf-8')
        self.assertEqual(build.transformed_source(source),source.replace(' / 255.0f)',' * (1.0f / 255.0f))'))
        with self.assertRaises(RuntimeError): build.transformed_source(source+'\n')
        with self.assertRaises(RuntimeError): build.transformed_source(source.replace(' / 255.0f)',' / 256.0f)',1))

if __name__=='__main__': unittest.main()
