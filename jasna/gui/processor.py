"""Background processor for video processing jobs."""

import threading
import traceback
import queue
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Callable

from jasna.gui.models import JobItem, JobStatus, AppSettings
from jasna.media import UnsupportedColorspaceError


@dataclass
class ProgressUpdate:
    job_id: int
    status: JobStatus
    progress: float = 0.0
    fps: float = 0.0
    eta_seconds: float = 0.0
    frames_processed: int = 0
    total_frames: int = 0
    message: str = ""


def _cleanup_torch(torch_mod) -> None:
    import gc

    gc.collect()
    if torch_mod.cuda.is_available():
        torch_mod.cuda.synchronize()
        torch_mod.cuda.empty_cache()
        torch_mod.cuda.ipc_collect()
        torch_mod.cuda.reset_peak_memory_stats()


class Processor:
    """Handles video processing in a background thread."""
    
    def __init__(
        self,
        on_progress: Callable[[ProgressUpdate], None] = None,
        on_log: Callable[[str, str], None] = None,
        on_complete: Callable[[], None] = None,
    ):
        self._on_progress = on_progress
        self._on_log = on_log
        self._on_complete = on_complete
        
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()  # Not paused by default
        
        self._jobs: list[JobItem] = []
        self._settings: AppSettings | None = None
        self._output_folder: str = ""
        self._output_pattern: str = "{original}_restored.mp4"
        self._disable_basicvsrpp_tensorrt_for_run = False

        # Heavy models are loaded once and reused across consecutive jobs of the
        # same type; the other session is unloaded when the type switches.
        self._img_session: tuple | None = None      # (detector, restorer, device)
        self._video_session: dict | None = None
        
    def start(
        self,
        jobs: list[JobItem],
        settings: AppSettings,
        output_folder: str,
        output_pattern: str,
        *,
        disable_basicvsrpp_tensorrt: bool,
    ):
        if self._thread and self._thread.is_alive():
            return
            
        self._jobs = jobs
        self._settings = settings
        self._output_folder = output_folder
        self._output_pattern = output_pattern
        self._disable_basicvsrpp_tensorrt_for_run = bool(disable_basicvsrpp_tensorrt)
        
        self._stop_event.clear()
        self._pause_event.set()
        
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        
    def pause(self):
        if self._pause_event.is_set():
            self._pause_event.clear()
        else:
            self._pause_event.set()
            
    def is_paused(self) -> bool:
        return not self._pause_event.is_set()
        
    def stop(self):
        self._stop_event.set()
        self._pause_event.set()  # Unpause to allow thread to exit

    def join(self, timeout: float = 5.0):
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()
        
    def _log(self, level: str, message: str):
        if self._on_log:
            self._on_log(level, message)
            
    def _progress(self, update: ProgressUpdate):
        if self._on_progress:
            self._on_progress(update)
            
    def _next_pending_job(self) -> JobItem | None:
        for job in self._jobs:
            if job.status == JobStatus.PENDING:
                return job
        return None

    def _run(self):
        self._log("INFO", "Processing started")

        try:
            while not self._stop_event.is_set():
                self._pause_event.wait()
                if self._stop_event.is_set():
                    break

                job = self._next_pending_job()
                if job is None:
                    break

                self._process_job(job)
        finally:
            self._close_image_session()
            self._close_video_session()

        if self._stop_event.is_set():
            self._log("INFO", "Processing stopped by user")
        else:
            self._log("INFO", "Processing completed")
        if self._on_complete:
            self._on_complete()
            
    def _process_job(self, job: JobItem):
        job.status = JobStatus.PROCESSING
        self._log("INFO", f"Started processing {job.filename}")
        self._progress(ProgressUpdate(
            job_id=job.id,
            status=JobStatus.PROCESSING,
            message=f"Starting {job.filename}",
        ))
        
        input_path = job.path
        from jasna.media.image_io import IMAGE_EXTENSIONS
        is_image = input_path.suffix.lower() in IMAGE_EXTENSIONS

        # Determine output path
        if self._output_folder:
            output_dir = Path(self._output_folder)
        else:
            output_dir = input_path.parent

        output_name = self._output_pattern.replace("{original}", input_path.stem)
        output_path = output_dir / output_name
        if is_image:
            # The video output pattern carries a video extension; images keep their own.
            output_path = output_path.with_suffix(input_path.suffix)
        
        # Handle file conflict based on settings
        file_conflict = self._settings.file_conflict if self._settings else "auto_rename"
        
        if output_path.exists():
            if file_conflict == "skip":
                job.status = JobStatus.SKIPPED
                self._progress(ProgressUpdate(
                    job_id=job.id,
                    status=JobStatus.SKIPPED,
                    message=f"Output file already exists: {output_path.name}",
                ))
                self._log("WARNING", f"Skipped {job.filename}: output file already exists")
                return
            elif file_conflict == "auto_rename":
                output_path = self._get_unique_output_path(output_path)
                self._log("INFO", f"Renamed output to {output_path.name} to avoid overwrite")
            # "overwrite" - just proceed and let the file be replaced
        
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if is_image:
                self._close_video_session()
                self._run_image_job(job.id, input_path, output_path)
            else:
                self._close_image_session()
                self._run_video_job(job.id, input_path, output_path)

            job.status = JobStatus.COMPLETED
            self._progress(ProgressUpdate(
                job_id=job.id,
                status=JobStatus.COMPLETED,
                progress=100.0,
            ))
            self._log("INFO", f"Finished processing {job.filename}")

        except UnsupportedColorspaceError as e:
            e.__traceback__ = None
            job.status = JobStatus.SKIPPED
            self._progress(ProgressUpdate(
                job_id=job.id,
                status=JobStatus.SKIPPED,
                message=str(e),
            ))
            self._log("WARNING", f"Skipped {job.filename}: {e}")

        except Exception as e:
            tb = traceback.format_exc()
            e.__traceback__ = None
            job.status = JobStatus.ERROR
            self._progress(ProgressUpdate(
                job_id=job.id,
                status=JobStatus.ERROR,
                message=str(e),
            ))
            self._log("ERROR", f"Failed to process {job.filename}: {e}\n{tb}")

        try:
            import torch
            _cleanup_torch(torch)
        except Exception:
            pass
            
    def _ensure_video_session(self):
        """Compile engines + build the BasicVSR++ (and optional secondary) restorer
        once; reused across consecutive video jobs."""
        if self._video_session is not None:
            return
        from jasna._suppress_noise import install as _install_noise_filters
        _install_noise_filters()
        import torch
        from jasna.engine_compiler import EngineCompilationRequest, ensure_engines_compiled
        from jasna.engine_paths import model_weights_dir
        from jasna.media import parse_encoder_settings, validate_encoder_settings
        from jasna.mosaic.detection_registry import coerce_detection_model_name, detection_model_weights_path
        from jasna.restorer.basicvsrpp_mosaic_restorer import BasicvsrppMosaicRestorer
        from jasna.restorer.denoise import DenoiseStep, DenoiseStrength
        from jasna.restorer.restoration_pipeline import RestorationPipeline

        settings = self._settings
        device = torch.device("cuda:0")
        restoration_model_path = model_weights_dir() / "lada_mosaic_restoration_model_generic_v1.2.pth"
        det_name = coerce_detection_model_name(str(settings.detection_model))
        detection_model_path = detection_model_weights_path(det_name)

        compile_basicvsrpp = bool(settings.compile_basicvsrpp) and (not self._disable_basicvsrpp_tensorrt_for_run)
        compile_result = ensure_engines_compiled(
            EngineCompilationRequest(
                device=str(device),
                fp16=settings.fp16_mode,
                basicvsrpp=compile_basicvsrpp,
                basicvsrpp_model_path=str(restoration_model_path),
                basicvsrpp_max_clip_size=int(settings.max_clip_size),
                detection=True,
                detection_model_name=det_name,
                detection_model_path=str(detection_model_path),
                detection_batch_size=settings.batch_size,
                unet4x=(settings.secondary_restoration == "unet-4x"),
            ),
            log_callback=lambda msg: self._log("INFO", msg),
        )
        use_tensorrt = compile_result.use_basicvsrpp_tensorrt

        secondary_restorer = None
        if settings.secondary_restoration == "tvai":
            from jasna.restorer.tvai_secondary_restorer import TvaiSecondaryRestorer
            tvai_args_str = f"model={settings.tvai_model}:scale={settings.tvai_scale}:{settings.tvai_args}"
            secondary_restorer = TvaiSecondaryRestorer(
                ffmpeg_path=settings.tvai_ffmpeg_path,
                tvai_args=tvai_args_str,
                scale=settings.tvai_scale,
                num_workers=settings.tvai_workers,
            )
        elif settings.secondary_restoration == "unet-4x":
            from jasna.restorer.unet4x_secondary_restorer import Unet4xSecondaryRestorer
            secondary_restorer = Unet4xSecondaryRestorer(device=device, fp16=settings.fp16_mode)
        elif settings.secondary_restoration == "rtx-super-res":
            from jasna.restorer.rtx_superres_secondary_restorer import RtxSuperresSecondaryRestorer
            rtx_denoise = settings.rtx_denoise.lower()
            rtx_deblur = settings.rtx_deblur.lower()
            secondary_restorer = RtxSuperresSecondaryRestorer(
                device=device,
                scale=settings.rtx_scale,
                quality=settings.rtx_quality.lower(),
                denoise=None if rtx_denoise == "none" else rtx_denoise,
                deblur=None if rtx_deblur == "none" else rtx_deblur,
            )

        restoration_pipeline = RestorationPipeline(
            restorer=BasicvsrppMosaicRestorer(
                checkpoint_path=str(restoration_model_path),
                device=device,
                max_clip_size=settings.max_clip_size,
                use_tensorrt=use_tensorrt,
                fp16=settings.fp16_mode,
            ),
            secondary_restorer=secondary_restorer,
            denoise_strength=DenoiseStrength(settings.denoise_strength),
            denoise_step=DenoiseStep(settings.denoise_step),
        )

        encoder_settings = {}
        if settings.encoder_cq:
            encoder_settings["cq"] = settings.encoder_cq
        if settings.encoder_custom_args:
            encoder_settings.update(parse_encoder_settings(settings.encoder_custom_args))
        encoder_settings = validate_encoder_settings(encoder_settings)

        self._video_session = {
            "device": device,
            "det_name": det_name,
            "detection_model_path": detection_model_path,
            "restoration_pipeline": restoration_pipeline,
            "secondary_restorer": secondary_restorer,
            "encoder_settings": encoder_settings,
            "working_directory": Path(settings.working_directory) if (settings.working_directory or "").strip() else None,
            "lut_path": (settings.lut_path or "").strip() or None,
        }
        self._log("INFO", "Restoration models loaded (reused across video jobs)")

    def _run_video_job(self, job_id: int, input_path: Path, output_path: Path):
        from jasna.pipeline import Pipeline

        self._ensure_video_session()
        s = self._video_session
        settings = self._settings
        last_update_time = [0.0]

        def progress_callback(progress_pct: float, fps: float, eta_seconds: float, frames_done: int, total: int):
            current_time = time.time()
            if current_time - last_update_time[0] < 0.1:
                return
            last_update_time[0] = current_time

            self._pause_event.wait()
            if self._stop_event.is_set():
                raise InterruptedError("Processing stopped")

            self._progress(ProgressUpdate(
                job_id=job_id,
                status=JobStatus.PROCESSING,
                progress=progress_pct,
                fps=fps,
                eta_seconds=eta_seconds,
                frames_processed=frames_done,
                total_frames=total,
            ))

        pipeline = None
        try:
            pipeline = Pipeline(
                input_video=input_path,
                output_video=output_path,
                detection_model_name=s["det_name"],
                detection_model_path=s["detection_model_path"],
                detection_score_threshold=settings.detection_score_threshold,
                restoration_pipeline=s["restoration_pipeline"],
                codec=settings.codec,
                encoder_settings=s["encoder_settings"],
                batch_size=settings.batch_size,
                device=s["device"],
                max_clip_size=settings.max_clip_size,
                temporal_overlap=settings.temporal_overlap,
                enable_crossfade=settings.enable_crossfade,
                fp16=settings.fp16_mode,
                disable_progress=True,
                progress_callback=progress_callback,
                working_directory=s["working_directory"],
                lut_path=s["lut_path"],
            )
            pipeline.run()
        finally:
            if pipeline is not None:
                pipeline.close()
            from jasna.tracking.blending import _KERNEL_CACHE
            _KERNEL_CACHE.clear()
            from jasna.media.rgb_to_p010 import _cache as _p010_cache
            _p010_cache.clear()

    def _close_video_session(self):
        if self._video_session is None:
            return
        s = self._video_session
        self._video_session = None
        s["restoration_pipeline"].restorer.close()
        secondary = s["secondary_restorer"]
        if secondary is not None and hasattr(secondary, "close"):
            secondary.close()
        import gc
        import torch
        for _ in range(3):
            gc.collect()
        _cleanup_torch(torch)
        self._log("INFO", "Restoration models unloaded")

    def _ensure_image_session(self):
        """Load the rf-detr detector + SD 1.5 restorer once; reused across image jobs."""
        if self._img_session is not None:
            return
        from jasna._suppress_noise import install as _install_noise_filters
        _install_noise_filters()
        import torch
        from jasna.engine_compiler import EngineCompilationRequest, ensure_engines_compiled
        from jasna.engine_paths import SD15_DIR
        from jasna.mosaic.detection_registry import build_detection_model, coerce_detection_model_name, detection_model_weights_path
        from jasna.restorer.sd15_download import bundle_present
        from jasna.restorer.sd15_inpaint_restorer import Sd15InpaintRestorer

        settings = self._settings
        device = torch.device("cuda:0")
        if not bundle_present(SD15_DIR):
            raise FileNotFoundError(
                f"SD 1.5 model not found at {SD15_DIR}. Use 'Download model' in the "
                "Image Restoration settings."
            )

        det_name = coerce_detection_model_name(str(settings.detection_model))
        detection_model_path = detection_model_weights_path(det_name)
        ensure_engines_compiled(
            EngineCompilationRequest(
                device=str(device),
                fp16=settings.fp16_mode,
                detection=True,
                detection_model_name=det_name,
                detection_model_path=str(detection_model_path),
                detection_batch_size=settings.batch_size,
            ),
            log_callback=lambda msg: self._log("INFO", msg),
        )
        detector = build_detection_model(
            det_name,
            detection_model_path,
            batch_size=settings.batch_size,
            device=device,
            score_threshold=settings.detection_score_threshold,
            fp16=settings.fp16_mode,
        )
        restorer = Sd15InpaintRestorer(SD15_DIR, device, settings.fp16_mode)
        self._img_session = (detector, restorer, device)
        self._log("INFO", "SD 1.5 model loaded (reused across image jobs)")

    def _run_image_job(self, job_id: int, input_path: Path, output_path: Path):
        """Restore a still image with the (shared) SD 1.5 inpaint session."""
        from jasna.image_restore import clamp_strength, restore_image, variant_output_paths
        from jasna.media import image_io
        from jasna.restorer.sd15_inpaint_restorer import DEFAULT_FREEU

        self._ensure_image_session()
        detector, restorer, device = self._img_session
        settings = self._settings

        self._pause_event.wait()
        if self._stop_event.is_set():
            raise InterruptedError("Processing stopped")
        self._progress(ProgressUpdate(job_id=job_id, status=JobStatus.PROCESSING, progress=20.0, message="Detecting mosaics"))

        num_variants = max(1, int(settings.image_restore_variants))
        freeu = dict(DEFAULT_FREEU) if bool(settings.image_restore_freeu) else None
        strength = clamp_strength(float(settings.image_restore_strength))

        img = image_io.read_image_rgb_chw(input_path)
        outputs = restore_image(
            img, detector, restorer,
            device=device, fp16=settings.fp16_mode,
            steps=int(settings.image_restore_steps),
            strength=strength, seed=int(settings.image_restore_seed),
            num_variants=num_variants, freeu=freeu,
        )
        for path, out in zip(variant_output_paths(output_path, num_variants), outputs):
            image_io.write_image_rgb_chw(path, out)
            self._log("INFO", f"Wrote {path.name}")
        self._progress(ProgressUpdate(job_id=job_id, status=JobStatus.PROCESSING, progress=100.0))

    def _close_image_session(self):
        if self._img_session is None:
            return
        detector, restorer, _ = self._img_session
        self._img_session = None
        detector.close()
        restorer.close()
        import gc
        import torch
        for _ in range(3):
            gc.collect()
        _cleanup_torch(torch)
        self._log("INFO", "SD 1.5 model unloaded")

    def _get_unique_output_path(self, output_path: Path) -> Path:
        """Find a unique output path by adding a counter suffix if file exists."""
        if not output_path.exists():
            return output_path
            
        stem = output_path.stem
        suffix = output_path.suffix
        parent = output_path.parent
        
        counter = 1
        while True:
            new_name = f"{stem} ({counter}){suffix}"
            new_path = parent / new_name
            if not new_path.exists():
                return new_path
            counter += 1
            if counter > 9999:
                raise RuntimeError(f"Could not find unique filename after 9999 attempts: {output_path}")
