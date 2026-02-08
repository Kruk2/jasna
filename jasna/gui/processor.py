"""Background processor for video processing jobs."""

import threading
import queue
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Callable

from jasna.gui.models import JobItem, JobStatus, AppSettings


@dataclass
class ProgressUpdate:
    job_index: int
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
        self._output_pattern: str = "{original}_restored.mkv"
        
    def start(
        self,
        jobs: list[JobItem],
        settings: AppSettings,
        output_folder: str,
        output_pattern: str,
    ):
        if self._thread and self._thread.is_alive():
            return
            
        self._jobs = jobs
        self._settings = settings
        self._output_folder = output_folder
        self._output_pattern = output_pattern
        
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
        
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()
        
    def _log(self, level: str, message: str):
        if self._on_log:
            self._on_log(level, message)
            
    def _progress(self, update: ProgressUpdate):
        if self._on_progress:
            self._on_progress(update)
            
    def _run(self):
        self._log("INFO", "Processing started")
        
        for idx, job in enumerate(self._jobs):
            if self._stop_event.is_set():
                self._log("INFO", "Processing stopped by user")
                break
                
            self._pause_event.wait()  # Block if paused
            
            if self._stop_event.is_set():
                break
                
            self._process_job(idx, job)
            
        self._log("INFO", "Processing completed")
        if self._on_complete:
            self._on_complete()
            
    def _process_job(self, idx: int, job: JobItem):
        self._log("INFO", f"Started processing {job.filename}")
        self._progress(ProgressUpdate(
            job_index=idx,
            status=JobStatus.PROCESSING,
            message=f"Starting {job.filename}",
        ))
        
        input_path = job.path
        
        # Determine output path
        if self._output_folder:
            output_dir = Path(self._output_folder)
        else:
            output_dir = input_path.parent
            
        output_name = self._output_pattern.replace("{original}", input_path.stem)
        output_path = output_dir / output_name
        
        # Handle file conflict based on settings
        file_conflict = self._settings.file_conflict if self._settings else "auto_rename"
        
        if output_path.exists():
            if file_conflict == "skip":
                self._progress(ProgressUpdate(
                    job_index=idx,
                    status=JobStatus.SKIPPED,
                    message=f"Output file already exists: {output_path.name}",
                ))
                self._log("WARNING", f"Skipped {job.filename}: output file already exists")
                return
            elif file_conflict == "auto_rename":
                # Find a unique filename with counter suffix
                output_path = self._get_unique_output_path(output_path)
                self._log("INFO", f"Renamed output to {output_path.name} to avoid overwrite")
            # "overwrite" - just proceed and let the file be replaced
        
        try:
            self._run_pipeline(idx, input_path, output_path)
            
            self._progress(ProgressUpdate(
                job_index=idx,
                status=JobStatus.COMPLETED,
                progress=100.0,
            ))
            self._log("INFO", f"Finished processing {job.filename}")
            
        except Exception as e:
            self._progress(ProgressUpdate(
                job_index=idx,
                status=JobStatus.ERROR,
                message=str(e),
            ))
            self._log("ERROR", f"Failed to process {job.filename}: {e}")
            
    def _run_pipeline(self, job_idx: int, input_path: Path, output_path: Path):
        """Run the actual processing pipeline."""
        import torch
        from jasna.media import get_video_meta_data, parse_encoder_settings, validate_encoder_settings
        from jasna.pipeline import Pipeline
        from jasna.restorer.basicvrspp_tenorrt_compilation import basicvsrpp_startup_policy
        from jasna.restorer.basicvsrpp_mosaic_restorer import BasicvsrppMosaicRestorer
        from jasna.restorer.denoise import DenoiseStep, DenoiseStrength
        from jasna.restorer.restoration_pipeline import RestorationPipeline
        from jasna.restorer.swin2sr_secondary_restorer import Swin2srSecondaryRestorer
        from jasna.restorer.tvai_secondary_restorer import TvaiSecondaryRestorer, _parse_tvai_args_kv
        
        settings = self._settings
        device = torch.device("cuda:0")
        
        # Get video metadata for progress tracking
        metadata = get_video_meta_data(str(input_path))
        total_frames = metadata.num_frames
        
        # Model paths
        restoration_model_path = Path("model_weights") / "lada_mosaic_restoration_model_generic_v1.2.pth"
        detection_model_path = Path("model_weights") / "rfdetr-v3.onnx"
        
        use_tensorrt = basicvsrpp_startup_policy(
            restoration_model_path=str(restoration_model_path),
            max_clip_size=settings.max_clip_size,
            device=device,
            fp16=settings.fp16_mode,
            compile_basicvsrpp=settings.compile_basicvsrpp,
        )
        
        secondary_restorer = None
        restoration_pipeline = None
        pipeline = None
        stream = None
        try:
            if settings.secondary_restoration == "swin2sr":
                secondary_restorer = Swin2srSecondaryRestorer(
                    device=device,
                    fp16=settings.fp16_mode,
                    batch_size=settings.swin2sr_batch_size,
                    use_tensorrt=settings.swin2sr_tensorrt,
                )
            elif settings.secondary_restoration == "tvai":
                tvai_args = f"model={settings.tvai_model}:scale={settings.tvai_scale}"
                if settings.tvai_args.strip():
                    tvai_args = f"{tvai_args}:{settings.tvai_args}"
                secondary_restorer = TvaiSecondaryRestorer(
                    device=device,
                    ffmpeg_path=settings.tvai_ffmpeg_path,
                    tvai_args=tvai_args,
                    max_clip_size=settings.max_clip_size,
                    num_workers=settings.tvai_workers,
                )

            denoise_strength = DenoiseStrength(settings.denoise_strength)
            denoise_step = DenoiseStep(settings.denoise_step)

            restoration_pipeline = RestorationPipeline(
                restorer=BasicvsrppMosaicRestorer(
                    checkpoint_path=str(restoration_model_path),
                    device=device,
                    max_clip_size=settings.max_clip_size,
                    use_tensorrt=use_tensorrt,
                    fp16=settings.fp16_mode,
                ),
                secondary_restorer=secondary_restorer,
                denoise_strength=denoise_strength,
                denoise_step=denoise_step,
            )

            encoder_settings = {}
            if settings.encoder_cq:
                encoder_settings["cq"] = settings.encoder_cq
            if settings.encoder_custom_args:
                encoder_settings.update(parse_encoder_settings(settings.encoder_custom_args))
            encoder_settings = validate_encoder_settings(encoder_settings)

            stream = torch.cuda.Stream()

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
                    job_index=job_idx,
                    status=JobStatus.PROCESSING,
                    progress=progress_pct,
                    fps=fps,
                    eta_seconds=eta_seconds,
                    frames_processed=frames_done,
                    total_frames=total,
                ))

            pipeline = Pipeline(
                input_video=input_path,
                output_video=output_path,
                detection_model_path=detection_model_path,
                detection_score_threshold=settings.detection_score_threshold,
                restoration_pipeline=restoration_pipeline,
                codec=settings.codec,
                encoder_settings=encoder_settings,
                stream=stream,
                batch_size=settings.batch_size,
                device=device,
                max_clip_size=settings.max_clip_size,
                temporal_overlap=settings.temporal_overlap,
                enable_crossfade=settings.enable_crossfade,
                fp16=settings.fp16_mode,
                disable_progress=True,
                progress_callback=progress_callback,
            )

            pipeline.run()
        finally:
            if secondary_restorer is not None and hasattr(secondary_restorer, "close"):
                try:
                    secondary_restorer.close()
                except Exception as e:
                    self._log("WARNING", f"Cleanup warning: failed to close secondary restorer: {e}")

            del pipeline
            del restoration_pipeline
            del secondary_restorer
            del stream

            _cleanup_torch(torch)

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
            if counter > 9999:  # Safety limit
                raise RuntimeError(f"Could not find unique filename after 9999 attempts: {output_path}")
