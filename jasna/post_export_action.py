import logging
import os
import re
import shlex
import signal
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from jasna.os_utils import subprocess_no_window_kwargs


logger = logging.getLogger(__name__)


POST_EXPORT_ACTION_NONE = "none"
POST_EXPORT_ACTION_SHUTDOWN = "shutdown"
POST_EXPORT_ACTION_COMMAND = "command"
POST_EXPORT_ACTIONS = (
    POST_EXPORT_ACTION_NONE,
    POST_EXPORT_ACTION_SHUTDOWN,
    POST_EXPORT_ACTION_COMMAND,
)


class PostExportVideoCommandError(RuntimeError):
    pass


class PostExportVideoCommandCancelled(Exception):
    pass


def validate_post_export_action(action: str, command: str) -> None:
    if action not in POST_EXPORT_ACTIONS:
        raise ValueError(f"Unsupported post-export action: {action}")
    if action == POST_EXPORT_ACTION_COMMAND and not command.strip():
        raise ValueError("--post-export-command is required when --post-export-action=command")


def run_post_export_action(action: str, command: str = "") -> None:
    validate_post_export_action(action, command)

    if action == POST_EXPORT_ACTION_NONE:
        return

    if action == POST_EXPORT_ACTION_SHUTDOWN:
        cmd = ["shutdown", "/s", "/t", "0"] if sys.platform == "win32" else ["shutdown", "-h", "now"]
        subprocess.Popen(cmd, **subprocess_no_window_kwargs())
        return

    subprocess.Popen(command.strip(), shell=True, **subprocess_no_window_kwargs())


def run_post_export_action_safely(action: str, command: str, report_error: Callable[[str], None]) -> bool:
    try:
        run_post_export_action(action, command)
        return True
    except Exception as e:
        report_error(f"Post-export action failed: {e}")
        return False


def _quote_shell_value(value: str) -> str:
    if sys.platform == "win32":
        return subprocess.list2cmdline([value])
    return shlex.quote(value)


def expand_post_export_video_command(
    command: str,
    input_path: Path,
    output_path: Path,
) -> str:
    resolved_input = input_path.resolve(strict=False)
    resolved_output = output_path.resolve(strict=False)
    values = {
        "input": str(resolved_input),
        "output": str(resolved_output),
        "output_dir": str(resolved_output.parent),
        "output_stem": resolved_output.stem,
        "output_suffix": resolved_output.suffix,
    }
    pattern = re.compile(r"\{(" + "|".join(values) + r")\}")
    return pattern.sub(
        lambda match: _quote_shell_value(values[match.group(1)]),
        command.strip(),
    )


def _terminate_process_tree(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            **subprocess_no_window_kwargs(),
        )
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            logger.debug("Post-export command exited during termination", exc_info=True)
            return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if sys.platform == "win32":
            process.kill()
        else:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                logger.debug("Post-export command exited before forced termination", exc_info=True)
        process.wait()


def run_post_export_video_command(
    command: str,
    input_path: Path,
    output_path: Path,
    cancel_requested: Callable[[], bool],
) -> None:
    expanded = expand_post_export_video_command(command, input_path, output_path)
    kwargs = subprocess_no_window_kwargs()
    if sys.platform == "win32":
        kwargs["creationflags"] = (
            kwargs.get("creationflags", 0) | subprocess.CREATE_NEW_PROCESS_GROUP
        )
    else:
        kwargs["start_new_session"] = True
    try:
        process = subprocess.Popen(
            expanded,
            shell=True,
            cwd=output_path.resolve(strict=False).parent,
            **kwargs,
        )
    except OSError as exc:
        raise PostExportVideoCommandError(
            f"Could not start post-export video command: {exc}"
        ) from exc
    try:
        while True:
            if cancel_requested():
                _terminate_process_tree(process)
                raise PostExportVideoCommandCancelled("Post-export video command cancelled")
            try:
                return_code = process.wait(timeout=0.1)
                break
            except subprocess.TimeoutExpired:
                continue
    except KeyboardInterrupt:
        _terminate_process_tree(process)
        raise
    if return_code != 0:
        raise PostExportVideoCommandError(
            f"Post-export video command failed with exit code {return_code}"
        )
