import subprocess
import sys

from jasna.os_utils import subprocess_no_window_kwargs


POST_EXPORT_ACTION_NONE = "none"
POST_EXPORT_ACTION_SHUTDOWN = "shutdown"
POST_EXPORT_ACTION_COMMAND = "command"
POST_EXPORT_ACTIONS = (
    POST_EXPORT_ACTION_NONE,
    POST_EXPORT_ACTION_SHUTDOWN,
    POST_EXPORT_ACTION_COMMAND,
)


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
