"""Fail-closed verification for one completed Windows Job guard invocation.

This module deliberately has no filesystem or process operations.  A caller
must separately own process lifetime, capture exactly one stderr guard record,
and read the uniquely allocated guard report within its own bounded protocol.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
import json
import re
from typing import Any, Final


MIB: Final = 1024 * 1024
JOB_MEMORY_LIMIT_BYTES: Final = 6144 * MIB
MINIMUM_HOST_COMMIT_FREE_BYTES: Final = 10240 * MIB
MINIMUM_HOST_PHYSICAL_AVAILABLE_BYTES: Final = 8192 * MIB
MAX_TOTAL_PROCESSES: Final = 16

# The guard records are normally a few KiB.  This cap is deliberately much
# smaller than a general log capture and is checked before decoding JSON.
MAX_RECORD_BYTES: Final = 64 * 1024
MAX_COMMAND_ARGUMENTS: Final = 256
MAX_COMMAND_TEXT_CHARACTERS: Final = 32 * 1024
MAX_TEXT_CHARACTERS: Final = 16 * 1024
MAX_REASON_CHARACTERS: Final = 4096

_UINT32_MAX: Final = (1 << 32) - 1
_UINT64_MAX: Final = (1 << 64) - 1
_REQUIRED_FIELDS: Final = frozenset(
    {
        "started_utc",
        "finished_utc",
        "status",
        "reason",
        "launcher_exit_code",
        "child_exit_code",
        "child_process_id",
        "job_memory_limit_bytes",
        "peak_job_memory_bytes",
        "total_processes",
        "active_processes",
        "total_terminated_processes",
        "minimum_host_commit_free_bytes",
        "minimum_host_physical_available_bytes",
        "working_directory",
        "command",
    }
)
_TIMESTAMP_RE: Final = re.compile(
    r"\A(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})\.(\d{7})Z\Z"
)


class GuardVerificationError(ValueError):
    """Raised when there is no verified child exit code to preserve."""


@dataclass(frozen=True, slots=True, init=False)
class ExpectedGuardContext:
    """Trusted exact values for the one guard invocation being verified.

    Construction snapshots a sequence into a tuple, so later mutation of a
    caller-owned list cannot change the context that is compared to a record.
    These are literal decoded strings: this class never resolves, normalizes,
    case-folds, or otherwise rewrites Windows paths.
    """

    command: tuple[str, ...]
    working_directory: str

    def __init__(
        self, command: Sequence[str], working_directory: str
    ) -> None:
        object.__setattr__(self, "command", _freeze_command(command, "expected_command"))
        object.__setattr__(
            self,
            "working_directory",
            _require_text(
                working_directory,
                "expected_working_directory",
                MAX_TEXT_CHARACTERS,
                allow_empty=False,
            ),
        )


def verify_guard_exit(
    report_bytes: bytes,
    stderr_record_bytes: bytes,
    *,
    expected_command: Sequence[str],
    expected_working_directory: str,
    actual_outer_exit_code: int,
) -> int:
    """Return a child exit code only after one guarded run is fully verified.

    ``stderr_record_bytes`` is the JSON payload after the caller has removed
    the single ``JASNA_JOB_RESULT=`` prefix.  The caller must already have
    enforced uniqueness of that stderr record and retained both byte strings
    within its own capture bounds.  On every mismatch or unsafe state this
    function raises :class:`GuardVerificationError`; callers must not forward a
    previously observed child exit code after that exception.
    """

    context = ExpectedGuardContext(expected_command, expected_working_directory)
    outer_exit_code = _require_uint32(actual_outer_exit_code, "actual_outer_exit_code")
    report = _decode_record(report_bytes, "report")
    stderr_record = _decode_record(stderr_record_bytes, "stderr guard record")

    # Validate each decoded object before structural comparison.  The schema
    # reduces accepted values to one shallow object plus a flat string list, so
    # an adversarial JSON nesting cannot drive recursive comparison first.
    report_child_exit = _validate_safe_record(report, context, outer_exit_code)
    stderr_child_exit = _validate_safe_record(stderr_record, context, outer_exit_code)

    if not _structurally_equal(report, stderr_record):
        raise GuardVerificationError("report and stderr guard record differ structurally")

    # Structural equality makes this redundant check explicit and guards the
    # return value if this function is changed later.
    if report_child_exit != stderr_child_exit:
        raise GuardVerificationError("report and stderr child exits differ")
    return report_child_exit


def _decode_record(raw: bytes, label: str) -> dict[str, Any]:
    if not isinstance(raw, bytes):
        raise GuardVerificationError(f"{label} must be bytes")
    if not raw:
        raise GuardVerificationError(f"{label} is empty")
    if len(raw) > MAX_RECORD_BYTES:
        raise GuardVerificationError(f"{label} exceeds {MAX_RECORD_BYTES} bytes")

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise GuardVerificationError(f"{label} is not UTF-8") from error

    try:
        decoded = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_int=_parse_json_integer,
            parse_float=_reject_json_float,
            parse_constant=_reject_json_constant,
        )
    except GuardVerificationError:
        raise
    except (json.JSONDecodeError, OverflowError, RecursionError, ValueError) as error:
        raise GuardVerificationError(f"{label} is not strict JSON") from error

    if not isinstance(decoded, dict):
        raise GuardVerificationError(f"{label} must contain a JSON object")
    return decoded


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise GuardVerificationError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _parse_json_integer(literal: str) -> int:
    digits = literal[1:] if literal.startswith("-") else literal
    # Every numeric field in the v2 result is at most an unsigned 64-bit
    # integer.  Bound parsing before constructing an arbitrary-size Python int.
    if len(digits) > 20:
        raise GuardVerificationError("JSON integer is wider than 64 bits")
    return int(literal)


def _reject_json_float(_: str) -> Any:
    raise GuardVerificationError("JSON floating-point values are not allowed")


def _reject_json_constant(_: str) -> Any:
    raise GuardVerificationError("non-finite JSON constants are not allowed")


def _validate_safe_record(
    record: dict[str, Any], context: ExpectedGuardContext, outer_exit_code: int
) -> int:
    field_names = frozenset(record)
    if field_names != _REQUIRED_FIELDS:
        missing = sorted(_REQUIRED_FIELDS - field_names)
        unexpected = sorted(field_names - _REQUIRED_FIELDS)
        detail: list[str] = []
        if missing:
            detail.append("missing=" + ",".join(missing))
        if unexpected:
            detail.append("unexpected=" + ",".join(unexpected))
        raise GuardVerificationError("guard result schema mismatch: " + "; ".join(detail))

    started = _parse_timestamp(record["started_utc"], "started_utc")
    finished = _parse_timestamp(record["finished_utc"], "finished_utc")
    if finished < started:
        raise GuardVerificationError("finished_utc precedes started_utc")

    status = _require_text(record["status"], "status", 64, allow_empty=False)
    _require_text(record["reason"], "reason", MAX_REASON_CHARACTERS, allow_empty=False)
    launcher_exit_code = _require_uint32(
        record["launcher_exit_code"], "launcher_exit_code"
    )
    child_exit_code = _require_uint32(record["child_exit_code"], "child_exit_code")
    child_process_id = _require_uint32(record["child_process_id"], "child_process_id")
    job_memory_limit = _require_uint64(
        record["job_memory_limit_bytes"], "job_memory_limit_bytes"
    )
    peak_job_memory = _require_uint64(
        record["peak_job_memory_bytes"], "peak_job_memory_bytes"
    )
    total_processes = _require_uint32(record["total_processes"], "total_processes")
    active_processes = _require_uint32(record["active_processes"], "active_processes")
    total_terminated = _require_uint32(
        record["total_terminated_processes"], "total_terminated_processes"
    )
    minimum_commit = _require_uint64(
        record["minimum_host_commit_free_bytes"], "minimum_host_commit_free_bytes"
    )
    minimum_physical = _require_uint64(
        record["minimum_host_physical_available_bytes"],
        "minimum_host_physical_available_bytes",
    )
    working_directory = _require_text(
        record["working_directory"],
        "working_directory",
        MAX_TEXT_CHARACTERS,
        allow_empty=False,
    )
    command = _freeze_command(record["command"], "command")

    # These are literal comparisons after one strict JSON decode.  In
    # particular, do not replace path separators or collapse doubled slashes.
    if command != context.command:
        raise GuardVerificationError("guard command does not match the trusted command")
    if working_directory != context.working_directory:
        raise GuardVerificationError(
            "guard working_directory does not match the trusted working directory"
        )

    if job_memory_limit != JOB_MEMORY_LIMIT_BYTES:
        raise GuardVerificationError("guard Job memory limit is not the fixed 6144 MiB cap")
    if peak_job_memory > job_memory_limit:
        raise GuardVerificationError("guard peak Job memory exceeds its cap")
    if total_processes < 1 or total_processes > MAX_TOTAL_PROCESSES:
        raise GuardVerificationError("guard total process count is outside 1..16")
    if active_processes != 0:
        raise GuardVerificationError("guard still reports active Job processes")
    if total_terminated != 0:
        raise GuardVerificationError("guard reports terminated Job processes")
    if child_process_id == 0:
        raise GuardVerificationError("ordinary guarded run has no child process id")
    if minimum_commit < MINIMUM_HOST_COMMIT_FREE_BYTES:
        raise GuardVerificationError("guard minimum host commit reserve is too low")
    if minimum_physical < MINIMUM_HOST_PHYSICAL_AVAILABLE_BYTES:
        raise GuardVerificationError("guard minimum host physical reserve is too low")

    if status == "completed":
        if child_exit_code != 0:
            raise GuardVerificationError("completed guard result has a non-zero child exit")
        if launcher_exit_code != 0 or outer_exit_code != 0:
            raise GuardVerificationError("completed guard result must have outer exit code zero")
    elif status == "child_failed":
        if child_exit_code == 0:
            raise GuardVerificationError("child_failed guard result has zero child exit")
        if launcher_exit_code != 1 or outer_exit_code != 1:
            raise GuardVerificationError("child_failed guard result must have outer exit code one")
    else:
        raise GuardVerificationError(f"unsafe guard status {status!r}")

    return child_exit_code


def _freeze_command(value: object, field_name: str) -> tuple[str, ...]:
    if (
        isinstance(value, (str, bytes, bytearray))
        or not isinstance(value, Sequence)
    ):
        raise GuardVerificationError(f"{field_name} must be a sequence of strings")
    if not value:
        raise GuardVerificationError(f"{field_name} is empty")
    if len(value) > MAX_COMMAND_ARGUMENTS:
        raise GuardVerificationError(
            f"{field_name} has more than {MAX_COMMAND_ARGUMENTS} arguments"
        )

    command = tuple(
        _require_text(item, f"{field_name}[{index}]", MAX_TEXT_CHARACTERS)
        for index, item in enumerate(value)
    )
    if not command[0]:
        raise GuardVerificationError(f"{field_name}[0] is empty")
    if sum(len(item) for item in command) > MAX_COMMAND_TEXT_CHARACTERS:
        raise GuardVerificationError(f"{field_name} text is too large")
    return command


def _require_text(
    value: object, field_name: str, maximum_characters: int, *, allow_empty: bool = True
) -> str:
    if not isinstance(value, str):
        raise GuardVerificationError(f"{field_name} must be a string")
    if not allow_empty and not value:
        raise GuardVerificationError(f"{field_name} is empty")
    if len(value) > maximum_characters:
        raise GuardVerificationError(f"{field_name} exceeds its text bound")
    if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
        raise GuardVerificationError(f"{field_name} contains an unpaired surrogate")
    return value


def _require_uint32(value: object, field_name: str) -> int:
    return _require_bounded_integer(value, field_name, _UINT32_MAX)


def _require_uint64(value: object, field_name: str) -> int:
    return _require_bounded_integer(value, field_name, _UINT64_MAX)


def _require_bounded_integer(value: object, field_name: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise GuardVerificationError(f"{field_name} must be an integer")
    if value < 0 or value > maximum:
        raise GuardVerificationError(f"{field_name} is outside its unsigned range")
    return value


def _parse_timestamp(value: object, field_name: str) -> tuple[int, int, int, int, int, int, int]:
    text = _require_text(value, field_name, 28, allow_empty=False)
    match = _TIMESTAMP_RE.fullmatch(text)
    if match is None:
        raise GuardVerificationError(
            f"{field_name} is not a UTC round-trip timestamp with seven fractions"
        )
    parts = tuple(int(group) for group in match.groups())
    year, month, day, hour, minute, second, fractional_100ns = parts
    try:
        # datetime supplies calendar and clock validation.  The seventh digit
        # remains in the returned tuple so ordering retains 100 ns precision.
        datetime(year, month, day, hour, minute, second, fractional_100ns // 10)
    except ValueError as error:
        raise GuardVerificationError(f"{field_name} is not a real UTC timestamp") from error
    return parts


def _structurally_equal(left: object, right: object) -> bool:
    """Compare decoded JSON without Python's bool/int equality shortcut."""

    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return (
            len(left) == len(right)
            and left.keys() == right.keys()
            and all(_structurally_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _structurally_equal(first, second) for first, second in zip(left, right)
        )
    return left == right


__all__ = [
    "ExpectedGuardContext",
    "GuardVerificationError",
    "JOB_MEMORY_LIMIT_BYTES",
    "MAX_RECORD_BYTES",
    "MAX_TOTAL_PROCESSES",
    "MINIMUM_HOST_COMMIT_FREE_BYTES",
    "MINIMUM_HOST_PHYSICAL_AVAILABLE_BYTES",
    "verify_guard_exit",
]
