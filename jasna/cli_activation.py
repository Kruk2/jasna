"""The activation options of the command line: --license-code, --online, --activation-status and the
license code environment variable."""

import argparse
import logging
import os
import sys
from collections.abc import Callable


LICENSE_CODE_ENVIRONMENT_VARIABLE = "JASNA_LICENSE_CODE"


def license_code_to_activate(flag_code: str | None, environment_code: str, activated: Callable[[], bool]) -> str | None:
    """--license-code always activates; the environment variable only while nothing is activated, so a code kept there
    never re-activates a PC on later runs."""
    if flag_code is not None:
        return flag_code
    if not environment_code.strip() or activated():
        return None
    return environment_code.strip()


def _require_protection_for_activation(parser: argparse.ArgumentParser) -> None:
    from jasna.license_api import ProtectionUnavailableError, require_protection

    try:
        require_protection()
    except ProtectionUnavailableError as exc:
        parser.exit(1, f"Activation error: {exc}\n")


def _saved_activation_exists(parser: argparse.ArgumentParser) -> bool:
    _require_protection_for_activation(parser)
    from jasna.protection import lease

    return lease.activated()


def _print_activation_status(parser: argparse.ArgumentParser) -> None:
    from jasna.license_status import device_use_text, license_status_key, read_license_info
    from jasna.gui.locales import t

    try:
        licensed, mode = read_license_info()
    except Exception as exc:  # ProtectionError or OSError; not imported here, jasna.protection may be missing
        parser.exit(1, f"Could not read the activation: {exc}\n")
    if not licensed:
        parser.exit(1, t("license_problem_no_activation") + "\n")
    print(t(license_status_key(mode)))
    from jasna.protection import ProtectionError, lease

    try:
        use = lease.device_count()
    except (ProtectionError, OSError) as exc:
        print(f"Could not check devices in use: {exc}", file=sys.stderr)
        return
    print(device_use_text(use))


def run_activation_options(
    parser: argparse.ArgumentParser, args: argparse.Namespace, configure_logging: Callable[[int], None]
) -> bool:
    """Activates with --license-code or the environment variable and prints --activation-status; True when
    nothing else was asked for."""
    environment_code = os.environ.get(LICENSE_CODE_ENVIRONMENT_VARIABLE, "")
    if args.online and args.license_code is None and not environment_code.strip():
        parser.error(f"--online needs --license-code or {LICENSE_CODE_ENVIRONMENT_VARIABLE}")
    if args.activation_status and (args.input is not None or args.stream or args.benchmark):
        parser.error("--activation-status cannot be combined with --input, --stream or --benchmark")
    license_code = license_code_to_activate(args.license_code, environment_code, lambda: _saved_activation_exists(parser))
    if license_code is not None:
        _require_protection_for_activation(parser)
        from jasna.license_api import typed_license_code
        from jasna.protection import ProtectionError, lease
        if sys.platform == "win32" and not args.online:
            print("Windows may ask for permission once. This is a one-time license check for offline use.")
        # Activation warnings always show, as they did before logging was configured; --log-level can add more.
        configure_logging(min(getattr(logging, args.log_level.upper()), logging.WARNING))
        try:
            result = lease.activate_license_code(typed_license_code(license_code), try_offline=not args.online)
        except (ProtectionError, OSError) as exc:
            parser.exit(1, f"Activation error: {exc}\n")
        print(lease.activated_message(result.online_reason))
        if args.input is None and not args.stream and not args.benchmark and not args.activation_status:
            return True
    if args.activation_status:
        _print_activation_status(parser)
        return True
    return False
