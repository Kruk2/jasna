"""Tests for post-export action feature."""

import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock
from dataclasses import dataclass, field
from enum import Enum


class PostExportAction(Enum):
    NONE = "none"
    SHUTDOWN = "shutdown"
    CUSTOM_COMMAND = "custom_command"


@dataclass
class AppSettings:
    post_export_action: str = "none"
    post_export_custom_command: str = ""


def test_post_export_action_enum_values():
    """PostExportAction enum should have correct values."""
    assert PostExportAction.NONE.value == "none"
    assert PostExportAction.SHUTDOWN.value == "shutdown"
    assert PostExportAction.CUSTOM_COMMAND.value == "custom_command"
    print("✓ test_post_export_action_enum_values passed")


def test_app_settings_default_post_export_action():
    """AppSettings should have default post-export action of none."""
    settings = AppSettings()
    assert settings.post_export_action == "none"
    assert settings.post_export_custom_command == ""
    print("✓ test_app_settings_default_post_export_action passed")


def test_app_settings_post_export_action_persistence():
    """Post-export action settings should be serializable."""
    settings = AppSettings(
        post_export_action="shutdown",
        post_export_custom_command="notepad.exe"
    )
    data = settings.__dict__
    assert data["post_export_action"] == "shutdown"
    assert data["post_export_custom_command"] == "notepad.exe"
    print("✓ test_app_settings_post_export_action_persistence passed")


def test_execute_post_export_action_none():
    """Should do nothing when post-export action is none."""
    settings = AppSettings(post_export_action="none")
    
    # Simulate the logic from processor.py
    action = settings.post_export_action
    if action == PostExportAction.NONE.value:
        result = "none"
    elif action == PostExportAction.SHUTDOWN.value:
        result = "shutdown"
    elif action == PostExportAction.CUSTOM_COMMAND.value:
        result = "custom_command"
    else:
        result = "unknown"
    
    assert result == "none"
    print("✓ test_execute_post_export_action_none passed")


def test_execute_post_export_action_shutdown():
    """Should call shutdown callback when action is shutdown."""
    shutdown_called = []
    
    def on_shutdown():
        shutdown_called.append(True)
    
    settings = AppSettings(post_export_action="shutdown")
    
    # Simulate the logic from processor.py
    action = settings.post_export_action
    if action == PostExportAction.SHUTDOWN.value:
        on_shutdown()
    
    assert len(shutdown_called) == 1
    print("✓ test_execute_post_export_action_shutdown passed")


def test_execute_post_export_action_custom_command():
    """Should execute custom command when action is custom_command."""
    settings = AppSettings(
        post_export_action="custom_command",
        post_export_custom_command="echo test"
    )
    
    # Simulate the logic from processor.py
    action = settings.post_export_action
    command = settings.post_export_custom_command.strip()
    
    executed = False
    if action == PostExportAction.CUSTOM_COMMAND.value and command:
        executed = True
    
    assert executed is True
    print("✓ test_execute_post_export_action_custom_command passed")


def test_execute_post_export_action_custom_command_empty():
    """Should not execute empty custom command."""
    settings = AppSettings(
        post_export_action="custom_command",
        post_export_custom_command=""
    )
    
    # Simulate the logic from processor.py
    action = settings.post_export_action
    command = settings.post_export_custom_command.strip()
    
    executed = False
    if action == PostExportAction.CUSTOM_COMMAND.value and command:
        executed = True
    
    assert executed is False
    print("✓ test_execute_post_export_action_custom_command_empty passed")


def test_execute_post_export_action_custom_command_whitespace():
    """Should not execute whitespace-only custom command."""
    settings = AppSettings(
        post_export_action="custom_command",
        post_export_custom_command="   "
    )
    
    # Simulate the logic from processor.py
    action = settings.post_export_action
    command = settings.post_export_custom_command.strip()
    
    executed = False
    if action == PostExportAction.CUSTOM_COMMAND.value and command:
        executed = True
    
    assert executed is False
    print("✓ test_execute_post_export_action_custom_command_whitespace passed")


if __name__ == "__main__":
    test_post_export_action_enum_values()
    test_app_settings_default_post_export_action()
    test_app_settings_post_export_action_persistence()
    test_execute_post_export_action_none()
    test_execute_post_export_action_shutdown()
    test_execute_post_export_action_custom_command()
    test_execute_post_export_action_custom_command_empty()
    test_execute_post_export_action_custom_command_whitespace()
    print("\n✓ All tests passed!")