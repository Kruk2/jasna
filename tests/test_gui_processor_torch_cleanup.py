import types


def test_cleanup_torch_calls_cuda_hooks_when_available():
    from jasna.gui.processor import _cleanup_torch

    calls: list[str] = []

    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            calls.append("is_available")
            return True

        @staticmethod
        def synchronize() -> None:
            calls.append("synchronize")

        @staticmethod
        def empty_cache() -> None:
            calls.append("empty_cache")

        @staticmethod
        def ipc_collect() -> None:
            calls.append("ipc_collect")

    fake_torch = types.SimpleNamespace(cuda=_Cuda)

    _cleanup_torch(fake_torch)

    assert calls == ["is_available", "synchronize", "empty_cache", "ipc_collect"]


def test_cleanup_torch_noops_when_cuda_unavailable():
    from jasna.gui.processor import _cleanup_torch

    calls: list[str] = []

    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            calls.append("is_available")
            return False

        @staticmethod
        def synchronize() -> None:
            calls.append("synchronize")

        @staticmethod
        def empty_cache() -> None:
            calls.append("empty_cache")

        @staticmethod
        def ipc_collect() -> None:
            calls.append("ipc_collect")

    fake_torch = types.SimpleNamespace(cuda=_Cuda)

    _cleanup_torch(fake_torch)

    assert calls == ["is_available"]

