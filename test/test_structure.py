"""
test/test_structure.py — structural smoke tests for the amodal package.

These run WITHOUT a GPU. Heavy third-party deps (torch, cv2, diffusers, …) and
the OpenAI client are stubbed so we can verify the package *wires together*:
every module imports, the LangGraph compiles, and the three nodes + State are
importable. They do NOT exercise real model inference — that needs a GPU and is
covered by the end-to-end run (client.py / server.py).

Run:  pytest test/   (from the repo root)
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def _install_stubs():
    """Install minimal stub modules for heavy deps absent in a CPU-only CI env.
    A real env with the deps installed skips each stub (only stubs what's
    missing)."""
    # OpenAI: avoid real client construction / network.
    if "openai" not in sys.modules:
        openai = types.ModuleType("openai")
        openai.OpenAI = lambda *a, **k: types.SimpleNamespace(responses=None)
        sys.modules["openai"] = openai

    # dotenv
    if "dotenv" not in sys.modules:
        dotenv = types.ModuleType("dotenv")
        dotenv.load_dotenv = lambda *a, **k: None
        sys.modules["dotenv"] = dotenv

    # torch (only the few attributes runtime/models touch at import time).
    if "torch" not in sys.modules:
        torch = types.ModuleType("torch")
        torch.cuda = types.SimpleNamespace(
            is_available=lambda: False,
            get_device_properties=lambda i: types.SimpleNamespace(total_memory=0),
            get_device_name=lambda i: "cpu",
            empty_cache=lambda: None,
            synchronize=lambda: None,
            set_per_process_memory_fraction=lambda *a, **k: None,
            mem_get_info=lambda: (0, 0),
            ipc_collect=lambda: None,
        )
        torch.backends = types.SimpleNamespace(
            cudnn=types.SimpleNamespace(benchmark=False, allow_tf32=False),
            cuda=types.SimpleNamespace(matmul=types.SimpleNamespace(allow_tf32=False)),
        )
        torch.bfloat16 = "bfloat16"
        torch.set_float32_matmul_precision = lambda *a, **k: None
        torch.nn = types.SimpleNamespace(Module=type("Module", (), {}))
        torch.no_grad = lambda: types.SimpleNamespace(
            __enter__=lambda s: None, __exit__=lambda s, *a: False)
        torch.Generator = lambda *a, **k: types.SimpleNamespace(manual_seed=lambda s: None)
        sys.modules["torch"] = torch

    for name in ("cv2",):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)

    # numpy is a hard requirement for the type hints; if it's genuinely missing
    # we skip the whole module (nothing meaningful to test).
    try:
        import numpy  # noqa: F401
    except ImportError:  # pragma: no cover
        pytest.skip("numpy not installed", allow_module_level=True)


_install_stubs()


def test_config_dispatch_keys():
    import config
    assert config.RUN_ENV in {"local", "lightning", "colab"}
    for key in ("LIGHTNING_URL", "COLAB_URL", "INPAINT_BACKEND",
                "FLUX_FILL_MODEL_ID", "SAM3_MODEL_ID", "ALL_FAILURE_CODES"):
        assert hasattr(config, key), f"config missing {key}"


def test_client_env_resolution(monkeypatch):
    import config
    import client
    monkeypatch.setattr(config, "RUN_ENV", "colab", raising=False)
    assert client._resolve_run_env() == "colab"
    monkeypatch.setattr(config, "RUN_ENV", "", raising=False)
    monkeypatch.setattr(config, "USE_LIGHTNING", True, raising=False)
    assert client._resolve_run_env() == "lightning"
    monkeypatch.setattr(config, "USE_LIGHTNING", False, raising=False)
    assert client._resolve_run_env() == "local"


def test_state_importable():
    from pipeline.state import State
    assert "image_path" in State.__annotations__
    assert "output_path" in State.__annotations__


@pytest.mark.skipif("langgraph" not in sys.modules
                    and __import__("importlib").util.find_spec("langgraph") is None,
                    reason="langgraph not installed")
def test_graph_compiles():
    """The full graph wires occlusion → completion → reviewer and compiles.
    Requires langgraph; node bodies are not executed here."""
    from pipeline.graph import build_graph, route
    g = build_graph()
    assert g is not None
    # route() terminates on an accepted score.
    assert route({"review_score": 9.0, "attempt": 0}) == "end"
