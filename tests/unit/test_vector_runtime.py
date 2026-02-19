import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from stemmacodicum.infrastructure.vector import server_runtime
from stemmacodicum.infrastructure.vector.qdrant_store import QdrantLocalStore


def test_ensure_container_running_recovers_name_conflict(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], *, check: bool) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        if cmd[:2] == ["docker", "inspect"] and len(cmd) == 3:
            return subprocess.CompletedProcess(cmd, 1, "", "Error: No such object")
        if cmd[:2] == ["docker", "run"]:
            return subprocess.CompletedProcess(
                cmd,
                1,
                "",
                'Conflict. The container name "/stemma-qdrant" is already in use by container',
            )
        if cmd[:2] == ["docker", "start"]:
            return subprocess.CompletedProcess(cmd, 0, "stemma-qdrant", "")
        if cmd[:2] == ["docker", "update"]:
            return subprocess.CompletedProcess(cmd, 0, "", "")
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(server_runtime, "_run", fake_run)
    server_runtime._ensure_container_running(
        container_name="stemma-qdrant",
        image="qdrant/qdrant:v1.16.2",
        storage_dir=tmp_path / "qdrant-server-data",
        http_port=6333,
        grpc_port=6334,
    )

    assert any(cmd[:2] == ["docker", "start"] for cmd in calls)


def test_ensure_container_running_surfaces_docker_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_run(cmd: list[str], *, check: bool) -> subprocess.CompletedProcess[str]:
        if cmd[:2] == ["docker", "inspect"] and len(cmd) == 3:
            return subprocess.CompletedProcess(cmd, 1, "", "Error: No such object")
        if cmd[:2] == ["docker", "run"]:
            return subprocess.CompletedProcess(
                cmd,
                1,
                "",
                "Cannot connect to the Docker daemon at unix:///var/run/docker.sock.",
            )
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(server_runtime, "_run", fake_run)
    with pytest.raises(RuntimeError, match="Cannot connect to the Docker daemon"):
        server_runtime._ensure_container_running(
            container_name="stemma-qdrant",
            image="qdrant/qdrant:v1.16.2",
            storage_dir=tmp_path / "qdrant-server-data",
            http_port=6333,
            grpc_port=6334,
        )


def test_ensure_qdrant_server_auto_starts_docker_desktop_on_macos(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[list[str]] = []
    state = {"opened": False, "docker_info_checks": 0}

    def fake_run(cmd: list[str], *, check: bool) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        if cmd[:2] == ["docker", "info"]:
            state["docker_info_checks"] += 1
            if state["opened"] and state["docker_info_checks"] >= 2:
                return subprocess.CompletedProcess(cmd, 0, "ok", "")
            return subprocess.CompletedProcess(cmd, 1, "", "docker daemon not running")
        if cmd[:3] == ["open", "-a", "Docker"]:
            state["opened"] = True
            return subprocess.CompletedProcess(cmd, 0, "", "")
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(server_runtime, "_run", fake_run)
    monkeypatch.setattr(server_runtime, "_is_healthy", lambda _url: False)
    monkeypatch.setattr(server_runtime, "_ensure_container_running", lambda **_kwargs: None)
    monkeypatch.setattr(server_runtime, "_wait_healthy", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server_runtime.sys, "platform", "darwin")
    monkeypatch.setattr(server_runtime.time, "sleep", lambda _sec: None)

    server_runtime.ensure_qdrant_server(
        server_url="http://127.0.0.1:6333",
        default_storage_dir=tmp_path / "qdrant-server-data",
    )

    assert ["open", "-a", "Docker"] in calls
    assert state["docker_info_checks"] >= 2


def test_qdrant_store_falls_back_to_local_when_server_start_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import stemmacodicum.infrastructure.vector.qdrant_store as qstore

    class DummyQdrantClient:
        def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
            self.args = args
            self.kwargs = kwargs

    fake_qdrant_client = types.ModuleType("qdrant_client")
    fake_qdrant_client.QdrantClient = DummyQdrantClient
    fake_qdrant_http = types.ModuleType("qdrant_client.http")
    fake_qdrant_models = types.ModuleType("qdrant_client.http.models")
    fake_qdrant_http.models = fake_qdrant_models

    monkeypatch.setitem(sys.modules, "qdrant_client", fake_qdrant_client)
    monkeypatch.setitem(sys.modules, "qdrant_client.http", fake_qdrant_http)
    monkeypatch.setitem(sys.modules, "qdrant_client.http.models", fake_qdrant_models)

    def fail_start(**kwargs) -> None:  # noqa: ANN003
        raise RuntimeError("docker run failed")

    monkeypatch.setattr(qstore, "ensure_qdrant_server", fail_start)

    storage = tmp_path / "qdrant"
    store = QdrantLocalStore(storage_path=storage, url="http://127.0.0.1:6333")
    client, models = store._client_and_models()

    assert isinstance(client, DummyQdrantClient)
    assert client.kwargs.get("path") == str(storage)
    assert models is fake_qdrant_models
    assert store.backend_name == "qdrant-local-fallback"
    assert store.server_url is None


def test_qdrant_store_uses_isolated_path_when_primary_local_locked(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import stemmacodicum.infrastructure.vector.qdrant_store as qstore

    class DummyQdrantClient:
        def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
            self.args = args
            self.kwargs = kwargs
            path = kwargs.get("path")
            if path and path.endswith("/qdrant"):
                raise RuntimeError(
                    "Storage folder /tmp/qdrant is already accessed by another instance of Qdrant client."
                )

    fake_qdrant_client = types.ModuleType("qdrant_client")
    fake_qdrant_client.QdrantClient = DummyQdrantClient
    fake_qdrant_http = types.ModuleType("qdrant_client.http")
    fake_qdrant_models = types.ModuleType("qdrant_client.http.models")
    fake_qdrant_http.models = fake_qdrant_models

    monkeypatch.setitem(sys.modules, "qdrant_client", fake_qdrant_client)
    monkeypatch.setitem(sys.modules, "qdrant_client.http", fake_qdrant_http)
    monkeypatch.setitem(sys.modules, "qdrant_client.http.models", fake_qdrant_models)

    storage = tmp_path / "qdrant"
    store = QdrantLocalStore(storage_path=storage, url=None)
    client, models = store._client_and_models()

    assert isinstance(client, DummyQdrantClient)
    assert "/qdrant-isolated/" in str(client.kwargs.get("path"))
    assert str(store.storage_path).endswith(f"/qdrant-isolated/pid-{os.getpid()}")
    assert store.backend_name == "qdrant-local-isolated"
    assert models is fake_qdrant_models
