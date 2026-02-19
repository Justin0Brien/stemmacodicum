from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


def ensure_qdrant_server(
    *,
    server_url: str,
    default_storage_dir: Path,
) -> None:
    if not _is_local_qdrant_url(server_url):
        return
    if not _read_bool_env("STEMMA_QDRANT_AUTO_START", True):
        return
    if _is_healthy(server_url):
        return

    container_name = os.getenv("STEMMA_QDRANT_DOCKER_CONTAINER_NAME", "stemma-qdrant").strip() or "stemma-qdrant"
    image = os.getenv("STEMMA_QDRANT_DOCKER_IMAGE", "qdrant/qdrant:v1.16.2").strip() or "qdrant/qdrant:v1.16.2"
    storage_dir = (
        Path(os.getenv("STEMMA_QDRANT_DOCKER_STORAGE_DIR", "")).expanduser().resolve()
        if os.getenv("STEMMA_QDRANT_DOCKER_STORAGE_DIR")
        else default_storage_dir
    )
    timeout_seconds = _read_float_env("STEMMA_QDRANT_DOCKER_START_TIMEOUT_SECONDS", 120.0)

    parsed = urllib.parse.urlparse(server_url)
    http_port = parsed.port or (443 if parsed.scheme == "https" else 80)
    grpc_port = _read_int_env("STEMMA_QDRANT_DOCKER_GRPC_PORT", http_port + 1)

    _ensure_docker_daemon_ready()

    _ensure_container_running(
        container_name=container_name,
        image=image,
        storage_dir=storage_dir,
        http_port=http_port,
        grpc_port=grpc_port,
    )
    _wait_healthy(server_url, timeout_seconds)


def _ensure_container_running(
    *,
    container_name: str,
    image: str,
    storage_dir: Path,
    http_port: int,
    grpc_port: int,
) -> None:
    inspect = _run(["docker", "inspect", container_name], check=False)
    if inspect.returncode != 0:
        storage_dir.mkdir(parents=True, exist_ok=True)
        launch = _run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                container_name,
                "--restart",
                "unless-stopped",
                "-p",
                f"{http_port}:6333",
                "-p",
                f"{grpc_port}:6334",
                "-v",
                f"{storage_dir}:/qdrant/storage",
                image,
            ],
            check=False,
        )
        if launch.returncode == 0:
            return

        launch_err = ((launch.stderr or "") + " " + (launch.stdout or "")).strip()
        launch_err_lc = launch_err.lower()
        if "already in use by container" in launch_err_lc:
            _run(["docker", "start", container_name], check=True)
            _run(["docker", "update", "--restart", "unless-stopped", container_name], check=False)
            return
        if "port is already allocated" in launch_err_lc:
            # Another process/container may already own the target ports.
            # Caller will verify readiness via _wait_healthy.
            return
        raise RuntimeError(
            f"Failed to launch Qdrant container '{container_name}': "
            f"{launch_err or 'unknown docker error'}"
        )
        return

    state = _run(["docker", "inspect", "--format", "{{.State.Running}}", container_name], check=True)
    running = (state.stdout or "").strip().lower() == "true"
    if not running:
        _run(["docker", "start", container_name], check=True)
    _run(["docker", "update", "--restart", "unless-stopped", container_name], check=False)


def _wait_healthy(server_url: str, timeout_seconds: float) -> None:
    deadline = time.time() + max(1.0, timeout_seconds)
    while time.time() < deadline:
        if _is_healthy(server_url):
            return
        time.sleep(1.0)
    raise RuntimeError(
        "Qdrant server did not become healthy within timeout "
        f"({timeout_seconds:.1f}s): {server_url}"
    )


def _ensure_docker_daemon_ready() -> None:
    info = _run(["docker", "info"], check=False)
    if info.returncode == 0:
        return

    if sys.platform != "darwin":
        return
    if not _read_bool_env("STEMMA_QDRANT_AUTO_START_DOCKER_DESKTOP", True):
        return

    _open_docker_desktop()
    timeout_seconds = _read_float_env("STEMMA_QDRANT_DOCKER_DESKTOP_START_TIMEOUT_SECONDS", 120.0)
    deadline = time.time() + max(5.0, timeout_seconds)
    last_err = (info.stderr or info.stdout or "").strip()
    while time.time() < deadline:
        current = _run(["docker", "info"], check=False)
        if current.returncode == 0:
            return
        last_err = (current.stderr or current.stdout or last_err).strip()
        time.sleep(1.0)

    raise RuntimeError(
        "Docker daemon is not ready after attempting to launch Docker Desktop. "
        f"Last error: {last_err or 'unknown docker info error'}"
    )


def _open_docker_desktop() -> None:
    launch = _run(["open", "-a", "Docker"], check=False)
    if launch.returncode == 0:
        return
    launch_alt = _run(["open", "-a", "Docker Desktop"], check=False)
    if launch_alt.returncode == 0:
        return
    details = (
        (launch.stderr or launch.stdout or "").strip()
        or (launch_alt.stderr or launch_alt.stdout or "").strip()
    )
    raise RuntimeError(f"Unable to launch Docker Desktop automatically. {details}")


def _is_healthy(server_url: str) -> bool:
    parsed = urllib.parse.urlparse(server_url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    request = urllib.request.Request(f"{base}/healthz", method="GET")
    try:
        with urllib.request.urlopen(request, timeout=2.0) as response:
            return 200 <= int(response.status) < 300
    except (urllib.error.URLError, TimeoutError, ValueError):
        return False


def _is_local_qdrant_url(server_url: str) -> bool:
    parsed = urllib.parse.urlparse(server_url)
    host = (parsed.hostname or "").lower()
    return host in {"127.0.0.1", "localhost", "::1"}


def _run(cmd: list[str], *, check: bool) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=check, text=True, capture_output=True)


def _read_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _read_float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _read_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default
