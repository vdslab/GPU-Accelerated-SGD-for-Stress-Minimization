#!/usr/bin/env python3
"""Serve the single-run page and its local experiment API (Python stdlib only)."""

from __future__ import annotations

import argparse
import json
import math
import mimetypes
import os
import re
import signal
import subprocess
import sys
import threading
import uuid
import webbrowser
from dataclasses import dataclass, field
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlsplit

try:
    from artifact_paths import resolve_artifact_path
    from experiment_plan import METHODS, REPO_ROOT
except ModuleNotFoundError:
    from experiments.artifact_paths import resolve_artifact_path
    from experiments.experiment_plan import METHODS, REPO_ROOT

DEFAULTS = {
    "graph": "data/dwt_419.mtx",
    "method": "rr_sparse_sgd",
    "iterations": 15,
    "epsilon": 0.1,
    "pivots": 200,
    "seed": "0",
    "image_size": "2048x2048",
}
METHOD_LABELS = {
    "sgd": "SGD (CPU / full)",
    "atomic_sgd": "Atomic SGD (GPU / full)",
    "rr_sgd": "RR SGD (GPU / full)",
    "sparse_sgd": "Sparse SGD (CPU)",
    "atomic_sparse_sgd": "Atomic Sparse SGD (GPU)",
    "rr_sparse_sgd": "RR Sparse SGD (GPU)",
}
LOG_LIMIT = 32_768
BODY_LIMIT = 16_384
LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}


class RequestError(ValueError):
    pass


class BusyError(RuntimeError):
    pass


def loopback_url(value: str, *, host_header: bool = False) -> bool:
    """Reject non-loopback hosts, userinfo, malformed ports, and URL extras."""
    if any(character.isspace() for character in value):
        return False
    try:
        parsed = urlsplit("http://" + value if host_header else value)
        if parsed.scheme not in {"http", "https"} or parsed.hostname not in LOOPBACK_HOSTS:
            return False
        if parsed.username is not None or parsed.password is not None:
            return False
        if parsed.path or parsed.query or parsed.fragment or not parsed.netloc:
            return False
        port = parsed.port  # Raises ValueError for malformed or out-of-range ports.
        if parsed.netloc.endswith(":") or (port is not None and not 1 <= port <= 65535):
            return False
        return True
    except ValueError:
        return False


def _read_graph_statistics(path: Path) -> tuple[int, int, int]:
    """Count input edges as the sparse loader does, before component selection."""
    with path.open(encoding="utf-8-sig") as source:
        header = source.readline(8193).strip().lower().split()
        if header[:3] != ["%%matrixmarket", "matrix", "coordinate"]:
            raise ValueError("Expected a coordinate Matrix Market header")
        for _ in range(1024):
            line = source.readline(8193)
            if not line or len(line) > 8192:
                raise ValueError("Missing or invalid Matrix Market dimensions")
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            rows, columns, entries = map(int, line.split())
            if rows != columns or rows <= 0 or entries < 0:
                raise ValueError("Expected a nonempty square adjacency matrix")
            break
        else:
            raise ValueError("Missing Matrix Market dimensions")

        unique_edges: set[int] = set()
        actual_entries = 0
        while line := source.readline(8193):
            if len(line) > 8192:
                raise ValueError("Invalid Matrix Market coordinate row")
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            # Numeric matrix values are ignored by the experiment's graph loader.
            parts = line.split(None, 2)
            row, column = int(parts[0]), int(parts[1])
            if not (1 <= row <= rows and 1 <= column <= rows):
                raise ValueError("Matrix Market coordinate is outside the matrix")
            actual_entries += 1
            if actual_entries > entries:
                raise ValueError("More coordinates than declared Matrix Market entries")
            if row != column:
                u, v = (row - 1, column - 1) if row < column else (column - 1, row - 1)
                unique_edges.add(u * rows + v)
        if actual_entries != entries:
            raise ValueError("Incomplete Matrix Market coordinates")
        return rows, entries, len(unique_edges)


@lru_cache(maxsize=256)
def _graph_statistics(path: Path, modified_ns: int, size: int) -> tuple[int, int, int] | None:
    """Keep only numeric stats, keyed by file identity and its change signature."""
    try:
        return _read_graph_statistics(path)
    except (OSError, UnicodeError, ValueError, IndexError):
        return None


def graph_catalog(root: Path) -> list[dict[str, Any]]:
    """Return graph names and cached input node/edge counts."""
    root = root.resolve()
    data_root = (root / "data").resolve()
    graphs = []
    for candidate in sorted(data_root.rglob("*.mtx")):
        try:
            path = candidate.resolve()
            if not path.is_relative_to(data_root) or not path.is_relative_to(root) or not path.is_file():
                continue
            stat = path.stat()
            statistics = _graph_statistics(path, stat.st_mtime_ns, stat.st_size)
            if statistics is not None:
                nodes, entries, edges = statistics
                relative = candidate.relative_to(root).as_posix()
                graphs.append({"path": relative, "name": candidate.stem, "nodes": nodes, "entries": entries, "edges": edges})
        except (OSError, UnicodeError, ValueError):
            continue
    return graphs


def graph_version_membership(root: Path) -> tuple[list[str], dict[Path, set[str]]]:
    """Read version input manifests without mistaking source graphs for inputs."""
    root = root.resolve()
    results_root = root / "experiment_results"
    versions = {"v1", "v2", "v3"}
    membership: dict[Path, set[str]] = {}
    try:
        folders = sorted(results_root.iterdir())
    except OSError:
        folders = []
    for folder in folders:
        if not re.fullmatch(r"v[1-9][0-9]*", folder.name) or not folder.is_dir():
            continue
        if not folder.resolve().is_relative_to(results_root.resolve()):
            continue
        version = folder.name
        versions.add(version)
        manifest_datasets = []
        catalog_datasets = []
        has_manifest = False
        for config_path in sorted((folder / "config").glob("*.json")):
            try:
                document = json.loads(config_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, ValueError):
                continue
            if isinstance(document, dict) and isinstance(document.get("datasets"), list):
                has_manifest = True
                manifest_datasets.extend(document["datasets"])
            elif config_path.name == "datasets.json" and isinstance(document, list):
                catalog_datasets.extend(document)
        # Prepared-but-unused catalog entries must not override an actual manifest.
        for dataset in manifest_datasets if has_manifest else catalog_datasets:
            if not isinstance(dataset, dict) or not isinstance(dataset.get("path"), str):
                continue
            try:
                path = resolve_artifact_path(dataset["path"], root).resolve()
                if path.is_relative_to(root / "data") and path.is_file():
                    membership.setdefault(path, set()).add(version)
            except (OSError, UnicodeError, ValueError, KeyError, TypeError):
                continue
    return sorted(versions, key=lambda version: int(version[1:])), membership


def unsigned_integer(value: Any, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise RequestError(f"{name}は整数で指定してください")
    if isinstance(value, str) and (len(value) > 20 or not re.fullmatch(r"[0-9]+", value)):
        raise RequestError(f"{name}は整数で指定してください")
    result = int(value)
    if not minimum <= result <= 2**64 - 1:
        raise RequestError(f"{name}は{minimum}以上、u64の範囲で指定してください")
    return result


def validate_settings(payload: Any, graphs: list[dict[str, Any]]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise RequestError("JSON objectで実験設定を指定してください")
    unknown = set(payload) - set(DEFAULTS)
    if unknown:
        raise RequestError("未対応の設定: " + ", ".join(sorted(unknown)))
    settings = {**DEFAULTS, **payload}
    if not isinstance(settings["graph"], str) or settings["graph"] not in {graph["path"] for graph in graphs}:
        raise RequestError("graphはグラフ一覧から選択してください")
    if not isinstance(settings["method"], str) or settings["method"] not in METHODS:
        raise RequestError("methodは手法一覧から選択してください")
    settings["iterations"] = unsigned_integer(settings["iterations"], "iterations")
    settings["pivots"] = unsigned_integer(settings["pivots"], "pivots")
    settings["seed"] = str(unsigned_integer(settings["seed"], "seed", minimum=0))
    epsilon = settings["epsilon"]
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float, str)):
        raise RequestError("epsilonは正の有限数で指定してください")
    try:
        settings["epsilon"] = float(epsilon)
    except (ValueError, OverflowError) as error:
        raise RequestError("epsilonは正の有限数で指定してください") from error
    if not math.isfinite(settings["epsilon"]) or settings["epsilon"] <= 0:
        raise RequestError("epsilonは正の有限数で指定してください")
    size = settings["image_size"]
    if not isinstance(size, str) or not re.fullmatch(r"[0-9]{1,4}[xX][0-9]{1,4}", size):
        raise RequestError("image_sizeはWxH形式で指定してください (例: 1024x1024)")
    dimensions = [int(piece) for piece in size.lower().split("x")]
    if any(not 1 <= number <= 8192 for number in dimensions):
        raise RequestError("image_sizeの幅と高さは1..8192です")
    settings["image_size"] = "x".join(map(str, dimensions))
    return settings


@dataclass
class Job:
    id: str
    settings: dict[str, Any]
    state: str = "running"
    log: str = ""
    run_directory: Path | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    process: subprocess.Popen[str] | None = field(default=None, repr=False)
    thread: threading.Thread | None = field(default=None, repr=False)


class ExperimentService:
    def __init__(self, root: Path = REPO_ROOT) -> None:
        self.root = root.resolve()
        self.output_root = (self.root / "output").resolve()
        if not self.output_root.is_relative_to(self.root):
            raise ValueError("outputはリポジトリ内のフォルダである必要があります")
        self.graphs = graph_catalog(self.root)
        self.jobs: dict[str, Job] = {}
        self.active_run: str | None = None
        self.lock = threading.RLock()
        self.stopping = False

    def refresh_graphs(self) -> list[dict[str, Any]]:
        graphs = graph_catalog(self.root)
        with self.lock:
            self.graphs = graphs
        return graphs

    def config(self) -> dict[str, Any]:
        graphs = self.refresh_graphs()
        versions, membership = graph_version_membership(self.root)
        graphs = [{**graph, "versions": [version for version in versions
                                        if version in membership.get((self.root / graph["path"]).resolve(), set())]}
                  for graph in graphs]
        defaults = DEFAULTS.copy()
        if defaults["graph"] not in {graph["path"] for graph in graphs}:
            defaults["graph"] = graphs[0]["path"] if graphs else None
        with self.lock:
            return {
                "graphs": graphs,
                "versions": versions,
                "methods": [{"id": name, "label": METHOD_LABELS[name], "sparse": spec.sparse} for name, spec in METHODS.items()],
                "defaults": defaults,
                "active_run": self.active_run,
            }

    def start(self, payload: Any) -> Job:
        settings = validate_settings(payload, self.refresh_graphs())
        graph_path = (self.root / settings["graph"]).resolve()
        if not graph_path.is_relative_to((self.root / "data").resolve()) or not graph_path.is_file():
            raise RequestError("選択したグラフがありません。ページを再読み込みしてください")
        with self.lock:
            if self.active_run is not None or self.stopping:
                raise BusyError("別の実験を実行中です。完了してから実行してください")
            job = Job(uuid.uuid4().hex, settings)
            self.jobs[job.id] = job
            self.active_run = job.id
            job.thread = threading.Thread(target=self.execute, args=(job,), daemon=True, name=f"experiment-{job.id[:8]}")
            job.thread.start()
            return job

    def command(self, job: Job) -> list[str]:
        settings = job.settings
        return [
            sys.executable, "-u", str(self.root / "experiments/run_single.py"),
            "--input", str(self.root / settings["graph"]),
            "--method", settings["method"],
            "--iterations", str(settings["iterations"]), "--epsilon", str(settings["epsilon"]),
            "--pivots", str(settings["pivots"]), "--seed", settings["seed"],
            "--image-size", settings["image_size"], "--output-root", str(self.output_root),
        ]

    def output_path(self, value: str) -> Path | None:
        path = Path(value)
        if not path.is_absolute():
            path = self.root / path
        path = path.resolve()
        return path if path.is_relative_to(self.output_root) and path != self.output_root else None

    def append_log(self, job: Job, line: str) -> None:
        with self.lock:
            job.log = (job.log + line)[-LOG_LIMIT:]
            if line.startswith("Output: "):
                job.run_directory = self.output_path(line[len("Output: "):].strip())

    def execute(self, job: Job) -> None:
        try:
            # A process group includes the runner, Cargo, SGD binary, and renderer.
            with self.lock:
                if self.stopping:
                    raise RuntimeError("ローカルサーバーを終了しました")
                job.process = subprocess.Popen(
                    self.command(job), cwd=self.root, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                    start_new_session=True,
                )
            assert job.process.stdout is not None
            for line in iter(lambda: job.process.stdout.readline(8192), ""):
                self.append_log(job, line)
            job.process.stdout.close()
            exit_code = job.process.wait()
            with self.lock:
                self.read_metadata(job)
                if exit_code == 0 and job.result and job.result.get("status") == "success":
                    job.state = "success"
                else:
                    job.state = "failure"
                    native_error = job.result.get("error") if job.result else None
                    if exit_code == 0:
                        job.error = native_error or "実験結果のsuccessメタデータがありません。保存先と実行ログを確認してください"
                    else:
                        job.error = native_error or f"実験が終了コード {exit_code} で失敗しました。実行ログを確認してください"
        except (OSError, ValueError, RuntimeError) as error:
            with self.lock:
                job.state = "failure"
                job.error = f"実験を実行できません: {error}"
                job.log = (job.log + "\n" + job.error + "\n")[-LOG_LIMIT:]
                self.read_metadata(job)
        finally:
            with self.lock:
                job.process = None
                if self.active_run == job.id:
                    self.active_run = None

    def read_metadata(self, job: Job) -> None:
        if job.run_directory is None:
            return
        try:
            candidates = sorted(job.run_directory.glob("*_result.json"))
            if len(candidates) == 1 and self.output_path(str(candidates[0])) is not None:
                metadata = json.loads(candidates[0].read_text(encoding="utf-8"))
                if isinstance(metadata, dict):
                    job.result = metadata
        except (OSError, ValueError):
            pass  # The process error/log remains usable even if metadata is incomplete.

    def artifact_urls(self, job: Job) -> dict[str, str | None]:
        urls: dict[str, str | None] = {name: None for name in ("info", "result", "before_image", "after_image")}
        artifacts = job.result.get("artifacts", {}) if job.result else {}
        if not isinstance(artifacts, dict):
            return urls
        for name in urls:
            value = artifacts.get(name)
            if not isinstance(value, str):
                continue
            path = self.output_path(value)
            if path is not None and path.is_file() and job.run_directory and path.is_relative_to(job.run_directory):
                urls[name] = "/output/" + quote(path.relative_to(self.output_root).as_posix(), safe="/")
        return urls

    def summary(self, job: Job) -> str:
        result = job.result or {}
        settings = result.get("config") or job.settings
        record = result.get("record") or {}
        state = {"running": "実行中", "success": "完了", "failure": "失敗"}.get(job.state, job.state)
        method = settings.get("method", job.settings["method"])
        parameters = f"反復回数: {settings.get('iterations')} / ε: {settings.get('epsilon')}"
        if method in METHODS and METHODS[method].sparse:
            parameters += f" / ピボット数: {settings.get('pivots')}"
        lines = [
            f"状態: {state}",
            f"グラフ: {job.settings['graph']}",
            f"手法: {METHOD_LABELS.get(method, method)}",
            parameters,
            f"乱数シード: {settings.get('seed')} / 画像サイズ: {settings.get('image_size')}",
        ]
        if record:
            lines.append(f"頂点数: {record.get('nodes')} / 辺数: {record.get('edges')} / 制約数: {record.get('constraints')}")
            lines.append("時間（ビルド・画像生成・ストレス評価を除く）")
            for key, label in (
                ("iteration_time_ms", "SGD反復（全反復の合計）"),
                ("algorithm_time_cold_ms", "処理全体（初期化を含む・入力読込を除く）"),
                ("algorithm_time_warm_ms", "処理全体（初期化を除く・差引値）"),
                ("cli_total_time_cold_ms", "入力読込＋処理全体（初期化を含む）"),
            ):
                value = record.get(key)
                lines.append(f"  {label}: {value:.3f} ms" if isinstance(value, (int, float)) else f"  {label}: 未計測")
            lines.append(f"Stress: {record.get('stress_value')} ({record.get('stress_kind')})")
        if job.error:
            lines.append(f"エラー ({result.get('stage', 'run')}): {job.error}")
        return "\n".join(lines)

    def snapshot(self, job_id: str) -> dict[str, Any] | None:
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                return None
            return {
                "id": job.id, "state": job.state, "log": job.log,
                "run_directory": job.run_directory.relative_to(self.root).as_posix() if job.run_directory else None,
                "result": job.result, "summary": self.summary(job),
                "artifacts": self.artifact_urls(job), "error": job.error,
            }

    def shutdown(self) -> None:
        with self.lock:
            self.stopping = True
            job = self.jobs.get(self.active_run) if self.active_run else None
            process = job.process if job else None
        if process is not None:
            for sig, timeout in ((signal.SIGINT, 3), (signal.SIGTERM, 2), (signal.SIGKILL, 2)):
                try:
                    os.killpg(process.pid, sig)
                except ProcessLookupError:
                    break
                try:
                    process.wait(timeout=timeout)
                    break
                except subprocess.TimeoutExpired:
                    continue
            # The runner can finish while a descendant is still alive in its group.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if job and job.thread:
            job.thread.join(timeout=3)


class LocalServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], service: ExperimentService):
        self.service = service
        super().__init__(address, RequestHandler)


class RequestHandler(BaseHTTPRequestHandler):
    server: LocalServer

    def access_allowed(self) -> bool:
        origin = self.headers.get("Origin")
        if not loopback_url(self.headers.get("Host", ""), host_header=True):
            self.json_response(403, {"error": "ローカルホストからアクセスしてください"}, cors=False)
            return False
        if origin is not None and not loopback_url(origin):
            self.json_response(403, {"error": "ローカルのページからアクセスしてください"}, cors=False)
            return False
        return True

    def cors_headers(self) -> None:
        origin = self.headers.get("Origin")
        if origin is not None and loopback_url(origin):
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def json_response(self, status: int, payload: Any, *, cors: bool = True) -> None:
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        if cors:
            self.cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        if self.access_allowed():
            self.json_response(200, {})

    def do_POST(self) -> None:
        if not self.access_allowed():
            return
        if self.path != "/api/runs":
            self.json_response(404, {"error": "APIがありません"})
            return
        if self.headers.get_content_type() != "application/json":
            self.json_response(415, {"error": "Content-Type: application/jsonが必要です"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 1 <= length <= BODY_LIMIT or self.headers.get("Transfer-Encoding") is not None:
                raise RequestError("JSONリクエストは1..16384バイトで指定してください")
            payload = json.loads(self.rfile.read(length))
            job = self.server.service.start(payload)
            self.json_response(202, self.server.service.snapshot(job.id))
        except (ValueError, UnicodeError) as error:
            self.json_response(400, {"error": str(error)})
        except BusyError as error:
            self.json_response(409, {"error": str(error), "active_run": self.server.service.config()["active_run"]})

    def do_GET(self) -> None:
        if not self.access_allowed():
            return
        path = urlsplit(self.path).path
        service = self.server.service
        if path == "/api/config":
            self.json_response(200, service.config())
        elif path.startswith("/api/runs/"):
            snapshot = service.snapshot(path[len("/api/runs/"):])
            self.json_response(200 if snapshot else 404, snapshot or {"error": "実験がありません"})
        elif path in {"/", "/index.html", "/app.js", "/styles.css"}:
            filename = "index.html" if path in {"/", "/index.html"} else path[1:]
            self.file_response(service.root / "tools/single_run" / filename)
        elif path.startswith("/output/"):
            try:
                candidate = (service.output_root / unquote(path[len("/output/"):])).resolve()
                allowed = candidate.is_relative_to(service.output_root) and candidate.is_file()
            except (OSError, ValueError):
                allowed = False
            if allowed:
                self.file_response(candidate)
            else:
                self.json_response(404, {"error": "ファイルがありません"})
        else:
            self.json_response(404, {"error": "ページがありません"})

    def file_response(self, path: Path) -> None:
        try:
            with path.open("rb") as source:
                self.send_response(200)
                content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
                if content_type.startswith("text/") or content_type == "application/javascript":
                    content_type += "; charset=utf-8"
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(os.fstat(source.fileno()).st_size))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.cors_headers()
                self.end_headers()
                while chunk := source.read(65536):
                    self.wfile.write(chunk)
        except (FileNotFoundError, IsADirectoryError, PermissionError):
            self.json_response(404, {"error": "ファイルがありません"})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765, help="localhost port (default: 8765)")
    parser.add_argument("--open", action="store_true", help="ブラウザも開く")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--portは1..65535で指定してください")
    url = f"http://127.0.0.1:{args.port}/"

    service = ExperimentService(REPO_ROOT)
    try:
        server = LocalServer(("127.0.0.1", args.port), service)
    except OSError as error:
        print(f"起動できません: {error}\n別のポートを --port 8766 で指定してください", file=sys.stderr)
        return 1
    print(f"Single-run page: {url}\nOutput: {service.output_root}\nCtrl+Cで終了します", flush=True)
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        print("\n実験とローカルサーバーを終了します", flush=True)
    finally:
        try:
            service.shutdown()
        finally:
            server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
