from __future__ import annotations

import http.client
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments import serve_single


STUB_RUNNER = r'''
import argparse
import json
import sys
import time
import uuid
from pathlib import Path

parser = argparse.ArgumentParser()
for name in ("input", "method", "iterations", "epsilon", "pivots", "seed", "image-size", "output-root"):
    parser.add_argument("--" + name, required=True)
args = parser.parse_args()
folder = Path(args.output_root) / "graph" / uuid.uuid4().hex
folder.mkdir(parents=True)
print(f"Output: {folder}", flush=True)
if args.seed == "9":
    while not (Path(args.output_root).parent / "release").exists():
        time.sleep(0.02)
if args.seed == "2":
    print("x" * 40000, flush=True)
config = {"input": args.input, "method": args.method, "iterations": int(args.iterations),
          "epsilon": float(args.epsilon), "pivots": int(args.pivots), "seed": int(args.seed),
          "image_size": args.image_size, "warmups": 0}
artifacts = {"info": str(folder / "run_info.txt"), "result": str(folder / "run_result.json")}
failure = args.seed == "1"
record = None if failure else {"nodes": 2, "edges": 1, "constraints": 1,
    "iteration_time_ms": 2.5, "algorithm_time_cold_ms": 3.0,
    "algorithm_time_warm_ms": 2.8, "cli_total_time_cold_ms": 4.0,
    "stress_value": 0.125, "stress_kind": "full"}
if not failure:
    for name, suffix in (("before_image", "0"), ("after_image", "1")):
        image = folder / f"run_{suffix}.png"
        image.write_bytes(b"PNG fixture")
        artifacts[name] = str(image)
metadata = {"status": "failure" if failure else "success", "config": config,
    "record": record, "run_directory": str(folder), "artifacts": artifacts,
    "stage": "build_method" if failure else "complete",
    "error": "cargo unavailable; install Rust" if failure else None}
Path(artifacts["result"]).write_text(json.dumps(metadata))
Path(artifacts["info"]).write_text("settings\n# Edges\n0 1\n# Positions\n0 0\n1 1\n")
if failure:
    print("Failed (build_method): cargo unavailable; install Rust", file=sys.stderr, flush=True)
    raise SystemExit(1)
print(f"Saved: {folder}", flush=True)
'''


class GraphCatalogTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="graph catalog ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        (self.root / "data").mkdir()

    def write_graph(self, name: str, contents: str) -> Path:
        path = self.root / "data" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("%%MatrixMarket matrix coordinate real general\n" + contents)
        return path

    def test_edges_exclude_diagonals_and_merge_reversed_and_duplicate_coordinates(self):
        self.write_graph("normalized.mtx", "7 7 8\n1 1 1\n1 2 0\n2 1 7\n1 2 9\n2 3 1\n4 5 1\n5 4 1\n7 7 1\n")
        self.assertEqual(serve_single.graph_catalog(self.root), [{
            "path": "data/normalized.mtx", "name": "normalized", "nodes": 7, "entries": 8, "edges": 3,
        }])

    def test_counts_cover_disconnected_input_and_isolated_nodes_before_lcc_selection(self):
        self.write_graph("disconnected.mtx", "8 8 3\n1 2\n3 5\n5 7\n")
        graph = serve_single.graph_catalog(self.root)[0]
        self.assertEqual(graph["nodes"], 8)
        self.assertEqual(graph["edges"], 3)

    def test_catalog_skips_incomplete_or_invalid_coordinates_without_losing_valid_graphs(self):
        invalid = {
            "missing_dimensions": "",
            "not_square": "2 3 1\n1 2\n",
            "header_only": "2 2 1\n",
            "missing_coordinate": "2 2 2\n1 2\n",
            "extra_coordinate": "2 2 1\n1 2\n2 1\n",
            "missing_column": "2 2 1\n1\n",
            "bad_index": "2 2 1\na 2\n",
            "zero_index": "2 2 1\n0 2\n",
            "negative_index": "2 2 1\n-1 2\n",
            "out_of_range": "2 2 1\n1 3\n",
            "negative_entries": "2 2 -1\n",
        }
        for name, contents in invalid.items():
            self.write_graph(name + ".mtx", contents)
        self.write_graph("valid.mtx", "% comment\n2 2 1\n% entry comment\n\n1 2\n")
        graphs = serve_single.graph_catalog(self.root)
        self.assertEqual([graph["name"] for graph in graphs], ["valid"])
        self.assertEqual(graphs[0]["edges"], 1)

    def test_cached_counts_refresh_when_mtime_or_size_changes(self):
        path = self.write_graph("graph.mtx", "2 2 1\n1 2\n")
        with patch.object(serve_single, "_read_graph_statistics", wraps=serve_single._read_graph_statistics) as read:
            self.assertEqual(serve_single.graph_catalog(self.root)[0]["edges"], 1)
            serve_single.graph_catalog(self.root)
            self.assertEqual(read.call_count, 1, "unchanged graphs should not rescan entries")

            # Same size, changed mtime: replacing an edge with a self loop.
            previous = path.stat()
            self.write_graph("graph.mtx", "2 2 1\n1 1\n")
            os.utime(path, ns=(previous.st_atime_ns, previous.st_mtime_ns + 1_000_000))
            self.assertEqual(path.stat().st_size, previous.st_size)
            self.assertEqual(serve_single.graph_catalog(self.root)[0]["edges"], 0)
            self.assertEqual(read.call_count, 2)

            # Same mtime, changed size: a separate change-signature component.
            previous = path.stat()
            self.write_graph("graph.mtx", "3 3 2\n1 2\n2 3\n")
            os.utime(path, ns=(previous.st_atime_ns, previous.st_mtime_ns))
            self.assertEqual(path.stat().st_mtime_ns, previous.st_mtime_ns)
            graph = serve_single.graph_catalog(self.root)[0]
            self.assertEqual((graph["nodes"], graph["edges"]), (3, 2))
            serve_single.graph_catalog(self.root)
            self.assertEqual(read.call_count, 3)

    def test_cached_invalid_file_is_reconsidered_after_repair(self):
        path = self.write_graph("graph.mtx", "2 2 1\n")
        self.assertEqual(serve_single.graph_catalog(self.root), [])
        with path.open("a") as source:
            source.write("1 2\n")
        self.assertEqual(serve_single.graph_catalog(self.root)[0]["edges"], 1)


class ServeSingleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.http_logging = patch.object(serve_single.RequestHandler, "log_message", return_value=None)
        self.http_logging.start()
        self.addCleanup(self.http_logging.stop)
        self.temporary = tempfile.TemporaryDirectory(prefix="single api ")
        self.root = Path(self.temporary.name).resolve()
        (self.root / "data/v1_lcc").mkdir(parents=True)
        self.graph = self.root / "data/graph.mtx"
        self.graph.write_text("%%MatrixMarket matrix coordinate pattern symmetric\n% comment\n2 2 1\n1 2\n")
        (self.root / "data/v1_lcc/graph.mtx").write_text(self.graph.read_text())
        (self.root / "data/not_graph.mtx").write_text("%%MatrixMarket matrix array real general\n2 2\n")
        (self.root / "experiments").mkdir()
        (self.root / "experiments/run_single.py").write_text(STUB_RUNNER)
        (self.root / "tools/single_run").mkdir(parents=True)
        (self.root / "tools/single_run/index.html").write_text("<html>single-run page</html>")
        self.service = serve_single.ExperimentService(self.root)
        self.server = serve_single.LocalServer(("127.0.0.1", 0), self.service)
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()

    def tearDown(self) -> None:
        self.service.shutdown()
        self.server.shutdown()
        self.server.server_close()
        self.server_thread.join(timeout=2)
        self.temporary.cleanup()

    def payload(self, **overrides):
        return {"graph": "data/graph.mtx", "method": "rr_sparse_sgd", "iterations": 2,
                "epsilon": 0.1, "pivots": 1, "seed": "0", "image_size": "1024x1024", **overrides}

    def request(self, method: str, path: str, body=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=3)
        request_headers = dict(headers or {})
        if isinstance(body, dict):
            body = json.dumps(body)
            request_headers.setdefault("Content-Type", "application/json")
        connection.request(method, path, body=body, headers=request_headers)
        response = connection.getresponse()
        status, returned_headers, data = response.status, dict(response.getheaders()), response.read()
        connection.close()
        if returned_headers.get("Content-Type", "").startswith("application/json"):
            data = json.loads(data)
        return status, returned_headers, data

    def finish(self, job_id: str):
        job = self.service.jobs[job_id]
        job.thread.join(timeout=5)
        self.assertFalse(job.thread.is_alive(), "stub runner did not finish")
        return self.service.snapshot(job_id)

    def test_catalog_and_config_include_graph_counts_and_distinct_paths(self):
        status, _, config = self.request("GET", "/api/config")
        self.assertEqual(status, 200)
        self.assertEqual(len(config["graphs"]), 2)
        self.assertEqual(config["graphs"][0], {"path": "data/graph.mtx", "name": "graph", "nodes": 2, "entries": 1, "edges": 1})
        self.assertEqual(config["graphs"][1]["name"], "graph")
        self.assertEqual(config["graphs"][1]["path"], "data/v1_lcc/graph.mtx")
        self.assertEqual(config["defaults"]["graph"], "data/graph.mtx")
        self.assertEqual(len(config["methods"]), 6)
        self.assertIsNone(config["active_run"])
        self.assertEqual(self.request("GET", "/")[2], b"<html>single-run page</html>")

    def test_config_refresh_and_post_validation_rescan_added_or_removed_graphs(self):
        added = self.root / "data/added.mtx"
        added.write_text(self.graph.read_text())
        config = self.request("GET", "/api/config")[2]
        self.assertIn("data/added.mtx", {graph["path"] for graph in config["graphs"]})
        fresh = self.root / "data/fresh.mtx"
        fresh.write_text(self.graph.read_text())
        # POST must see files added even without a preceding GET /api/config.
        job = self.service.start(self.payload(graph="data/fresh.mtx"))
        self.assertEqual(self.finish(job.id)["state"], "success")
        self.graph.unlink()
        self.assertEqual(self.request("POST", "/api/runs", self.payload())[0], 400)
        config = self.request("GET", "/api/config")[2]
        self.assertNotIn("data/graph.mtx", {graph["path"] for graph in config["graphs"]})

    def test_success_keeps_native_four_files_and_returns_short_copyable_summary(self):
        maximum_seed = str(2**64 - 1)
        status, _, response = self.request("POST", "/api/runs", self.payload(seed=maximum_seed))
        self.assertEqual(status, 202)
        result = self.finish(response["id"])
        self.assertEqual(result["state"], "success")
        self.assertIsNone(result["error"])
        self.assertEqual(result["result"]["config"]["seed"], 2**64 - 1)
        self.assertIn("頂点数: 2 / 辺数: 1", result["summary"])
        self.assertIn("SGD反復（全反復の合計）: 2.500 ms", result["summary"])
        self.assertIn("処理全体（初期化を含む・入力読込を除く）: 3.000 ms", result["summary"])
        self.assertIn("処理全体（初期化を除く・差引値）: 2.800 ms", result["summary"])
        self.assertIn("入力読込＋処理全体（初期化を含む）: 4.000 ms", result["summary"])
        self.assertIn("グラフ: data/graph.mtx", result["summary"])
        self.assertIn("Stress: 0.125 (full)", result["summary"])
        self.assertNotIn(str(self.root), result["summary"])
        self.assertNotIn("Output:", result["summary"])
        self.assertNotIn("warmups", result["summary"])
        self.assertNotIn("# Edges", result["summary"])
        self.assertEqual(len(list((self.root / result["run_directory"]).iterdir())), 4)
        for name, url in result["artifacts"].items():
            self.assertTrue(url.startswith("/output/"), (name, url))
            self.assertEqual(self.request("GET", url)[0], 200)
        self.assertIn("Output:", result["log"])
        self.assertIn("Saved:", result["log"])
        self.assertIsNone(self.service.config()["active_run"])

    def test_failure_retains_native_metadata_and_saved_folder_then_releases_slot(self):
        status, _, response = self.request("POST", "/api/runs", self.payload(seed="1"))
        self.assertEqual(status, 202)
        result = self.finish(response["id"])
        self.assertEqual(result["state"], "failure")
        self.assertIn("install Rust", result["error"])
        self.assertIn("build_method", result["summary"])
        self.assertIsNotNone(result["run_directory"])
        self.assertIsNotNone(result["artifacts"]["info"])
        self.assertIsNone(result["artifacts"]["before_image"])
        self.assertIn("Failed", result["log"])
        next_job = self.service.start(self.payload())
        self.assertEqual(self.finish(next_job.id)["state"], "success")

    def test_only_one_native_job_at_a_time(self):
        first = self.service.start(self.payload(seed="9"))
        status, _, busy = self.request("POST", "/api/runs", self.payload())
        self.assertEqual(status, 409)
        self.assertEqual(busy["active_run"], first.id)
        self.assertEqual(self.service.config()["active_run"], first.id)
        (self.root / "release").touch()
        self.assertEqual(self.finish(first.id)["state"], "success")

    def test_spawn_failure_is_reported_and_does_not_hold_busy_slot(self):
        with patch.object(serve_single.subprocess, "Popen", side_effect=OSError("python unavailable")):
            job = self.service.start(self.payload())
            result = self.finish(job.id)
        self.assertEqual(result["state"], "failure")
        self.assertIn("python unavailable", result["log"])
        self.assertIsNone(result["run_directory"])
        self.assertIsNone(self.service.config()["active_run"])

    def test_progress_log_is_bounded_but_output_directory_still_retained(self):
        job = self.service.start(self.payload(seed="2"))
        result = self.finish(job.id)
        self.assertEqual(result["state"], "success")
        self.assertLessEqual(len(result["log"]), serve_single.LOG_LIMIT)
        self.assertIsNotNone(result["run_directory"])

    def test_validation_blocks_bad_numbers_unknown_fields_and_graph_traversal(self):
        invalid_settings = [
            {"graph": "../secret.mtx"}, {"graph": "/etc/passwd"}, {"graph": []},
            {"method": "shell"}, {"method": {}}, {"iterations": 0}, {"iterations": True},
            {"iterations": 1.5}, {"epsilon": "NaN"}, {"epsilon": "Infinity"},
            {"epsilon": -1}, {"epsilon": None}, {"pivots": 0}, {"seed": -1},
            {"seed": str(2**64)}, {"seed": "$(touch secret)"}, {"image_size": "8193x1024"},
            {"image_size": "1024x0"}, {"image_size": "1024"}, {"output_root": "/tmp/elsewhere"},
        ]
        for invalid in invalid_settings:
            with self.subTest(invalid=invalid):
                self.assertEqual(self.request("POST", "/api/runs", self.payload(**invalid))[0], 400)
        self.assertEqual(self.request("POST", "/api/runs", "[]", {"Content-Type": "application/json"})[0], 400)
        self.assertEqual(self.request("POST", "/api/runs", "{", {"Content-Type": "application/json"})[0], 400)
        self.assertFalse((self.root / "output").exists())

    def test_request_security_allows_live_server_and_blocks_remote_origin_or_host(self):
        local_origin = "http://localhost:5500"
        status, headers, _ = self.request("OPTIONS", "/api/runs", headers={"Origin": local_origin})
        self.assertEqual(status, 200)
        self.assertEqual(headers["Access-Control-Allow-Origin"], local_origin)
        for origin in ("https://evil.example", "null", "http://localhost.evil:5500", "http://localhost:0", "http://localhost:65536", "http://user@localhost:5500", "http://localhost:5500/"):
            with self.subTest(origin=origin):
                status, headers, _ = self.request("POST", "/api/runs", self.payload(), {"Origin": origin})
                self.assertEqual(status, 403)
                self.assertNotIn("Access-Control-Allow-Origin", headers)
        self.assertEqual(self.request("GET", "/api/config", headers={"Host": "evil.example:8765"})[0], 403)
        self.assertEqual(self.request("POST", "/api/runs", "{}", {"Content-Type": "text/plain"})[0], 415)
        self.assertEqual(self.request("POST", "/api/runs", "{}", {"Content-Type": "application/json", "Content-Length": "20000"})[0], 400)
        self.assertTrue(serve_single.loopback_url("http://[::1]:5500"))
        self.assertFalse(serve_single.loopback_url("http://localhost:\n5500"))

    def test_file_serving_prevents_traversal_and_symlink_escape(self):
        output = self.root / "output"
        output.mkdir()
        secret = self.root / "secret.txt"
        secret.write_text("private")
        (output / "escape.txt").symlink_to(secret)
        for path in ("/output/../secret.txt", "/output/%2e%2e/secret.txt", "/output/%2fetc/passwd", "/output/%00", "/output/escape.txt", "/experiments/run_single.py"):
            self.assertEqual(self.request("GET", path)[0], 404, path)
        (self.root / "data/escape.mtx").symlink_to(secret)
        self.assertNotIn("data/escape.mtx", {graph["path"] for graph in serve_single.graph_catalog(self.root)})
        self.assertEqual(self.request("GET", "/api/runs/missing")[0], 404)

    def test_native_command_uses_current_python_and_fixed_output_root(self):
        job = serve_single.Job("fixture", self.payload(seed="18446744073709551615"))
        command = self.service.command(job)
        self.assertEqual(command[:3], [sys.executable, "-u", str(self.root / "experiments/run_single.py")])
        self.assertEqual(command[-2:], ["--output-root", str(self.root / "output")])
        self.assertEqual(command[command.index("--seed") + 1], "18446744073709551615")

    def test_shutdown_terminates_running_experiment_group(self):
        job = self.service.start(self.payload(seed="9"))
        deadline = time.monotonic() + 3
        while "Output:" not in self.service.snapshot(job.id)["log"] and time.monotonic() < deadline:
            time.sleep(0.01)
        process = job.process
        self.assertIsNotNone(process)
        self.service.shutdown()
        self.assertIsNotNone(process.poll())
        self.assertEqual(self.service.snapshot(job.id)["state"], "failure")
        self.assertIsNone(self.service.config()["active_run"])


if __name__ == "__main__":
    unittest.main()
