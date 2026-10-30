from __future__ import annotations

import contextlib
import datetime as dt
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments import run_single
from experiments.tests.test_runner import FAKE_METHOD


LAYOUT = "# Edges (source target)\n0 1\n\n# Positions (x y)\n0.25 0.5\n1.5 -0.75\n"
FAKE_RENDERER = r'''#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path
input_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
with open(os.environ["SINGLE_CALLS"], "a") as log:
    log.write(json.dumps({"program": "renderer", "args": sys.argv[1:]}) + "\n")
assert input_path.is_file()
assert sys.argv[3:] == ["--size", "2048x2048", "--stress", "off", "--palette", "sunset"]
if os.environ.get("SINGLE_RENDER_FAIL") == "after" and output_path.stem.endswith("_1"):
    raise SystemExit("renderer failure")
output_path.write_bytes(b"\x89PNG\r\n\x1a\nfixture")
print("rendered")
'''
FAKE_CARGO = r'''#!/usr/bin/env python3
import json
import os
import sys
with open(os.environ["SINGLE_CALLS"], "a") as log:
    log.write(json.dumps({"program": "cargo", "args": sys.argv[1:]}) + "\n")
assert sys.argv[1:4] == ["build", "--release", "--locked"]
if os.environ.get("SINGLE_BUILD_FAIL"):
    raise SystemExit("build failure")
'''


class SingleRunTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="single run ")
        self.root = Path(self.temp.name).resolve()
        self.graph = self.root / "graph with spaces.mtx"
        self.graph.write_text("%%MatrixMarket matrix coordinate pattern general\n2 2 1\n1 2\n")
        self.output = self.root / "output"
        self.calls = self.root / "calls.jsonl"
        self.environment = patch.dict(os.environ, {"SINGLE_CALLS": str(self.calls), "PATH": str(self.root / "bin") + os.pathsep + os.environ["PATH"]})
        self.environment.start()
        self.repository = patch.object(run_single, "REPO_ROOT", self.root)
        self.repository.start()
        self.make_executable(self.root / "bin/cargo", FAKE_CARGO)
        method_code = FAKE_METHOD.replace('final_path = output_dir / "final.txt"', 'final_path = output_dir / f"{args.run_id}.final.txt"')
        method_code = method_code.replace('final_path.write_text("positions\\n", encoding="utf-8")', "final_path.write_text(" + repr(LAYOUT) + ", encoding='utf-8')\ninitial_path = final_path.with_name(final_path.name.replace('.final.txt', '.initial.txt'))\ninitial_path.write_text(" + repr(LAYOUT) + ", encoding='utf-8')")
        # Add a real vertex-map header, so relocation is checked beyond JSON paths.
        method_code = method_code.replace('family = "sparse" if args.pivots is not None else "full"', '''if args.pivots is not None:
    for path in (final_path, initial_path):
        path.write_text(f"# Vertex map file: {vertex_map}\\n" + path.read_text())
family = "sparse" if args.pivots is not None else "full"''')
        for spec in run_single.METHODS.values():
            self.make_executable(self.root / spec.binary, method_code)
        self.make_executable(self.root / run_single.RENDERER, FAKE_RENDERER)

    def tearDown(self) -> None:
        self.repository.stop()
        self.environment.stop()
        self.temp.cleanup()

    def make_executable(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        path.chmod(0o755)

    def execute(self, *options: str) -> tuple[int, str, str]:
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = run_single.main(["--input", str(self.graph), "--output-root", str(self.output), *options])
        return code, stdout.getvalue(), stderr.getvalue()

    def results(self) -> list[dict]:
        return [json.loads(path.read_text()) for path in sorted(self.output.rglob("*_result.json"))]

    def test_single_run_preserves_config_timings_and_layout_and_renders_twice(self) -> None:
        code, console, errors = self.execute("--iterations", "2", "--pivots", "1", "--seed", "3")
        self.assertEqual(code, 0, errors)
        metadata = self.results()[0]
        record, artifacts = metadata["record"], metadata["artifacts"]
        self.assertEqual(metadata["status"], "success")
        self.assertEqual(record["seed"], 3)
        self.assertEqual(record["iterations"], 2)
        self.assertEqual(record["iteration_time_ms"], 6.0)
        self.assertEqual(metadata["config"]["warmups"], 0)
        self.assertEqual(dt.datetime.fromisoformat(metadata["started_at"]).utcoffset(), dt.timedelta(hours=9))
        folder = Path(metadata["run_directory"])
        self.assertEqual(folder.parent.name, "graph-with-spaces")
        self.assertRegex(folder.name, r"^\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}_rr_sparse_sgd$")
        self.assertEqual(len(list(folder.iterdir())), 4)
        info = Path(artifacts["info"]).read_text()
        self.assertIn("Experiment settings", info)
        self.assertIn("Iteration: 6.000 ms", info)
        self.assertIn("# Edges (source target; 0-based vertex IDs)\n0 1", info)
        self.assertIn("# Positions (x y; after SGD; line index = 0-based vertex ID)\n0.25 0.5\n1.5 -0.75", info)
        self.assertIn("Experiment settings", console)
        self.assertIn("Iteration: 6.000 ms", console)
        self.assertEqual(record["final_positions_path"], artifacts["info"])
        self.assertIsNone(record["vertex_map_path"])
        self.assertEqual(set(artifacts), {"info", "result", "before_image", "after_image"})
        for name, suffix in (("before_image", "_0.png"), ("after_image", "_1.png")):
            self.assertTrue(Path(artifacts[name]).is_file())
            self.assertTrue(Path(artifacts[name]).name.endswith(suffix))
        self.assertLess(Path(artifacts["before_image"]).name, Path(artifacts["after_image"]).name)
        self.assertFalse(any(path.is_dir() for path in folder.iterdir()))
        calls = [json.loads(line) for line in self.calls.read_text().splitlines()]
        self.assertEqual([call["program"] for call in calls], ["cargo", "cargo", "renderer", "renderer"])
        self.assertTrue(all("--manifest-path" in call["args"] for call in calls[:2]))

    def test_full_method_has_no_pivots_or_vertex_map(self) -> None:
        code, _, errors = self.execute("--method", "sgd", "--iterations", "2")
        self.assertEqual(code, 0, errors)
        metadata = self.results()[0]
        self.assertIsNone(metadata["config"]["pivots"])
        self.assertIsNone(metadata["record"]["pivots"])
        self.assertIsNone(metadata["record"]["vertex_map_path"])
        self.assertEqual(len(list(Path(metadata["run_directory"]).iterdir())), 4)

    def test_renderer_failure_keeps_layout_in_info_and_failure_status(self) -> None:
        with patch.dict(os.environ, {"SINGLE_RENDER_FAIL": "after"}):
            code, console, errors = self.execute("--iterations", "2", "--pivots", "1")
        self.assertEqual(code, 1)
        self.assertNotIn("Saved:", console)
        self.assertIn("renderer failure", errors)
        metadata = self.results()[0]
        self.assertEqual(metadata["status"], "failure")
        self.assertEqual(metadata["stage"], "render_after")
        self.assertEqual(metadata["record"]["status"], "success")
        artifacts = metadata["artifacts"]
        self.assertTrue(Path(artifacts["before_image"]).is_file())
        self.assertNotIn("after_image", artifacts)
        self.assertIn("renderer failure", metadata["error"])
        self.assertEqual(len(list(Path(metadata["run_directory"]).iterdir())), 3)
        info = Path(artifacts["info"]).read_text()
        self.assertTrue(info.startswith("Status: failure"))
        self.assertIn("# Positions", info)

    def test_build_failure_is_not_reported_as_success(self) -> None:
        with patch.dict(os.environ, {"SINGLE_BUILD_FAIL": "1"}):
            code, _, errors = self.execute()
        self.assertEqual(code, 1)
        self.assertIn("build failure", errors)
        metadata = self.results()[0]
        self.assertEqual(metadata["stage"], "build_method")
        self.assertIsNone(metadata["record"])
        self.assertEqual(metadata["status"], "failure")
        self.assertIn("build failure", Path(metadata["artifacts"]["info"]).read_text())
        self.assertEqual(len(list(Path(metadata["run_directory"]).iterdir())), 2)

    def test_sgd_failure_keeps_error_in_info_and_does_not_render(self) -> None:
        with patch.dict(os.environ, {"FAKE_FAIL_SEED": "0"}):
            code, console, errors = self.execute()
        self.assertEqual(code, 1)
        self.assertNotIn("Saved:", console)
        self.assertIn("intentional failure", errors)
        metadata = self.results()[0]
        self.assertEqual(metadata["status"], "failure")
        self.assertEqual(metadata["stage"], "sgd")
        self.assertIsNone(metadata["record"])
        self.assertIn("intentional failure", Path(metadata["artifacts"]["info"]).read_text())
        self.assertEqual(len(list(Path(metadata["run_directory"]).iterdir())), 2)
        calls = [json.loads(line) for line in self.calls.read_text().splitlines()]
        self.assertEqual([call["program"] for call in calls], ["cargo", "cargo"])

    def test_same_second_folder_collision_never_overwrites(self) -> None:
        first = run_single.create_run_dir(self.output, "../ graph ..", "2026-10-13-00-00-00", "rr_sparse_sgd")
        marker = first / "keep.txt"
        marker.write_text("keep")
        second = run_single.create_run_dir(self.output, "../ graph ..", "2026-10-13-00-00-00", "rr_sparse_sgd")
        self.assertNotEqual(first, second)
        self.assertTrue(second.name.endswith("_2"))
        self.assertEqual(marker.read_text(), "keep")
        self.assertTrue(first.is_relative_to(self.output))
        self.assertTrue(second.is_relative_to(self.output))
        self.assertEqual(run_single.safe_name(".."), "graph")

    def test_invalid_inputs_do_not_create_a_run(self) -> None:
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = run_single.main(["--input", str(self.root / "missing.mtx"), "--output-root", str(self.output)])
        self.assertEqual(code, 1)
        self.assertFalse(self.output.exists())
        for value in ("0", "8193", "2x0", "invalid"):
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(stderr):
                run_single.parse_args(["--input", str(self.graph), "--image-size", value])
        self.assertEqual(run_single.image_size("1024X2048"), "1024x2048")


if __name__ == "__main__":
    unittest.main()
