from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NATIVE_CLI = ROOT / "runtime" / "native" / "bin" / "docker"


class NativeRuntimeCliTest(unittest.TestCase):
    def run_cli(self, root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        env = dict(os.environ)
        env["TONGBENCH_NATIVE_ROOT"] = str(root / "runtime")
        return subprocess.run(
            [sys.executable, str(NATIVE_CLI), *args],
            env=env,
            capture_output=True,
            text=True,
            check=check,
        )

    def test_task_lifecycle_and_path_mapping(self) -> None:
        with tempfile.TemporaryDirectory(prefix="tongbench_native_test_") as raw_root:
            root = Path(raw_root)
            app = root / "app"
            exported = root / "exported"
            app.mkdir()
            exported.mkdir()
            (app / "input.txt").write_text("input\n", encoding="utf-8")

            run = self.run_cli(
                root,
                "run",
                "-d",
                "--name",
                "unit_task",
                "-e",
                "EXAMPLE=value",
                "-v",
                f"{app}:/app:ro",
                "fake:image",
                "/bin/bash",
                "-c",
                "tail -f /dev/null",
            )
            self.assertEqual(run.stdout.strip(), "unit_task")

            self.run_cli(
                root,
                "exec",
                "unit_task",
                "/bin/bash",
                "-c",
                "cp -r /app/. /tmp_workspace && mkdir -p /tmp_workspace/tmp && printf '%s\\n' \"$EXAMPLE\" > /tmp_workspace/result.txt && printf 'nested\\n' > /tmp_workspace/tmp/nested.txt",
            )
            self.run_cli(root, "cp", "unit_task:/tmp_workspace/.", str(exported))
            self.assertEqual((exported / "input.txt").read_text(encoding="utf-8"), "input\n")
            self.assertEqual((exported / "result.txt").read_text(encoding="utf-8"), "value\n")
            task_tmp = root / "runtime" / "tasks" / "unit_task" / "tmp_workspace" / "tmp"
            self.assertEqual((task_tmp / "nested.txt").read_text(encoding="utf-8"), "nested\n")
            self.assertFalse((task_tmp / "runtime").exists())

            inspect = self.run_cli(root, "inspect", "unit_task")
            payload = json.loads(inspect.stdout)
            self.assertTrue(payload[0]["State"]["Running"])

            self.run_cli(root, "rm", "-f", "unit_task")
            self.assertFalse((root / "runtime" / "tasks" / "unit_task").exists())

    def test_unknown_task_fails(self) -> None:
        with tempfile.TemporaryDirectory(prefix="tongbench_native_test_") as raw_root:
            result = self.run_cli(Path(raw_root), "inspect", "missing", check=False)
            self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
