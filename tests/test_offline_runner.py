"""Subprocess contracts for the socket audit QA runner; no external endpoints."""

from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import textwrap
import unittest


RUNNER = Path(__file__).with_name("run_without_network.py").resolve()


class OfflineRunnerTests(unittest.TestCase):
    def run_target(self, *arguments, cwd=None):
        return subprocess.run(
            [sys.executable, "-B", str(RUNNER), *map(str, arguments)],
            cwd=cwd, capture_output=True, text=True, timeout=15,
        )

    @staticmethod
    def write_script(path, source):
        path.write_text(textwrap.dedent(source), encoding="utf-8")

    def test_self_test_blocks_connect_and_succeeds(self):
        result = self.run_target("--self-test")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("PASS", result.stdout)
        self.assertEqual(result.stderr.strip(), "Blocked Python socket attempts: 1")

    def test_caught_connect_or_dns_cannot_produce_false_success(self):
        with TemporaryDirectory() as directory:
            script = Path(directory) / "caught_attempt.py"
            self.write_script(script, """
                import socket
                import sys
                try:
                    if sys.argv[1] == "connect":
                        with socket.socket() as probe:
                            probe.connect(("127.0.0.1", 9))
                    else:
                        socket.getaddrinfo("localhost", 9)
                except RuntimeError:
                    pass
                print("TARGET_FINISHED")
            """)
            for operation in ("connect", "dns"):
                with self.subTest(operation=operation):
                    result = self.run_target(script, operation)
                    self.assertEqual(result.stdout.strip(), "TARGET_FINISHED")
                    self.assertNotEqual(result.returncode, 0)
                    self.assertEqual(result.stderr.strip(), "Blocked Python socket attempts: 1")

    def test_target_exit_code_is_preserved(self):
        with TemporaryDirectory() as directory:
            script = Path(directory) / "exit_code.py"
            self.write_script(script, "raise SystemExit(7)\n")
            result = self.run_target(script)
            self.assertEqual(result.returncode, 7, result.stderr)
            self.assertEqual(result.stderr, "")

    def test_script_receives_arguments_and_imports_from_its_directory(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            script_dir = root / "scripts"
            script_dir.mkdir()
            self.write_script(script_dir / "local_helper.py", "VALUE = 42\n")
            script = script_dir / "entry.py"
            self.write_script(script, """
                from pathlib import Path
                import sys
                import local_helper
                assert local_helper.VALUE == 42
                assert sys.argv[0] == str(Path(__file__))
                assert sys.argv[1:] == ["alpha", "two words"]
                assert Path(sys.path[0]) == Path(__file__).resolve().parent
                print("SCRIPT_OK")
            """)
            result = self.run_target(script, "alpha", "two words", cwd=root)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), "SCRIPT_OK")

    def test_module_receives_arguments_and_imports_from_working_directory(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "sample_package"
            package.mkdir()
            self.write_script(package / "__init__.py", "VALUE = 42\n")
            self.write_script(root / "local_helper.py", "VALUE = 99\n")
            self.write_script(package / "__main__.py", """
                from pathlib import Path
                import sys
                from . import VALUE
                import local_helper
                assert VALUE == 42 and local_helper.VALUE == 99
                assert sys.argv[0] == str(Path(__file__))
                assert sys.argv[1:] == ["two words"]
                assert Path(sys.path[0]) == Path.cwd()
                print("MODULE_OK")
            """)
            result = self.run_target("-m", "sample_package", "two words", cwd=root)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), "MODULE_OK")


if __name__ == "__main__":
    unittest.main()
