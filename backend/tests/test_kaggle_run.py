import os
import sys
import zipfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from scripts.kaggle_run import (
    resolve_kaggle_cmd,
    credentials_present,
    parse_status,
    verify_artifacts,
    main,
)


class TestResolveKaggleCmd:
    def test_found_via_which(self):
        with patch("shutil.which", return_value="C:\\Tools\\kaggle.exe"):
            cmd = resolve_kaggle_cmd("kaggle")
            assert cmd == ["C:\\Tools\\kaggle.exe"]

    def test_found_direct_file(self, tmp_path):
        dummy_exe = tmp_path / "custom_kaggle.exe"
        dummy_exe.write_text("dummy")
        with patch("shutil.which", return_value=None):
            cmd = resolve_kaggle_cmd(str(dummy_exe))
            assert cmd == [str(dummy_exe.resolve())]

    def test_found_windows_extension(self, tmp_path):
        dummy_bin = tmp_path / "kaggle.exe"
        dummy_bin.write_text("dummy")
        with patch("sys.platform", "win32"), \
             patch("shutil.which", side_effect=lambda x: str(dummy_bin) if x.endswith(".exe") else None):
            cmd = resolve_kaggle_cmd(str(tmp_path / "kaggle"))
            assert cmd is not None
            assert cmd[0].endswith(".exe")

    def test_found_in_python_dir(self, tmp_path):
        py_bin = tmp_path / "python.exe"
        py_bin.write_text("python")
        kaggle_bin = tmp_path / "kaggle.exe"
        kaggle_bin.write_text("kaggle")
        with patch("sys.platform", "win32"), patch("shutil.which", return_value=None):
            cmd = resolve_kaggle_cmd("kaggle", python_bin=str(py_bin))
            assert cmd == [str(kaggle_bin.resolve())]

    def test_fallback_to_module_when_standalone_not_found(self):
        with patch("shutil.which", return_value=None), \
             patch("pathlib.Path.is_file", return_value=False), \
             patch("importlib.util.find_spec", return_value=MagicMock()):
            cmd = resolve_kaggle_cmd("kaggle", python_bin="C:\\fake\\python.exe")
            # When python_bin != sys.executable, it probes --version
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0)
                cmd = resolve_kaggle_cmd("kaggle", python_bin="C:\\fake\\python.exe")
                assert cmd == ["C:\\fake\\python.exe", "-m", "kaggle"]

    def test_fallback_to_module_sys_executable(self):
        with patch("shutil.which", return_value=None), \
             patch("pathlib.Path.is_file", return_value=False), \
             patch("importlib.util.find_spec", return_value=MagicMock()):
            cmd = resolve_kaggle_cmd("kaggle", python_bin=sys.executable)
            assert cmd == [sys.executable, "-m", "kaggle"]

    def test_not_found_returns_none(self):
        with patch("shutil.which", return_value=None), \
             patch("pathlib.Path.is_file", return_value=False), \
             patch("importlib.util.find_spec", return_value=None):
            cmd = resolve_kaggle_cmd("kaggle", python_bin=sys.executable)
            assert cmd is None

    def test_custom_bin_not_found_returns_none(self):
        with patch("shutil.which", return_value=None), \
             patch("pathlib.Path.is_file", return_value=False):
            cmd = resolve_kaggle_cmd("nonexistent_kaggle_tool")
            assert cmd is None


class TestCredentialsPresent:
    def test_api_token_env(self, monkeypatch):
        monkeypatch.setenv("KAGGLE_API_TOKEN", "fake_token")
        assert credentials_present() is True

    def test_username_and_key_env(self, monkeypatch):
        monkeypatch.setenv("KAGGLE_USERNAME", "testuser")
        monkeypatch.setenv("KAGGLE_KEY", "testkey")
        assert credentials_present() is True

    def test_credentials_json_file(self, monkeypatch, tmp_path):
        monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)
        monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
        monkeypatch.delenv("KAGGLE_KEY", raising=False)
        kaggle_dir = tmp_path / ".kaggle"
        kaggle_dir.mkdir()
        (kaggle_dir / "credentials.json").write_text("{}")
        with patch("pathlib.Path.home", return_value=tmp_path):
            assert credentials_present() is True

    def test_kaggle_json_file(self, monkeypatch, tmp_path):
        monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)
        monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
        monkeypatch.delenv("KAGGLE_KEY", raising=False)
        kaggle_dir = tmp_path / ".kaggle"
        kaggle_dir.mkdir()
        (kaggle_dir / "kaggle.json").write_text("{}")
        with patch("pathlib.Path.home", return_value=tmp_path):
            assert credentials_present() is True

    def test_missing_credentials(self, monkeypatch, tmp_path):
        monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)
        monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
        monkeypatch.delenv("KAGGLE_KEY", raising=False)
        with patch("pathlib.Path.home", return_value=tmp_path):
            assert credentials_present() is False

    def test_oserror_handled_gracefully(self, monkeypatch):
        monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)
        monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
        monkeypatch.delenv("KAGGLE_KEY", raising=False)
        with patch("pathlib.Path.home", side_effect=OSError("Access denied")):
            assert credentials_present() is False


class TestParseStatus:
    def test_parse_complete(self):
        assert parse_status("Kernel status: complete") == "COMPLETE"

    def test_parse_running(self):
        assert parse_status("Kernel status: running (1m 20s)") == "RUNNING"

    def test_parse_error(self):
        assert parse_status("Kernel status: error") == "ERROR"

    def test_parse_cancelled(self):
        assert parse_status("Kernel status: cancelled") == "CANCELLED"

    def test_parse_unknown(self):
        assert parse_status("unexpected text") == "UNKNOWN"


class TestVerifyArtifacts:
    def test_valid_artifacts(self, tmp_path):
        for name in ("yolo_artifacts.zip", "cnn_artifacts.zip"):
            zpath = tmp_path / name
            with zipfile.ZipFile(zpath, "w") as zf:
                zf.writestr("test.txt", "sample content that makes zip > 1024 bytes" * 50)
        problems = verify_artifacts(tmp_path)
        assert len(problems) == 0

    def test_missing_artifacts(self, tmp_path):
        problems = verify_artifacts(tmp_path)
        assert len(problems) == 2
        assert any("yolo_artifacts.zip" in p for p in problems)
        assert any("cnn_artifacts.zip" in p for p in problems)

    def test_corrupt_zip(self, tmp_path):
        for name in ("yolo_artifacts.zip", "cnn_artifacts.zip"):
            zpath = tmp_path / name
            zpath.write_bytes(b"not a valid zip file content")
        problems = verify_artifacts(tmp_path)
        assert len(problems) == 2
        assert all("not a valid zip" in p for p in problems)


class TestMainFlow:
    def test_dry_run_exits_zero(self, monkeypatch):
        test_args = ["kaggle_run.py", "--owner", "testuser", "--dry-run"]
        monkeypatch.setattr(sys, "argv", test_args)
        assert main() == 0

    def test_cli_not_found_exits_two(self, monkeypatch):
        test_args = ["kaggle_run.py", "--owner", "testuser"]
        monkeypatch.setattr(sys, "argv", test_args)
        with patch("scripts.kaggle_run.resolve_kaggle_cmd", return_value=None):
            assert main() == 2

    def test_credentials_missing_exits_two(self, monkeypatch):
        test_args = ["kaggle_run.py", "--owner", "testuser"]
        monkeypatch.setattr(sys, "argv", test_args)
        with patch("scripts.kaggle_run.resolve_kaggle_cmd", return_value=["kaggle"]), \
             patch("scripts.kaggle_run.credentials_present", return_value=False):
            assert main() == 2
