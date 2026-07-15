"""
测试 CLI 模块。

覆盖：
  - hello 命令打印欢迎语
  - hello-llm 走 mock quick_chat
  - 错误情况返回非零 exit code
"""

from __future__ import annotations

import argparse

import pytest

from private_teacher.cli import build_parser, cmd_hello, cmd_hello_llm, main


class TestHelloCommand:
    """测试 hello 子命令。"""

    def test_hello_prints_greeting(self, capsys: pytest.CaptureFixture) -> None:
        """hello 应打印包含 'Hello' 的欢迎语。"""
        result = cmd_hello(argparse.Namespace())  # type: ignore[arg-type]
        captured = capsys.readouterr()

        assert result == 0
        assert "Hello" in captured.out
        assert "Private Teacher" in captured.out

    def test_hello_via_main(self, capsys: pytest.CaptureFixture) -> None:
        """通过 main(['hello']) 调用应工作。"""
        result = main(["hello"])
        captured = capsys.readouterr()
        assert result == 0
        assert "Hello" in captured.out


class TestHelloLLMCommand:
    """测试 hello-llm 子命令。"""

    def test_hello_llm_uses_quick_chat(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """hello-llm 应调用 quick_chat 并打印结果。"""
        # mock quick_chat
        monkeypatch.setattr(
            "private_teacher.cli.quick_chat",
            lambda prompt, settings=None: "I am a fake LLM",
        )
        # mock 配置加载（避免依赖真实 .env）
        from private_teacher.config import AppSettings, LLMSettings, LogSettings
        from private_teacher.utils.exceptions import ConfigurationError

        fake_settings = AppSettings(
            llm=LLMSettings(provider="claude"),
            logging=LogSettings(level="ERROR"),  # 测试时静音
            paths=type("P", (), {"data_dir": None})(),  # 占位
        )
        monkeypatch.setattr(
            "private_teacher.cli.AppSettings.load",
            classmethod(lambda cls: fake_settings),
        )
        # validate_for_runtime 不抛错
        monkeypatch.setattr(
            "private_teacher.config.AppSettings.validate_for_runtime",
            lambda self: None,
        )

        result = main(["hello-llm"])
        captured = capsys.readouterr()

        assert result == 0
        assert "fake LLM" in captured.out

    def test_hello_llm_returns_error_on_missing_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """缺 key 时 hello-llm 应返回非零 exit code。"""
        from private_teacher.config import AppSettings, LLMSettings, LogSettings
        from private_teacher.utils.exceptions import ConfigurationError

        fake_settings = AppSettings(
            llm=LLMSettings(provider="claude"),
            logging=LogSettings(level="ERROR"),
            paths=type("P", (), {"data_dir": None})(),
        )
        monkeypatch.setattr(
            "private_teacher.cli.AppSettings.load",
            classmethod(lambda cls: fake_settings),
        )

        # 让 validate_for_runtime 抛错
        def fake_validate(self) -> None:
            raise ConfigurationError("missing key", env_var="ANTHROPIC_API_KEY")

        monkeypatch.setattr(
            "private_teacher.config.AppSettings.validate_for_runtime",
            fake_validate,
        )

        result = main(["hello-llm"])
        assert result == 2  # 缺 key 用 exit code 2


class TestParser:
    """测试 argparse 配置。"""

    def test_parser_requires_command(self) -> None:
        """不指定子命令应报错。"""
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_parser_version(self, capsys: pytest.CaptureFixture) -> None:
        """--version 应打印版本号。"""
        parser = build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--version"])
        # version action 会调 sys.exit(0)
        assert exc_info.value.code == 0
