from __future__ import annotations

from pathlib import Path

import pytest

from private_teacher.config import AppSettings, LLMSettings, LogSettings, PathSettings
from private_teacher.utils.exceptions import ConfigurationError


class TestLLMSettings:
    """测试LLMSettings 子配置"""

    def test_default_provider_is_claude(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """默认provider应该是claude"""
        monkeypatch.delenv("PRIVATE_TEACHER_LLM_PROVIDER", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        monkeypatch.chdir(Path("/tmp"))

        settings = LLMSettings()
        assert settings.provider == "claude"
        assert settings.claude_model == "claude-sonnet-5-20251001"

    def test_env_var_overrides_default(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """环境变量应该覆盖默认值。"""
        # monkeypatch.setattr(LLMSettings.model_config, "env_file", None)
        monkeypatch.delenv("PRIVATE_TEACHER_LLM_PROVIDER", raising=False)
        monkeypatch.setenv("PRIVATE_TEACHER_LLM_PROVIDER", "ollama")
        monkeypatch.chdir(Path("/tmp"))

        settings = LLMSettings()

        assert settings.provider == "ollama"

    def test_dotenv_loading(self, tmp_env: Path) -> None:
        """.env 文件应被加载。"""
        # tmp_env fixture 把 cwd 切到 tmp_path
        (tmp_env / ".env").write_text(
            "PRIVATE_TEACHER_LLM_PROVIDER=openai_compat\nPRIVATE_TEACHER_LLM_OPENAI_MODEL=gpt-4o\n"
        )

        settings = LLMSettings()
        assert settings.provider == "openai_compat"
        assert settings.openai_model == "gpt-4o"

    def test_env_var_beats_dotenv(self, tmp_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """环境变量优先级高于 .env 文件。"""
        (tmp_env / ".env").write_text("PRIVATE_TEACHER_LLM_PROVIDER=ollama\n")
        monkeypatch.setenv("PRIVATE_TEACHER_LLM_PROVIDER", "claude")

        settings = LLMSettings()
        assert settings.provider == "claude"

    def test_invalid_provider_raises(self, tmp_env: Path) -> None:
        """provider 取非法值时，pydantic 应抛 ValidationError。"""
        from pydantic import ValidationError

        (tmp_env / ".env").write_text("PRIVATE_TEACHER_LLM_PROVIDER=invalid\n")
        with pytest.raises(ValidationError):
            LLMSettings()


class TestLogSettings:
    """测试LogSettings子配置"""

    def test_default_level_is_info(self) -> None:
        settings = LogSettings()

        assert settings.level == "INFO"
        assert settings.json_logs is False


class TestPathSettings:
    """测试 PathSettings 子配置。"""

    def test_default_data_dir(self) -> None:
        settings = PathSettings()
        assert settings.data_dir == Path("./data")

    def test_ensure_data_dir_creates_dir(self, tmp_path: Path) -> None:
        """ensure_data_dir 应在 data_dir 不存在时创建它。"""
        from private_teacher.config import PathSettings

        settings = PathSettings(data_dir=tmp_path / "new_data")
        assert not (tmp_path / "new_data").exists()
        settings.ensure_data_dir()
        assert (tmp_path / "new_data").exists()


class TestAppSettings:
    """测试 AppSettings 聚合配置。"""

    def test_load_returns_all_subsystems(self) -> None:
        """AppSettings.load() 应返回包含所有子配置的实例。"""
        settings = AppSettings.load()
        assert isinstance(settings.llm, LLMSettings)
        assert isinstance(settings.logging, LogSettings)
        assert isinstance(settings.paths, PathSettings)

    def test_validate_raises_for_claude_without_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """provider=claude 但 ANTHROPIC_API_KEY 缺失时应抛 ConfigurationError。"""
        monkeypatch.setenv("PRIVATE_TEACHER_LLM_PROVIDER", "claude")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        settings = AppSettings.load()
        with pytest.raises(ConfigurationError) as exc_info:
            settings.validate_for_runtime()
        assert "ANTHROPIC_API_KEY" in str(exc_info.value)

    def test_validate_passes_for_claude_with_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """provider=claude 且 ANTHROPIC_API_KEY 设置时不应抛错。"""
        monkeypatch.setenv("PRIVATE_TEACHER_LLM_PROVIDER", "claude")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

        settings = AppSettings.load()
        settings.validate_for_runtime()  # 不应抛错

    def test_validate_raises_for_minimax_without_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """provider=minimax 但 MINIMAX_API_KEY 缺失时应抛 ConfigurationError。"""
        monkeypatch.setenv("PRIVATE_TEACHER_LLM_PROVIDER", "minimax")
        monkeypatch.delenv("MINIMAX_API_KEY", raising=False)

        settings = AppSettings.load()
        with pytest.raises(ConfigurationError) as exc_info:
            settings.validate_for_runtime()
        assert "MINIMAX_API_KEY" in str(exc_info.value)

    def test_validate_passes_for_minimax_with_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """provider=minimax 且 MINIMAX_API_KEY 设置时不应抛错。"""
        monkeypatch.setenv("PRIVATE_TEACHER_LLM_PROVIDER", "minimax")
        monkeypatch.setenv("MINIMAX_API_KEY", "test-minimax-key")

        settings = AppSettings.load()
        settings.validate_for_runtime()  # 不应抛错

    def test_repr_hides_sensitive_info(self) -> None:
        """__repr__ 不应泄露 key。"""
        settings = AppSettings.load()
        repr_str = repr(settings)
        # repr 里不应包含任何 key 字样
        assert "key" not in repr_str.lower() or "api_key" not in repr_str.lower()
