"""
测试 LLM 工厂。

覆盖：
  - 各 provider 返回正确类型
  - 缺 key 抛 ConfigurationError
  - 未知 provider 抛 ConfigurationError
  - quick_chat 走 base 接口
"""

from __future__ import annotations

import pytest
from langchain_core.language_models import BaseChatModel

from private_teacher.config import LLMSettings
from private_teacher.llm.factory import build_chat_model, quick_chat
from private_teacher.utils.exceptions import ConfigurationError, LLMError


class TestBuildChatModel:
    """测试 build_chat_model 工厂。"""

    def test_claude_returns_chat_anthropic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """provider=claude 应返回 ChatAnthropic 实例。"""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        settings = LLMSettings(provider="claude", claude_model="claude-sonnet-5-20251001")

        model = build_chat_model(settings)

        assert isinstance(model, BaseChatModel)
        # 进一步检查具体类型
        from langchain_anthropic import ChatAnthropic

        assert isinstance(model, ChatAnthropic)

    def test_claude_without_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """provider=claude 缺 key 时应抛 ConfigurationError。"""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        settings = LLMSettings(provider="claude")

        with pytest.raises(ConfigurationError) as exc_info:
            build_chat_model(settings)
        assert "ANTHROPIC_API_KEY" in str(exc_info.value)

    def test_openai_compat_returns_chat_openai(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """provider=openai_compat 应返回 ChatOpenAI 实例。"""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        settings = LLMSettings(
            provider="openai_compat",
            openai_model="gpt-4o-mini",
            openai_base_url="https://api.openai.com/v1",
        )

        model = build_chat_model(settings)

        from langchain_openai import ChatOpenAI

        assert isinstance(model, ChatOpenAI)

    def test_ollama_returns_chat_ollama(self) -> None:
        """provider=ollama 应返回 ChatOllama 实例。"""
        settings = LLMSettings(
            provider="ollama",
            ollama_model="qwen2.5:7b",
            ollama_base_url="http://localhost:11434",
        )

        model = build_chat_model(settings)

        from langchain_ollama import ChatOllama

        assert isinstance(model, ChatOllama)

    def test_minimax_returns_chat_openai(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """provider=minimax 应返回 ChatOpenAI 实例（因为走 OpenAI 兼容协议）。"""
        monkeypatch.setenv("MINIMAX_API_KEY", "test-minimax-key")
        settings = LLMSettings(
            provider="minimax",
            minimax_model="minimaxm3",
            minimax_base_url="https://api.minimax.com/v1",
        )

        model = build_chat_model(settings)

        from langchain_openai import ChatOpenAI

        assert isinstance(model, ChatOpenAI)
        # 进一步验证 base_url 和 model 配置正确
        assert model.openai_api_base == "https://api.minimax.com/v1"  # type: ignore[attr-defined]

    def test_minimax_without_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """provider=minimax 缺 key 时应抛 ConfigurationError。"""
        monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
        settings = LLMSettings(provider="minimax")

        with pytest.raises(ConfigurationError) as exc_info:
            build_chat_model(settings)
        assert "MINIMAX_API_KEY" in str(exc_info.value)

    def test_unknown_provider_raises(self) -> None:
        """未知 provider 应抛 ConfigurationError。"""
        # 用 model_construct 绕过 pydantic Literal 校验
        settings = LLMSettings.model_construct(provider="unknown")  # type: ignore[arg-type]

        with pytest.raises(ConfigurationError):
            build_chat_model(settings)


class TestQuickChat:
    """测试 quick_chat 便捷函数。"""

    def test_quick_chat_with_fake_model(
        self, monkeypatch: pytest.MonkeyPatch, fake_chat_model
    ) -> None:
        """quick_chat 应能调用假 model 并返回响应。"""

        # 注入假 model
        def fake_build(settings: LLMSettings) -> BaseChatModel:
            return fake_chat_model(responses=["pong"])

        monkeypatch.setattr("private_teacher.llm.factory.build_chat_model", fake_build)

        result = quick_chat("ping")
        assert "pong" in result

    def test_quick_chat_wraps_exception(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """quick_chat 内部异常应被包装为 LLMError。"""
        from langchain_core.language_models.fake_chat_models import FakeListChatModel

        class BrokenModel(FakeListChatModel):
            def invoke(self, *args, **kwargs):  # type: ignore[override]
                raise ConnectionError("network down")

        def fake_build(settings: LLMSettings) -> BaseChatModel:
            return BrokenModel(responses=[])

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        monkeypatch.setattr("private_teacher.llm.factory.build_chat_model", fake_build)

        with pytest.raises(LLMError) as exc_info:
            quick_chat("ping")
        # ConnectionError 是 transient 错误
        assert exc_info.value.transient is True


# 可选：真实 LLM 烟囱测试（默认跳过）
@pytest.mark.requires_llm
class TestRealLLM:
    """真实 LLM 测试（需 ANTHROPIC_API_KEY 环境变量）。"""

    def test_real_claude_responds(self) -> None:
        import os

        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("需要 ANTHROPIC_API_KEY")

        # settings = LLMSettings(provider="claude")
        result = quick_chat("Say 'pong' and nothing else.")
        assert "pong" in result.lower()
