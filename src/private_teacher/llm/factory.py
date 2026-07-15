"""
LLM 工厂模块
============

根据配置选择不同的 LLM provider，返回统一的 LangChain BaseChatModel 实例。

设计要点：
  - 单一入口 build_chat_model(settings) -> BaseChatModel
  - 用 match/case（Python 3.10+）替代 if/elif 链，代码更清晰
  - key 不放在 LLMSettings 字段里，而是从环境变量读取
    （避免 key 被序列化到日志 / .dict() / 错误信息中）
  - 缺 key 立即抛 ConfigurationError，让错误前置

LangChain 速览：
  - BaseChatModel 是所有 chat model 的抽象基类
  - .invoke(messages) 是统一接口，messages 是 list[BaseMessage]
  - HumanMessage(content="...") 表示用户消息
  - AIMessage 表示 AI 回复
  - 返回值 .content 拿到字符串内容
"""

from __future__ import annotations

# ---------- 标准库导入 ----------
import os
from functools import lru_cache  # 缓存装饰器

# ---------- 第三方库导入 ----------
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

# ---------- 本项目导入 ----------
from private_teacher.config import LLMSettings
from private_teacher.utils.exceptions import ConfigurationError, LLMError


# ============================================================
# 工厂函数
# ============================================================
def build_chat_model(settings: LLMSettings) -> BaseChatModel:
    """
    根据 settings.provider 构造对应的 chat model。

    Args:
        settings: LLM 配置

    Returns:
        BaseChatModel 实例（ChatAnthropic / ChatOpenAI / ChatOllama）

    Raises:
        ConfigurationError: provider 非法或缺少必要的 API key
    """

    # 用 match/case（Python 3.10+ 结构化模式匹配）
    # 比 if/elif 链更清晰，且能强制覆盖所有 provider

    match settings.provider:
        # ---------- Claude ----------
        case "claude":
            # 从环境变量读 key（不要放进 LLMSettings 字段，避免泄漏）
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                # 立即抛错，让用户配置好 .env 再重试
                raise ConfigurationError(
                    "缺少 ANTHROPIC_API_KEY",
                    provider="claude",
                    env_var="ANTHROPIC_API_KEY",
                    hint="在 .env 文件中设置 ANTHROPIC_API_KEY=sk-ant-...",
                )
            # temperature=0.2 让输出相对稳定（教学场景需要可复现）
            # max_tokens 留 None 用 provider 默认
            return ChatAnthropic(model=settings.claude_model, api_key=api_key, temperature=0.2)  # type: ignore

        # ---------- OpenAI 兼容（含中转） ----------
        case "openai_compat":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ConfigurationError(
                    "缺少 OPENAI_API_KEY",
                    provider="openai_compat",
                    env_var="OPENAI_API_KEY",
                )
            # base_url 可选：None 时用 OpenAI 官方地址
            return ChatOpenAI(  # type: ignore
                model=settings.openai_model,
                api_key=api_key,
                base_url=settings.openai_base_url,
                temperature=0.2,
            )

        # ---------- Ollama 本地 ----------
        case "ollama":
            # Ollama 不需要 key，但需要 base_url 和 model 都配置好
            # 不做严格校验，留给用户自己排查
            return ChatOllama(  # type: ignore
                model=settings.ollama_model,
                base_url=settings.ollama_base_url,
                temperature=0.2,
            )

        # ---------- MiniMax（OpenAI 兼容协议）----------
        case "minimax":
            # MiniMax 提供的是 OpenAI 兼容 API，所以直接复用 ChatOpenAI
            # 关键是把 base_url 指向 MiniMax 的网关
            api_key = os.getenv("MINIMAX_API_KEY")
            if not api_key:
                raise ConfigurationError(
                    "缺少 MINIMAX_API_KEY",
                    provider="minimax",
                    env_var="MINIMAX_API_KEY",
                    hint="在 .env 文件中设置 MINIMAX_API_KEY=<你的 key>",
                )
            return ChatOpenAI(  # type: ignore
                model=settings.minimax_model,  # 默认 minimaxm3
                api_key=api_key,
                base_url=settings.minimax_base_url,  # 默认 https://api.minimax.com/v1
                temperature=0.2,
            )

        # ---------- 兜底 ----------
        case _:
            # 理论上不会到这里（pydantic 已经校验过 Literal）
            # 但留个兜底更安全
            raise ConfigurationError(
                f"未知的 LLM provider: {settings.provider!r}",
                provider=str(settings.provider),
                valid_options=["claude", "ollama", "openai_compat", "minimax"],
            )


# ============================================================
# 便捷函数（Phase 0 冒烟测试用）
# ============================================================
def quick_chat(
    prompt: str,
    settings: LLMSettings | None = None,
) -> str:
    """
    一句话调用 LLM，返回字符串回复。

    用法：
        from private_teacher.llm.factory import quick_chat
        reply = quick_chat("用一句话介绍你自己")

    Args:
        prompt: 用户输入
        settings: 可选，不传则从环境读默认配置

    Returns:
        LLM 的回复文本

    Raises:
        LLMError: 调用失败时（已包装）
    """

    # 默认从环境读（避免每次都传 settings）
    if settings is None:
        settings = LLMSettings()

    # 构造 chat model
    model = build_chat_model(settings)

    try:
        # invoke 是同步调用，耗时 1~10s
        # 异步版本用 model.ainvoke(messages)
        result = model.invoke([HumanMessage(content=prompt)])
    except Exception as exc:
        # 把所有底层异常包装成 LLMError，方便上层统一处理
        # 根据异常类型判断是否 transient
        transient = isinstance(exc, (TimeoutError, ConnectionError, OSError))
        # ⬇️ 安全获取 model 名（兼容所有 chat model，包括 FakeListChatModel）
        # ChatAnthropic/OpenAI/Ollama 都有 .model 属性，但 FakeListChatModel 没有
        # 用 getattr 安全访问，没有就 fallback 到类名
        model_name = getattr(model, "model", None) or type(model).__name__
        raise LLMError(
            f"LLM 调用失败: {exc}",
            transient=transient,
            provider=settings.provider,
            model=str(model_name),
        ) from exc

    return str(result.content)


# ============================================================
# 缓存版工厂（可选，Phase 0 不需要）
# ============================================================
@lru_cache(maxsize=1)
def get_default_chat_model() -> BaseChatModel:
    """
    缓存的默认 chat model。

    适合在长生命周期进程（如 Streamlit session）里复用同一个 model，
    避免每次都新建（langchain 内部会缓存 HTTP 连接）。

    Phase 0 暂不使用，先把基础功能跑通。
    """
    return build_chat_model(LLMSettings())
