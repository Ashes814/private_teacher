from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from private_teacher.utils.exceptions import ConfigurationError

# 1. 程序启动时立即加载 .env 文件到 os.environ
# 这一步必须在所有其他代码之前执行，确保后续 os.getenv() 能读到 .env 里的值
# pydantic-settings 的 env_file 只对它自己定义的字段生效，
# 像 ANTHROPIC_API_KEY / MINIMAX_API_KEY 这种「非字段」的 key 必须用 dotenv 显式加载
load_dotenv()


class LLMSettings(BaseSettings):
    """大语言模型相关配置。

    包含四个 provider 的参数：
      - Anthropic Claude（云端，质量高）
      - OpenAI 兼容（含中转 API）
      - MINIMAX
      - Ollama（本地运行，完全离线）

    环境变量前缀：PRIVATE_TEACHER_LLM_
    例如 PRIVATE_TEACHER_LLM_PROVIDER=claude
    """

    provider: Literal["claude", "ollama", "openai_compat", "minimax"] = Field(
        default="claude",
        description="LLM 提供方，可选：claude / ollama / openai_compat / minimax",
    )

    # -----Claude 配置 -----
    claude_model: str = Field(default="claude-sonnet-5-20251001", description="Claude 模型名")

    # -----OpenAI 配置 -----
    openai_model: str = Field(
        default="gpt-4o-mini",
        description="OpenAI 模型名（也用于中转API）",
    )

    openai_base_url: str | None = Field(
        default=None, description="OpenAI 兼容 API 的 base_url (用于中转，自建服务)"
    )

    # -----Ollma 配置 -----
    ollama_base_url: str = Field(default="http://localhost:11434", description="Ollama 服务地址")
    ollama_model: str = Field(default="qwen2.5:7b", description="Ollama 模型名")

    # ----- MiniMax 配置（OpenAI 兼容协议，复用 ChatOpenAI 客户端）-----
    minimax_model: str = Field(default="MiniMax-M3", description="MiniMax 模型名称")
    minimax_base_url: str = Field(
        default="https://api.minimax.com/v1", description="MiniMax API 基础地址"
    )

    # ----- Embedding 配置 -----
    # 注意：embedding 的 provider 和 chat 的 provider 是**独立**的。
    # 常见组合：用 Claude 做对话（质量高），用本地 bge 做 embedding（省钱且快）。
    embedding_provider: Literal["openai", "ollama", "huggingface", "hash"] = Field(
        default="openai",
        description="Embedding 提供方: openai / ollama / huggingface / hash (离线测试用)",
    )

    embedding_model: str = Field(
        default="text-embedding3-small",
        description="Embedding 模型名。openai: text-embedding-3-small；"
        "ollama: nomic-embed-text；huggingface: BAAI/bge-small-zh-v1.5",
    )
    embedding_base_url: str | None = Field(
        default=None,
        description="Embedding 服务地址（走中转 API 时填）。留空用官方地址",
    )
    embedding_dimension: int = Field(
        default=1536,
        description="向量维度。⚠️ 必须与模型实际输出一致，否则 Chroma 写入报错。"
        "text-embedding-3-small=1536, bge-small-zh=512, nomic-embed-text=768",
    )
    embedding_batch_size: int = Field(
        default=64,
        description="每批送多少条文本去向量化。太大容易超 API 的单请求上限，"
        "太小则请求次数多、速度慢",
    )

    # ----- pydantic-settings 配置 -----
    model_config = SettingsConfigDict(
        # 所有环境变量加前缀，避免与系统其他变量冲突
        # 例如 LLMSettings 的 provider 字段对应 PRIVATE_TEACHER_LLM_PROVIDER
        env_prefix="PRIVATE_TEACHER_LLM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


class LogSettings(BaseSettings):
    """日志相关配置"""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="日志级别",
    )
    json_logs: bool = Field(default=False, description="是否输出JSON格式日志（生产环境推荐true）")

    model_config = SettingsConfigDict(
        env_prefix="PRIVATE_TEACHER_LOG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


class PathSettings(BaseSettings):
    """路径相关配置

    环境变量前缀：PRIVATE_TEACHER_PATH_

    所有用户数据（课件，知识库，单元，成绩）都保存在data_dir下

    """

    data_dir: Path = Field(
        default=Path("./data"), description="用户的数据根目录（课件、知识库、单元、成绩等）"
    )

    model_config = SettingsConfigDict(
        env_prefix="PRIVATE_TEACHER_PATH_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    def course_dir(self, course_id: str) -> Path:
        """获取某门课的课件目录（子目录）"""
        return self.data_dir / "courseware" / course_id

    def ensure_data_dir(self) -> None:
        """确保 data_dir 存在，不存在则创建"""
        self.data_dir.mkdir(parents=True, exist_ok=True)


# 顶层聚合配置
class AppSettings:
    """
    应用级配置聚合。

    本身不是 BaseSettings（避免 pydantic 重复实例化），
    而是显式持有三个子 settings 实例。

    用法：
        settings = AppSettings.load()
        print(settings.llm.provider)
        settings.validate_for_runtime()  # 关键参数校验
    """

    # 类型注解
    llm: LLMSettings
    logging: LogSettings
    paths: PathSettings

    def __init__(
        self,
        llm: LLMSettings,
        logging: LogSettings,
        paths: PathSettings,
    ) -> None:
        self.llm = llm
        self.logging = logging
        self.paths = paths

    @classmethod
    def load(cls) -> AppSettings:
        """一次性加载所有子配置。

        这里手动实例化三个子 settings 而不是用嵌套 BaseSettings，
        是为了让三个子模块可以独立被其他模块引用（如 logging 模块只需 LogSettings）。
        """

        return cls(
            llm=LLMSettings(),
            logging=LogSettings(),
            paths=PathSettings(),
        )

    def validate_for_runtime(self) -> None:
        """
        运行前校验关键参数。

        当前检查：
          - provider=claude 时必须有 ANTHROPIC_API_KEY
          - provider=openai_compat 时必须有 OPENAI_API_KEY
          - provider=minimax 时必须有 MINIMAX_API_KEY
          - provider=ollama 时不强制检查（Ollama 本地无 key）
        """

        if self.llm.provider == "claude":
            if not os.getenv("ANTHROPIC_API_KEY"):
                raise ConfigurationError(
                    "provider=claude 需要设置环境变量 ANTHROPIC_API_KEY，",
                    env_var="ANTHROPIC_API_KEY",
                )

        elif self.llm.provider == "openai_compat":
            if not os.getenv("OPENAI_API_KEY"):
                raise ConfigurationError("provider=openai_compat 需要设置环境变量 OPENAI_API_KEY")

        elif self.llm.provider == "minimax" and not os.getenv("MINIMAX_API_KEY"):
            # MiniMax 也走 API key 校验（用环境变量 MINIMAX_API_KEY）

            raise ConfigurationError(
                "provider=minimax 需要设置环境变量 MINIMAX_API_KEY，", env_var="MINIMAX_API_KEY"
            )

    def __repr__(self) -> str:
        # 隐藏敏感信息：llm 配置里不打印 key
        return (
            f"AppSettings(llm.provider={self.llm.provider!r}, "
            f"logging.level={self.logging.level!r}, "
            f"paths.data_dir={self.paths.data_dir!r})"
        )
