from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from loguru import logger


def pytest_configure(config: pytest.Config) -> None:
    """
    注册自定义marker

    --strict-markers开启后，未在此注册的marker会报错
    """

    config.addinivalue_line("markers", "integration: 集成测试（可能涉及文件系统，慢）,")

    config.addinivalue_line("markers", "e2e: 端到端测试（需要完整环境）")
    config.addinivalue_line("markers", "requires_llm: 需要真实 LLM API Key （默认跳过）")


# 全局 fixture
@pytest.fixture(autouse=True)
def _quiet_logs() -> None:
    """
    自动应用：每个测试运行时临时静音 loguru 到 ERROR 级别。

    autouse=True 表示不需要显式传入，pytest 自动注入到所有测试。

    为什么这么做？
      - 日志会让 pytest 输出变脏
      - 测试时只看 assert 结果，不看 INFO 日志
    """

    logger.remove()


@pytest.fixture
def tmp_env(monkeypatch: pytest.MonkeyPatch, tmp_path) -> object:
    """
    临时 .env 环境。

    用法：
        def test_xxx(tmp_env):
            (tmp_env / ".env").write_text("PRIVATE_TEACHER_LLM_PROVIDER=ollama")
            settings = AppSettings.load()
            assert settings.llm.provider == "ollama"

    实现：
      1. 在 tmp_path 下创建 .env
      2. monkeypatch.chdir 到 tmp_path，让 pydantic 找到 .env
      3. 清掉进程已加载的 PRIVATE_TEACHER_* 环境变量，
         否则 config.py 顶部 load_dotenv() 已经把项目 .env 灌进去了，
         pydantic-settings 会优先用 os.environ 里的旧值
    """

    monkeypatch.chdir(tmp_path)

    # 清掉所有项目相关的环境变量（包括可能从项目 .env 加载进来的）
    for key in list(os.environ):
        if key.startswith("PRIVATE_TEACHER_"):
            monkeypatch.delenv(key, raising=False)

    return tmp_path


@pytest.fixture
def fake_chat_model():
    """
    假 ChatModel，循环返回预设 responses。

    用法：
        def test_xxx(fake_chat_model, monkeypatch):
            monkeypatch.setattr(
                "private_teacher.llm.factory.build_chat_model",
                lambda settings: fake_chat_model(responses=["pong", "hello"]),
            )
            from private_teacher.llm.factory import quick_chat
            assert "pong" in quick_chat("ping")

    LangChain 自带的 FakeListChatModel 已实现所有接口（invoke/ainvoke/stream 等），
    不需要自己写 _generate。
    """

    def _factory(responses: list[str]) -> FakeListChatModel:
        return FakeListChatModel(responses=responses)

    return _factory


@pytest.fixture(scope="session")
def settings():
    """
    共享 AppSettings（整个测试会话只 load 一次）。

    注意：session scope 需要测试中不修改 settings 字段。
    需要修改请用 tmp_env fixture。
    """

    from private_teacher.config import AppSettings

    return AppSettings.load()


# ============================================================
# Phase 1：数据库 fixture
# ============================================================


@pytest.fixture
def tmp_db(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    """一个建好表的临时数据库连接，测试结束自动销毁。

    tmp_path 是 pytest 内置 fixture：每个测试函数独享一个空目录，
    测试结束后 pytest 自动清理（默认保留最近 3 次运行，方便排查）。

    为什么用真数据库而不是 mock？
      - SQLite 建一个内存/临时库只要几毫秒，比写 mock 便宜
      - mock 数据库最容易出现"测试全过，上线全崩"——SQL 语法错误 mock 根本发现不了
    """
    from private_teacher.storage.db import Database, init_db

    db_path = tmp_path / "test.db"
    with Database(db_path) as conn:
        init_db(conn)
        # yield 之前的代码 = setup，之后的代码 = teardown
        # 这里 teardown 交给 with 语句（自动 commit + close）
        yield conn


@pytest.fixture
def sample_course(tmp_db: sqlite3.Connection):
    """预置一门课，省得每个测试都手写创建逻辑。"""
    from private_teacher.storage import course_repo

    return course_repo.create(tmp_db, name="测试课程", description="用于单元测试")


# ============================================================
# Phase 1：Embedding fixture
# ============================================================
@pytest.fixture
def fake_embedder():
    """确定性的假 embedder（HashEmbedder，256 维）。

    用 256 维而不是 1536 维：
      - 测试里只关心"相似的文本向量更接近"，维度不影响这个性质
      - 维度小 6 倍，测试快 6 倍
    """
    from private_teacher.rag.embeddings import HashEmbedder

    return HashEmbedder(dimension=256)


@pytest.fixture
def hash_llm_settings():
    """把 embedding provider 强制设成 hash 的 LLMSettings。

    有了它，Indexer / Retriever / KBService 的测试都能完全离线跑，
    不需要任何 API key。
    """
    from private_teacher.config import LLMSettings

    return LLMSettings(
        embedding_provider="hash",
        embedding_dimension=256,
    )
