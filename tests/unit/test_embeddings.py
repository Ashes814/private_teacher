"""embeddings 单元测试。

注意：所有测试都不需要网络和 API key —— 真实 provider 的测试用
@pytest.mark.requires_llm 标记，默认跳过。
"""

from __future__ import annotations

import math

import pytest

from private_teacher.config import LLMSettings
from private_teacher.rag.embeddings import (
    Embedder,
    HashEmbedder,
    clear_embedder_cache,
    embed_in_batches,
    get_cached_embedder,
    get_embedder,
)
from private_teacher.utils.exceptions import ConfigurationError, RAGError


def _cosine(a: list[float], b: list[float]) -> float:
    """余弦相似度。两个向量都归一化过时，它等价于点积。"""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


# ============================================================
# HashEmbedder
# ============================================================
class TestHashEmbedder:
    def test_dimension(self) -> None:
        e = HashEmbedder(dimension=128)
        assert len(e.embed_query("测试")) == 128

    def test_deterministic(self) -> None:
        """⭐ 同样输入必须永远同样输出。"""
        e = HashEmbedder(dimension=64)
        assert e.embed_query("机器学习") == e.embed_query("机器学习")

    def test_normalized(self) -> None:
        """输出应是单位向量（L2 范数为 1）。"""
        vec = HashEmbedder(dimension=64).embed_query("深度学习很有趣")
        norm = math.sqrt(sum(v * v for v in vec))
        assert norm == pytest.approx(1.0)

    def test_empty_text_no_nan(self) -> None:
        """空文本不能产生零向量（会让余弦相似度变 NaN）。"""
        vec = HashEmbedder(dimension=64).embed_query("")
        assert math.sqrt(sum(v * v for v in vec)) == pytest.approx(1.0)

    def test_similar_texts_closer(self) -> None:
        """⭐ 关键能力：相关文本的相似度应高于无关文本。

        这个测试保证 HashEmbedder 能支撑起后面 retriever 的测试断言。
        """
        e = HashEmbedder(dimension=512)
        query = e.embed_query("机器学习")
        related = e.embed_query("机器学习是人工智能的一个分支")
        unrelated = e.embed_query("今天天气不错适合出门散步")

        assert _cosine(query, related) > _cosine(query, unrelated)

    def test_identical_text_similarity_is_one(self) -> None:
        e = HashEmbedder(dimension=256)
        v = e.embed_query("完全相同的文本")
        assert _cosine(v, v) == pytest.approx(1.0)

    def test_embed_documents_batch(self) -> None:
        e = HashEmbedder(dimension=64)
        vecs = e.embed_documents(["a", "b", "c"])
        assert len(vecs) == 3
        assert all(len(v) == 64 for v in vecs)

    def test_satisfies_protocol(self) -> None:
        """结构化类型检查：HashEmbedder 应满足 Embedder 协议。"""
        assert isinstance(HashEmbedder(), Embedder)


# ============================================================
# 工厂
# ============================================================
class TestEmbedderFactory:
    def test_hash_provider(self) -> None:
        settings = LLMSettings(embedding_provider="hash", embedding_dimension=128)
        embedder = get_embedder(settings)
        assert isinstance(embedder, HashEmbedder)
        assert len(embedder.embed_query("x")) == 128

    def test_openai_requires_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """没有 OPENAI_API_KEY 时必须立刻报错，而不是等到调 API 才失败。"""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        settings = LLMSettings(embedding_provider="openai")

        with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
            get_embedder(settings)

    def test_openai_with_key_builds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """有 key 就能构造成功（不发起真实请求）。"""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake")
        settings = LLMSettings(embedding_provider="openai")

        embedder = get_embedder(settings)

        assert hasattr(embedder, "embed_query")

    def test_unknown_provider(self) -> None:
        """构造一个非法 provider（绕过 pydantic 校验来测兜底分支）。"""
        settings = LLMSettings(embedding_provider="hash")
        object.__setattr__(settings, "embedding_provider", "nope")

        with pytest.raises(ConfigurationError, match="未知"):
            get_embedder(settings)


class TestEmbedderCache:
    def test_same_config_same_instance(self) -> None:
        clear_embedder_cache()
        settings = LLMSettings(embedding_provider="hash", embedding_dimension=64)

        # is 比较的是对象身份，不是值 —— 确认真的复用了同一个实例
        assert get_cached_embedder(settings) is get_cached_embedder(settings)

    def test_different_config_different_instance(self) -> None:
        clear_embedder_cache()
        a = get_cached_embedder(LLMSettings(embedding_provider="hash", embedding_dimension=64))
        b = get_cached_embedder(LLMSettings(embedding_provider="hash", embedding_dimension=128))
        assert a is not b


# ============================================================
# 批处理
# ============================================================
class TestEmbedInBatches:
    def test_returns_all_vectors(self, fake_embedder) -> None:
        texts = [f"文本{i}" for i in range(10)]
        vecs = embed_in_batches(fake_embedder, texts, batch_size=3)
        assert len(vecs) == 10

    def test_empty_input(self, fake_embedder) -> None:
        assert embed_in_batches(fake_embedder, []) == []

    def test_progress_callback(self, fake_embedder) -> None:
        seen: list[tuple[int, int]] = []
        embed_in_batches(
            fake_embedder,
            [f"t{i}" for i in range(10)],
            batch_size=4,
            on_progress=lambda done, total: seen.append((done, total)),
        )
        # 10 条 / 每批 4 条 = 3 批
        assert seen == [(4, 10), (8, 10), (10, 10)]

    def test_wraps_error_as_ragerror(self) -> None:
        """底层异常必须被包装成 RAGError，且带上批次信息。"""

        class BrokenEmbedder:
            def embed_documents(self, texts):
                raise RuntimeError("模拟 API 挂了")

            def embed_query(self, text):
                return [0.0]

        with pytest.raises(RAGError, match="向量化失败"):
            embed_in_batches(BrokenEmbedder(), ["a", "b"])


# ============================================================
# 真实 provider（默认跳过）
# ============================================================
@pytest.mark.requires_llm
class TestRealProviders:
    def test_openai_embedding(self) -> None:
        """需要真实 key：uv run pytest -m requires_llm"""
        settings = LLMSettings(embedding_provider="ollama")
        vec = get_embedder(settings).embed_query("hello")
        assert len(vec) == settings.embedding_dimension
