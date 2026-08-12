"""splitters 单元测试。"""

from __future__ import annotations

import dataclasses

import pytest
from langchain_core.documents import Document

from private_teacher.rag.splitters import (
    ChunkerConfig,
    estimate_tokens,
    split_documents,
    split_text,
)


# ============================================================
# 配置校验
# ============================================================
class TestChunkerConfig:
    def test_defaults(self) -> None:
        cfg = ChunkerConfig()
        assert cfg.chunk_size == 800
        assert cfg.chunk_overlap == 100

    def test_overlap_must_be_smaller(self) -> None:
        """overlap >= chunk_size 会导致合并死循环，必须提前拦住。"""
        with pytest.raises(ValueError, match="chunk_overlap"):
            ChunkerConfig(chunk_size=100, chunk_overlap=100)

    def test_invalid_size(self) -> None:
        with pytest.raises(ValueError, match="chunk_size"):
            ChunkerConfig(chunk_size=0)

    def test_is_frozen(self) -> None:
        """frozen dataclass 不允许修改字段。"""
        cfg = ChunkerConfig()
        # slots=True 的 dataclass 赋值时会抛 FrozenInstanceError，
        # 没 slots 会抛 AttributeError —— 两种都是"不可变"的正确表现
        with pytest.raises((AttributeError, dataclasses.FrozenInstanceError)):
            cfg.chunk_size = 999  # type: ignore[misc]


# ============================================================
# split_text
# ============================================================
class TestSplitText:
    def test_short_text_single_chunk(self) -> None:
        assert split_text("这是一段很短的文本内容，不需要切分。") == [
            "这是一段很短的文本内容，不需要切分。"
        ]

    def test_all_chunks_within_size(self) -> None:
        """核心不变量：任何 chunk 都不能超过 chunk_size。"""
        text = "机器学习是人工智能的分支。" * 200  # 2600 字符
        chunks = split_text(text, ChunkerConfig(chunk_size=300, chunk_overlap=50))

        assert len(chunks) > 1
        for c in chunks:
            assert len(c) <= 300, f"chunk 超长：{len(c)}"

    def test_no_content_lost(self) -> None:
        """切分不能丢内容：所有 chunk 拼起来应覆盖原文的每个字符。"""
        text = "".join(f"第{i}句话。" for i in range(1, 101))
        chunks = split_text(text, ChunkerConfig(chunk_size=200, chunk_overlap=0))

        # overlap=0 时直接拼接应等于原文（去掉切分时 strip 的空白）
        assert "".join(chunks) == text

    def test_overlap_creates_repetition(self) -> None:
        """overlap > 0 时，相邻 chunk 应该有重叠内容。"""
        text = "".join(f"句子{i}。" for i in range(1, 201))
        chunks = split_text(text, ChunkerConfig(chunk_size=200, chunk_overlap=60))

        assert len(chunks) >= 2
        # 拼起来的总长度应该大于原文（因为有重复部分）
        assert sum(len(c) for c in chunks) > len(text)

    def test_prefers_paragraph_boundary(self) -> None:
        """有段落分隔时，应优先在段落处切，而不是把段落腰斩。"""
        para = "内容" * 100  # 200 字符
        text = f"{para}\n\n{para}\n\n{para}"
        chunks = split_text(text, ChunkerConfig(chunk_size=250, chunk_overlap=0))

        # 每块应该正好是一个完整段落（250 装不下两个 200 的段落）
        assert len(chunks) == 3

    def test_chinese_sentence_separator(self) -> None:
        """没有换行的纯中文长句串，也要能按句号切开。"""
        text = "这是一个很长的句子用来测试中文切分效果。" * 50
        chunks = split_text(text, ChunkerConfig(chunk_size=100, chunk_overlap=0))

        for c in chunks:
            assert len(c) <= 100

    def test_no_separator_hard_cut(self) -> None:
        """一个分隔符都没有的超长串，靠字符硬切保底。"""
        text = "x" * 1000
        chunks = split_text(text, ChunkerConfig(chunk_size=100, chunk_overlap=0))

        assert len(chunks) == 10
        assert all(len(c) == 100 for c in chunks)

    def test_tiny_fragments_dropped(self) -> None:
        """低于 min_chunk_size 的碎屑要被丢掉。"""
        chunks = split_text("短", ChunkerConfig(min_chunk_size=20))
        assert chunks == []

    def test_empty_input(self) -> None:
        assert split_text("") == []


# ============================================================
# split_documents
# ============================================================
class TestSplitDocuments:
    def _doc(self, content: str, **meta) -> Document:
        base = {"source_path": "/tmp/a.txt", "page": 1}
        base.update(meta)
        return Document(page_content=content, metadata=base)

    def test_short_doc_not_split(self) -> None:
        docs = [self._doc("这是一段足够长但不超过上限的测试文本内容。")]
        assert len(split_documents(docs)) == 1

    def test_long_doc_split(self) -> None:
        docs = [self._doc("句子内容。" * 500)]
        chunks = split_documents(docs, ChunkerConfig(chunk_size=800, chunk_overlap=100))
        assert len(chunks) >= 3

    def test_chunk_id_unique(self) -> None:
        docs = [self._doc("测试内容。" * 400)]
        chunks = split_documents(docs)
        ids = [c.metadata["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_id_stable_across_runs(self) -> None:
        """⭐ 最重要的测试：同样输入必须产出同样的 chunk_id。

        如果这个测试挂了，说明用了 hash() 或全局计数器，
        增量索引会因为 id 漂移而完全失效。
        """
        docs = [self._doc("稳定性测试内容。" * 300)]
        ids_1 = [c.metadata["chunk_id"] for c in split_documents(docs)]
        ids_2 = [c.metadata["chunk_id"] for c in split_documents(docs)]
        assert ids_1 == ids_2

    def test_chunk_id_isolated_per_page(self) -> None:
        """不同页的编号相互独立 —— 新增一页不该影响其他页的 id。"""
        p1 = self._doc("第一页内容。" * 300, page=1)
        p2 = self._doc("第二页内容。" * 300, page=2)

        only_p1 = [c.metadata["chunk_id"] for c in split_documents([p1])]
        both = [
            c.metadata["chunk_id"] for c in split_documents([p1, p2]) if c.metadata["page"] == 1
        ]
        assert only_p1 == both

    def test_preserves_metadata(self) -> None:
        docs = [
            self._doc(
                "内容。" * 400,
                source_path="/a.pdf",
                page=3,
                line_start=1,
                file_type="pdf",
            )
        ]
        for c in split_documents(docs):
            assert c.metadata["source_path"] == "/a.pdf"
            assert c.metadata["page"] == 3
            assert c.metadata["file_type"] == "pdf"

    def test_metadata_not_shared(self) -> None:
        """⚠️ 每个 chunk 必须有独立的 metadata dict（浅拷贝陷阱）。"""
        docs = [self._doc("内容。" * 400)]
        chunks = split_documents(docs)

        assert len(chunks) >= 2
        # 改一个不能影响另一个
        chunks[0].metadata["page"] = 999
        assert chunks[1].metadata["page"] == 1

    def test_empty_docs(self) -> None:
        assert split_documents([]) == []


class TestEstimateTokens:
    def test_chinese(self) -> None:
        # 10 个汉字 ≈ 10 token
        assert estimate_tokens("机器学习深度学习算法") == 10

    def test_english(self) -> None:
        # 40 个 ASCII 字符 ≈ 10 token
        assert estimate_tokens("a" * 40) == 10
