"""retriever 单元测试（离线）。"""
from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.documents import Document

from private_teacher.config import PathSettings
from private_teacher.rag.retriever import RetrievalResult, Retriever
from private_teacher.rag.splitters import split_documents
from private_teacher.rag.vector_store import VectorStoreManager

COURSE_ID = "c1"

@pytest.fixture
def paths(tmp_path: Path) -> PathSettings:
    return PathSettings(data_dir=tmp_path / "data")


@pytest.fixture
def seeded(paths, fake_embedder) -> VectorStoreManager:
    """预先灌了 4 条数据的向量库。"""
    store = VectorStoreManager(COURSE_ID, paths.data_dir)

    raw = [
        ("/course/ml.pdf", "pdf", "main", 1, "机器学习是人工智能的核心分支领域。"),
        ("/course/ml.pdf", "pdf", "main", 2, "监督学习需要带标签的训练数据集合。"),
        ("/course/code.py", "py", "auxiliary", 1, "def train_model(): 训练模型的代码"),
        ("/course/food.md", "md", "auxiliary", 1, "红烧肉需要先焯水再炒糖色调味。"),
    ]

    docs = [
        Document(
            page_content=content,
            metadata={
                "source_path": src,
                "source_name": Path(src).name,
                "file_type": ftype,
                "category": cat,
                "page": page,
                "total_pages": 2,
                "course_id": COURSE_ID,
            },
        )
        for src, ftype, cat, page, content in raw
    ]
    store.add_documents(fake_embedder, split_documents(docs))
    return store


@pytest.fixture
def retriever(hash_llm_settings, paths, fake_embedder, seeded) -> Retriever:
    return Retriever(COURSE_ID, hash_llm_settings, paths, embedder=fake_embedder)


# ============================================================
# 基本检索
# ============================================================
class TestRetrieve:
    def test_returns_relevant(self, retriever) -> None:
        """⭐ 查"机器学习"，最相关的应排第一。"""
        results = retriever.retrieve("机器学习", k=1)

        assert len(results) == 1
        assert "机器学习" in results[0].page_content

    def test_ranked_by_score(self, retriever) -> None:
        results = retriever.retrieve_with_scores("机器学习", k=4)
        scores = [r.score for r in results]

        assert scores == sorted(scores, reverse=True)
        assert [r.rank for r in results] == [1, 2, 3, 4]

    def test_score_in_metadata(self, retriever) -> None:
        doc = retriever.retrieve("机器学习", k=1)[0]
        assert "score" in doc.metadata

    def test_empty_query(self, retriever) -> None:
        assert retriever.retrieve("") == []
        assert retriever.retrieve("   ") == []

    def test_empty_store(self, hash_llm_settings, paths, fake_embedder) -> None:
        empty = Retriever("nobody", hash_llm_settings, paths, embedder=fake_embedder)
        assert empty.retrieve("任意查询") == []
        assert empty.is_ready() is False

    def test_is_ready(self, retriever) -> None:
        assert retriever.is_ready() is True

# ============================================================
# 过滤
# ============================================================
class TestFilters:
    def test_where_single_condition(self) -> None:
        assert Retriever._build_where("main", None) == {"category": "main"}

    def test_where_multiple_uses_and(self) -> None:
        """⚠️ 多条件必须包在 $and 里，否则 Chroma 报错。"""
        where = Retriever._build_where("main", ["pdf"])
        assert "$and" in where

    def test_where_none(self) -> None:
        assert Retriever._build_where(None, None) is None

    def test_file_type_normalized(self) -> None:
        where = Retriever._build_where(None, [".PDF", "MD"])
        assert where == {"file_type": {"$in": ["pdf", "md"]}}

    def test_filter_by_category(self, retriever) -> None:
        results = retriever.retrieve("学习", k=10, category="main")
        assert all(d.metadata["category"] == "main" for d in results)
        assert len(results) >= 1

    def test_filter_by_file_type(self, retriever) -> None:
        results = retriever.retrieve("训练", k=10, file_types=["py"])
        assert all(d.metadata["file_type"] == "py" for d in results)

    def test_min_score_filters(self, retriever) -> None:
        """阈值设到极高时应该一条都不返回。"""
        assert retriever.retrieve("机器学习", k=5, min_score=0.99) == []

    def test_max_per_source(self, retriever) -> None:
        """同一个 PDF 有 2 页，限制每源 1 条时只能出 1 条。"""
        results = retriever.retrieve_with_scores(
            "学习", k=5, max_per_source=1
        )
        sources = [r.source_path for r in results]
        assert len(sources) == len(set(sources))

# ============================================================
# RetrievalResult
# ============================================================
class TestRetrievalResult:
    def _make(self, **meta) -> RetrievalResult:
        base = {"source_name": "a.pdf", "source_path": "/a.pdf", "page": 3}
        base.update(meta)
        return RetrievalResult(
            document=Document(page_content="内容", metadata=base),
            score=0.8,
            rank=1,
        )

    def test_citation_with_page(self) -> None:
        r = self._make(total_pages=10)
        assert r.citation() == "a.pdf 第 3 页"

    def test_citation_with_lines(self) -> None:
        """有行号时优先显示行号（比页码精确）。"""
        r = self._make(line_start=10, line_end=25)
        assert r.citation() == "a.pdf 第 10-25 行"

    def test_citation_single_page_file(self) -> None:
        """单页文件不显示"第 1 页"（是噪音）。"""
        r = self._make(total_pages=1)
        assert r.citation() == "a.pdf"

    def test_properties(self) -> None:
        r = self._make()
        assert r.source_name == "a.pdf"
        assert r.page == 3
        assert r.content == "内容"

# ============================================================
# 上下文拼装
# ============================================================
class TestBuildContext:
    def test_numbered_blocks(self, retriever) -> None:
        context, used = retriever.build_context("机器学习", k=2)
        print(1)
        assert "[1]" in context
        assert len(used) <= 2

    def test_respects_max_chars(self, retriever) -> None:
        _context, used = retriever.build_context("学习", k=4, max_chars=60)

        assert len(used) >= 1
        # 允许第一块超限（至少要给一块），但不能塞满 4 块
        assert len(used) < 4

    def test_empty_when_no_hits(self, hash_llm_settings, paths, fake_embedder) -> None:
        empty = Retriever("nobody", hash_llm_settings, paths, embedder=fake_embedder)
        context, used = empty.build_context("查询")
        assert context == ""
        assert used == []

# # ============================================================
# # 混合检索
# # ============================================================
# class TestHybridRetriever:
#     def test_finds_exact_keyword(
#         self, hash_llm_settings, paths, fake_embedder, seeded
#     ) -> None:
#         hybrid = HybridRetriever(
#             COURSE_ID, hash_llm_settings, paths, embedder=fake_embedder
#         )
#         results = hybrid.retrieve("train_model", k=3)

#         assert len(results) >= 1
#         assert any("train_model" in d.page_content for d in results)

#     def test_rrf_score_attached(
#         self, hash_llm_settings, paths, fake_embedder, seeded
#     ) -> None:
#         hybrid = HybridRetriever(
#             COURSE_ID, hash_llm_settings, paths, embedder=fake_embedder
#         )
#         results = hybrid.retrieve("机器学习", k=2)
#         assert all("rrf_score" in d.metadata for d in results)

#     def test_no_duplicates(
#         self, hash_llm_settings, paths, fake_embedder, seeded
#     ) -> None:
#         """两路都召回的文档只能出现一次。"""
#         hybrid = HybridRetriever(
#             COURSE_ID, hash_llm_settings, paths, embedder=fake_embedder
#         )
#         results = hybrid.retrieve("机器学习", k=5)
#         ids = [d.metadata["chunk_id"] for d in results]
#         assert len(ids) == len(set(ids))

#     def test_empty_store(self, hash_llm_settings, paths, fake_embedder) -> None:
#         hybrid = HybridRetriever(
#             "nobody", hash_llm_settings, paths, embedder=fake_embedder
#         )
#         assert hybrid.retrieve("查询") == []
