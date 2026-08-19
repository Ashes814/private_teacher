"""vector_store 单元测试。

这些测试会真的建一个 Chroma 库（在 tmp_path 里），跑完自动删除。
不 mock Chroma 的原因：Chroma 的 API 细节（返回结构、距离定义）
正是最容易出错的地方，mock 掉就等于什么都没测。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.documents import Document

from private_teacher.rag.splitters import split_documents
from private_teacher.rag.vector_store import VectorStoreManager
from private_teacher.utils.exceptions import RAGError


def make_chunks(pairs: list[tuple[str, str]]) -> list[Document]:
    """辅助函数：从 [(源文件, 内容)] 造出带 chunk_id 的 chunk。

    走真正的 split_documents，保证 chunk_id 的生成逻辑和生产环境一致。
    """
    docs = [
        Document(
            page_content=content,
            metadata={
                "source_path": src,
                "source_name": Path(src).name,
                "file_type": Path(src).suffix.lstrip("."),
                "page": 1,
                "total_pages": 1,
            },
        )
        for src, content in pairs
    ]
    return split_documents(docs)


@pytest.fixture
def store(tmp_path: Path) -> VectorStoreManager:
    return VectorStoreManager("test_course", tmp_path)


# ============================================================
# 基础读写
# ============================================================
class TestAddAndSearch:
    def test_add_returns_ids(self, store, fake_embedder) -> None:
        chunks = make_chunks(
            [
                ("/a.txt", "Python 是一门解释型编程语言，语法简洁易读。"),
                ("/b.txt", "Java 是一门静态类型语言，广泛用于企业后端开发。"),
            ]
        )
        ids = store.add_documents(fake_embedder, chunks)

        assert len(ids) == 2
        assert store.count() == 2

    def test_search_finds_relevant(self, store, fake_embedder) -> None:
        """⭐ 核心能力：查"编程语言"应该把相关文本排在前面。"""
        chunks = make_chunks(
            [
                ("/a.txt", "Python 是一门解释型编程语言，语法简洁易读。"),
                ("/b.txt", "今天天气晴朗，适合出门散步和野餐。"),
            ]
        )
        store.add_documents(fake_embedder, chunks)

        results = store.similarity_search(fake_embedder, "编程语言", k=1)

        assert len(results) == 1
        assert "Python" in results[0].page_content

    def test_search_returns_scores(self, store, fake_embedder) -> None:
        store.add_documents(fake_embedder, make_chunks([("/a.txt", "机器学习基础")]))

        results = store.query(fake_embedder, "机器学习", k=1)

        assert len(results) == 1
        _doc, score = results[0]
        # 用 cosine 空间，完全相同的文本相似度应接近 1
        assert 0.0 < score <= 1.001

    def test_results_sorted_by_score(self, store, fake_embedder) -> None:
        """返回结果必须按相似度降序。"""
        store.add_documents(
            fake_embedder,
            make_chunks(
                [
                    ("/a.txt", "机器学习是人工智能的重要分支领域。"),
                    ("/b.txt", "机器学习模型需要大量训练数据支撑。"),
                    ("/c.txt", "红烧肉的做法是先焯水再炒糖色。"),
                ]
            ),
        )

        results = store.query(fake_embedder, "机器学习", k=3)
        scores = [s for _, s in results]

        assert scores == sorted(scores, reverse=True)

    def test_empty_store_returns_empty(self, store, fake_embedder) -> None:
        """空库检索不能崩，应该返回空列表。"""
        assert store.similarity_search(fake_embedder, "任意查询") == []

    def test_add_empty_list(self, store, fake_embedder) -> None:
        assert store.add_documents(fake_embedder, []) == []

    def test_k_larger_than_count(self, store, fake_embedder) -> None:
        """k 大于库里总数时应返回全部，而不是报错。"""
        store.add_documents(fake_embedder, make_chunks([("/a.txt", "只有一条数据")]))
        assert len(store.similarity_search(fake_embedder, "数据", k=100)) == 1


# ============================================================
# upsert 幂等性
# ============================================================
class TestUpsert:
    def test_reindex_does_not_duplicate(self, store, fake_embedder) -> None:
        """⭐ 同样的 chunk 写两次，数量不应翻倍（upsert 语义）。"""
        chunks = make_chunks([("/a.txt", "重复索引测试内容，这段文字会被写两次。")])

        store.add_documents(fake_embedder, chunks)
        n1 = store.count()
        store.add_documents(fake_embedder, chunks)
        n2 = store.count()

        assert n1 == n2

    def test_duplicate_ids_in_same_batch(self, store, fake_embedder) -> None:
        """同一批里有重复 id 时要自动去重，不能让 Chroma 报错。"""
        chunks = make_chunks([("/a.txt", "内容甲乙丙丁戊己庚辛壬癸")])
        doubled = chunks + chunks  # 故意重复

        ids = store.add_documents(fake_embedder, doubled)

        assert len(ids) == len(chunks)

    def test_missing_chunk_id_raises(self, store, fake_embedder) -> None:
        """没经过 splitter 的 Document 不允许直接写入。"""
        bad = [Document(page_content="没有 chunk_id", metadata={"page": 1})]

        with pytest.raises(RAGError, match="chunk_id"):
            store.add_documents(fake_embedder, bad)


# ============================================================
# 元数据过滤
# ============================================================
class TestMetadataFilter:
    def test_filter_by_file_type(self, store, fake_embedder) -> None:
        store.add_documents(
            fake_embedder,
            make_chunks(
                [
                    ("/a.pdf", "机器学习课件第一章内容介绍。"),
                    ("/b.py", "机器学习代码示例 def train(): pass"),
                ]
            ),
        )

        results = store.similarity_search(
            fake_embedder, "机器学习", k=10, where={"file_type": "pdf"}
        )

        assert len(results) == 1
        assert results[0].metadata["file_type"] == "pdf"

    def test_metadata_preserved(self, store, fake_embedder) -> None:
        """写进去的 metadata 必须能原样取回来（溯源功能依赖它）。"""
        store.add_documents(
            fake_embedder, make_chunks([("/data/course/a.pdf", "第一页的内容文本")])
        )

        doc = store.similarity_search(fake_embedder, "内容", k=1)[0]

        assert doc.metadata["source_path"] == "/data/course/a.pdf"
        assert doc.metadata["source_name"] == "a.pdf"
        assert doc.metadata["page"] == 1
        assert "chunk_id" in doc.metadata


# ============================================================
# 删除
# ============================================================
class TestDelete:
    def test_delete_by_source(self, store, fake_embedder) -> None:
        store.add_documents(
            fake_embedder,
            make_chunks([("/a.txt", "文件甲的内容部分"), ("/b.txt", "文件乙的内容部分")]),
        )
        assert store.count() == 2

        # ⚠️ delete_by_source 内部会 resolve() 路径，
        #    所以这里传的路径要和写入时 metadata 里的一致
        deleted = store.delete_by_source(Path("/a.txt"))

        assert deleted == 1
        assert store.count() == 1

    def test_delete_collection(self, store, fake_embedder) -> None:
        store.add_documents(fake_embedder, make_chunks([("/a.txt", "一些内容文本")]))
        store.delete_collection()
        assert store.count() == 0

    def test_delete_missing_collection_is_safe(self, store) -> None:
        """删一个不存在的 collection 不应该抛异常（幂等）。"""
        store.delete_collection()
        store.delete_collection()  # 再删一次

    def test_reset(self, store, fake_embedder) -> None:
        store.add_documents(fake_embedder, make_chunks([("/a.txt", "内容内容内容")]))
        store.reset()
        assert store.count() == 0
        # reset 后应该能立刻继续写
        store.add_documents(fake_embedder, make_chunks([("/b.txt", "新的内容文本")]))
        assert store.count() == 1


# ============================================================
# 持久化 & 辅助
# ============================================================
class TestPersistence:
    def test_survives_new_manager(self, tmp_path: Path, fake_embedder) -> None:
        """⭐ 写完之后新建一个 manager，数据必须还在（持久化生效）。"""
        m1 = VectorStoreManager("c1", tmp_path)
        m1.add_documents(fake_embedder, make_chunks([("/a.txt", "持久化测试内容")]))

        m2 = VectorStoreManager("c1", tmp_path)  # 模拟重启进程

        assert m2.count() == 1

    def test_courses_isolated(self, tmp_path: Path, fake_embedder) -> None:
        """不同课程的向量互不可见。"""
        a = VectorStoreManager("course_a", tmp_path)
        b = VectorStoreManager("course_b", tmp_path)

        a.add_documents(fake_embedder, make_chunks([("/a.txt", "甲课程的内容")]))

        assert a.count() == 1
        assert b.count() == 0


class TestHelpers:
    def test_peek(self, store, fake_embedder) -> None:
        store.add_documents(
            fake_embedder,
            make_chunks([("/a.txt", "内容一二三"), ("/b.txt", "内容四五六")]),
        )
        assert len(store.peek(n=10)) == 2

    def test_list_sources(self, store, fake_embedder) -> None:
        store.add_documents(
            fake_embedder,
            make_chunks([("/a.txt", "内容甲甲甲"), ("/b.txt", "内容乙乙乙")]),
        )
        assert store.list_sources() == ["/a.txt", "/b.txt"]

    def test_count_on_fresh_store(self, tmp_path: Path) -> None:
        assert VectorStoreManager("brand_new", tmp_path).count() == 0
