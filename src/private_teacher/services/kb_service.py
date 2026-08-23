"""
知识库服务
==========

对外提供三个动作：**建库 / 检索 / 溯源**。

它是 Phase 2 Agent 层唯一需要认识的入口 ——
Agent 只要会调 `kb.search_with_sources(...)`，就拥有了完整的 RAG 能力。
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Any

from loguru import logger

from private_teacher.config import LLMSettings, PathSettings
from private_teacher.rag.embeddings import Embedder, get_cached_embedder
from private_teacher.rag.indexer import Indexer, IndexResult
from private_teacher.rag.retriever import RetrievalResult, Retriever
from private_teacher.rag.source_resolver import SourceLocation, SourceResolver
from private_teacher.rag.splitters import ChunkerConfig

@dataclass(slots=True)
class SearchHit:
    """检索结果 + 溯源信息，UI 直接渲染这个对象。

    为什么要把 RetrievalResult 和 SourceLocation 打包在一起？
      → UI 每显示一条结果，都同时需要"内容 + 分数"和"来自哪里"。
        让 UI 自己去调两个方法再配对，既啰嗦又容易出错。
    """

    result: RetrievalResult
    location: SourceLocation

    @property
    def content(self) -> str:
        return self.result.content

    @property
    def score(self) -> float:
        return self.result.score

    @property
    def label(self) -> str:
        return self.location.label()
    
class KBService:
    """知识库服务。"""

    def __init__(
        self,
        llm: LLMSettings,
        paths: PathSettings,
        chunker_config: ChunkerConfig | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self.llm = llm
        self.paths = paths
        self.chunker_config = chunker_config or ChunkerConfig()
        # 允许注入 embedder（测试用）；否则用带缓存的工厂
        self._embedder = embedder or get_cached_embedder(llm)

    # ---------- 内部工厂 ----------
    # 为什么每次都新建 Indexer/Retriever，而不缓存成实例属性？
    #   → 它们很轻（真正重的 embedder 已经共享了），
    #     而且 KBService 可能同时服务多门课，缓存反而要处理 course_id 切换
    def _indexer(self, course_id: str) -> Indexer:
        return Indexer(
            course_id, self.llm, self.paths, self.chunker_config, self._embedder
        )

    def _retriever(self, course_id: str) -> Retriever:
        return Retriever(course_id, self.llm, self.paths, self._embedder)

    def _resolver(self, course_id: str) -> SourceResolver:
        return SourceResolver(course_id, self.paths)
    
    # ============================================================
    # 构建
    # ============================================================
    def build_kb(
        self,
        conn: sqlite3.Connection,
        course_id: str,
        incremental: bool = True,
        on_file: Any = None,
    ) -> IndexResult:
        """构建知识库。

        Args:
            incremental: True=增量（默认，快），False=全量重建
            on_file: 进度回调 (序号, 总数, 文件名)
        """
        indexer = self._indexer(course_id)

        if incremental:
            return indexer.build_incremental(conn, on_file=on_file)
        return indexer.build_full(conn, on_file=on_file)
    
    def stats(self, conn: sqlite3.Connection, course_id: str) -> dict[str, Any]:
        return self._indexer(course_id).stats(conn)

    def is_ready(self, course_id: str) -> bool:
        """知识库能不能用于检索。"""
        return self._retriever(course_id).is_ready()
    
    # ============================================================
    # 检索
    # ============================================================
    def search(
        self,
        course_id: str,
        query: str,
        k: int = 5,
        **kwargs: Any,
    ) -> list[RetrievalResult]:
        """检索（不含溯源）。"""
        return self._retriever(course_id).retrieve_with_scores(query, k, **kwargs)

    def search_with_sources(
        self,
        course_id: str,
        query: str,
        k: int = 5,
        **kwargs: Any,
    ) -> list[SearchHit]:
        """⭐ 检索 + 溯源，UI 和 Agent 的主入口。"""
        results = self.search(course_id, query, k, **kwargs)
        resolver = self._resolver(course_id)

        return [
            SearchHit(result=r, location=resolver.resolve(r.document))
            for r in results
        ]
    
    def resolve_sources(
        self,
        course_id: str,
        chunks: list[Any],
    ) -> list[SourceLocation]:
        """给一批 chunk 做溯源（兼容旧接口）。"""
        return self._resolver(course_id).resolve_many(chunks)
    
    # ============================================================
    # 给 LLM 用
    # ============================================================
    def build_context(
        self,
        course_id: str,
        query: str,
        k: int = 5,
        max_chars: int = 4000,
        **kwargs: Any,
    ) -> tuple[str, list[RetrievalResult]]:
        """检索并拼成带编号引用的上下文（Phase 2 Agent 会大量使用）。"""
        return self._retriever(course_id).build_context(
            query, k, max_chars, **kwargs
        )

    # ============================================================
    # 维护
    # ============================================================
    def clear_kb(self, conn: sqlite3.Connection, course_id: str) -> None:
        """清空知识库（保留课件文件本身）。"""
        indexer = self._indexer(course_id)
        indexer.store.reset()

        from private_teacher.rag.manifest import IndexManifest
        from private_teacher.storage import document_repo

        IndexManifest(course_id=course_id).save(self.paths.data_dir)
        document_repo.mark_all_unindexed(conn, course_id)

        logger.info(f"知识库已清空：{course_id}")