"""
检索器
======

职责：把用户的一句话，变成 top-k 个最相关的课件片段。

【为什么不直接调 VectorStoreManager.similarity_search？】
因为真实检索需要在"向量搜索"之外做这些事：
  1. 过滤：只搜主干课件 / 只搜 PDF
  2. 超采样 + 去重：一个文件可能霸占全部 5 个名额，
     用户其实更想看到 3 个不同来源的结果
  3. 分数阈值：相似度 0.1 的结果与其给出来误导 LLM，不如不给
  4. 上下文拼装：把 top-k 拼成一段带来源标注的文本，直接喂给 LLM

这些都是"检索策略"，不该塞进向量库封装层里。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from loguru import logger

from private_teacher.config import LLMSettings, PathSettings
from private_teacher.rag.embeddings import Embedder, get_cached_embedder
from private_teacher.rag.vector_store import VectorStoreManager


@dataclass(slots=True)
class RetrievalResult:
    """一条检索结果。"""

    document: Document
    score: float  # 相似度，越大越相关
    rank: int  # 名次，从 1 开始

    # ---------- 便捷属性（UI 直接用，不用到处翻 metadata） ----------
    @property
    def content(self) -> str:
        return self.document.page_content

    @property
    def source_name(self) -> str:
        return str(self.document.metadata.get("source_name", "未知文件"))

    @property
    def source_path(self) -> Path:
        return Path(str(self.document.metadata.get("source_path", "")))

    @property
    def page(self) -> int:
        return int(self.document.metadata.get("page", 1))

    @property
    def chunk_id(self) -> str:
        return str(self.document.metadata.get("chunk_id", ""))

    def citation(self) -> str:
        """生成人类可读的引用标记，如 "机器学习.pdf 第 12 页"。"""
        parts = [self.source_name]

        # 有行号的（代码/文本）优先显示行号，比页码更精确
        line_start = self.document.metadata.get("line_start")
        line_end = self.document.metadata.get("line_end")
        if line_start:
            parts.append(f"第 {line_start}-{line_end} 行")
        elif int(self.document.metadata.get("total_pages", 1)) > 1:
            parts.append(f"第 {self.page} 页")

        return " ".join(parts)


class Retriever:
    """某门课的向量检索器。"""

    def __init__(
        self,
        course_id: str,
        llm_settings: LLMSettings,
        paths: PathSettings,
        embedder: Embedder | None = None,
    ) -> None:
        self.course_id = course_id
        self.llm = llm_settings
        self._embedder = embedder or get_cached_embedder(llm_settings)
        self.store = VectorStoreManager(course_id, paths.data_dir)

    # ============================================================
    # 过滤条件构造
    # ============================================================
    @staticmethod
    def _build_where(
        category: str | None,
        file_types: list[str] | None,
    ) -> dict[str, Any] | None:
        """把易用的参数翻译成 Chroma 的 where 语法。

        Chroma 的规则（很容易踩坑）：
          - 单条件：{"category": "main"}
          - 多条件：必须显式 {"$and": [{...}, {...}]}，
            **不能**写成 {"category": "main", "file_type": "pdf"}（会报错）
          - 集合匹配：{"file_type": {"$in": ["pdf", "md"]}}
        """
        conditions: list[dict[str, Any]] = []

        if category:
            conditions.append({"category": category})

        if file_types:
            # 统一小写、去掉可能带的点：".PDF" → "pdf"
            normalized = [t.lower().lstrip(".") for t in file_types]
            conditions.append({"file_type": {"$in": normalized}})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    # ============================================================
    # 核心检索
    # ============================================================
    def retrieve_with_scores(
        self,
        query: str,
        k: int = 5,
        category: str | None = None,
        file_types: list[str] | None = None,
        min_score: float | None = None,
        max_per_source: int | None = None,
    ) -> list[RetrievalResult]:
        """检索并返回带分数的结果。

        Args:
            query: 查询语句
            k: 最终返回条数
            category: "main" / "auxiliary"，None = 不限
            file_types: 如 ["pdf", "md"]，None = 不限
            min_score: 相似度下限，低于它的结果直接丢弃。
                建议值 0.2~0.3（cosine）。设太高会经常一条都搜不到
            max_per_source: 同一个源文件最多返回几条。
                设成 2 可以避免"一个文件霸占全部结果"

        Returns:
            按相关度降序的 RetrievalResult 列表
        """
        if not query.strip():
            return []

        where = self._build_where(category, file_types)

        # ---------- 超采样 ----------
        # 因为后面要按 min_score 和 max_per_source 过滤掉一部分，
        # 所以先多取一些，保证过滤完还能凑够 k 条。
        # 3 倍是经验值，上限 50 防止 k 很大时把整个库都拉出来
        fetch_k = min(k * 3, 50) if (min_score or max_per_source) else k

        raw = self.store.query(self._embedder, query, k=fetch_k, where=where)

        results: list[RetrievalResult] = []
        per_source: dict[str, int] = {}

        for doc, score in raw:
            # ---------- 分数过滤 ----------
            if min_score is not None and score < min_score:
                # 结果已按分数降序，一旦低于阈值，后面的只会更低 → 直接停
                break

            # ---------- 来源配额 ----------
            if max_per_source is not None:
                src = str(doc.metadata.get("source_path", ""))
                if per_source.get(src, 0) >= max_per_source:
                    continue
                per_source[src] = per_source.get(src, 0) + 1

            # 分数也放进 metadata 一份，方便直接传给只认 Document 的下游
            doc.metadata["score"] = round(score, 4)

            results.append(RetrievalResult(document=doc, score=score, rank=len(results) + 1))

            if len(results) >= k:
                break

        logger.debug(f"检索 '{query[:30]}' → {len(results)} 条结果")
        return results

    def retrieve(self, query: str, k: int = 5, **kwargs: Any) -> list[Document]:
        """只要 Document 的简化版（分数在 metadata["score"] 里）。"""
        return [r.document for r in self.retrieve_with_scores(query, k, **kwargs)]

    # ============================================================
    # 给 LLM 用的上下文拼装
    # ============================================================
    def build_context(
        self,
        query: str,
        k: int = 5,
        max_chars: int = 4000,
        **kwargs: Any,
    ) -> tuple[str, list[RetrievalResult]]:
        """检索并拼成可直接塞进 prompt 的上下文字符串。

        Args:
            max_chars: 上下文长度上限。超过就截断 ——
                盲目把 20 个 chunk 全塞进 prompt 会爆 token 且稀释重点

        Returns:
            (上下文文本, 用到的结果列表)

        输出格式：
            [1] 机器学习.pdf 第 3 页
            决策树是一种树形结构的分类模型……

            [2] notes.md
            ……

        为什么要编号 [1] [2]？
          → 让 LLM 能在回答里写"根据 [1]"，我们就能把它映射回真实来源，
            这是"可溯源问答"的基础（Phase 2 会重度依赖）。
        """
        results = self.retrieve_with_scores(query, k, **kwargs)
        if not results:
            return "", []

        blocks: list[str] = []
        used: list[RetrievalResult] = []
        total = 0

        for i, r in enumerate(results, start=1):
            block = f"[{i}] {r.citation()}\n{r.content}"

            # 加上这一块会超长 → 停止（保证 prompt 不失控）
            if total + len(block) > max_chars and used:
                break

            blocks.append(block)
            used.append(r)
            total += len(block)

        return "\n\n".join(blocks), used

    # ============================================================
    # 状态
    # ============================================================
    def is_ready(self) -> bool:
        """知识库里有没有数据。UI 上用它决定是否禁用搜索框。"""
        return self.store.count() > 0


# rank-bm25 已经在依赖里了，这里给出完整实现。
# 建议 Phase 1 先只用向量检索，把全链路跑通再回来加这一层。
# 追加到 src/private_teacher/rag/retriever.py 末尾
# class HybridRetriever:
#     """向量检索 + BM25 关键词检索，用 RRF 融合。

#     【为什么要混合？】
#       向量检索懂语义但对精确词不敏感：
#         搜 "ResNet50"，可能返回一堆讲"深度网络"的段落，却漏掉真正提到 ResNet50 的那句
#       BM25 精确匹配强但不懂同义：
#         搜 "如何评估模型"，匹配不到只写了"模型评价指标"的段落
#       两者的失败模式正好互补，所以合起来效果最好。

#     【RRF（Reciprocal Rank Fusion，倒数排名融合）】
#         score(doc) = Σ  1 / (k + rank_i(doc))
#                     各路召回
#       k 通常取 60。它的妙处是**只看排名不看分数** ——
#       向量的余弦相似度（0~1）和 BM25 的分数（0~几十）量纲完全不同，
#       直接加权求和需要繁琐的归一化，而 RRF 天然免疫这个问题。
#       """

#     RRF_K = 60

#     def __init__(
#         self,
#         course_id: str,
#         llm_settings: LLMSettings,
#         paths: PathSettings,
#         embedder: Embedder | None = None,
#     ) -> None:
#         self.vector = Retriever(course_id, llm_settings, paths, embedder)
#         self.store = self.vector.store
#         # BM25 索引是懒加载的：它需要把全库文本读进内存，
#         # 只有真的用到混合检索时才付这个代价
#         self._bm25: Any = None
#         self._bm25_docs: list[Document] = []

#     # ---------- BM25 索引 ----------
#     @staticmethod
#     def _tokenize(text: str) -> list[str]:
#         """分词。

#         中文没有空格，这里用最朴素的"单字 + bigram"方案，
#         效果比整句当一个 token 好得多。
#         要更好的效果可以引入 jieba：`uv add jieba`，然后
#             return list(jieba.cut_for_search(text))
#         """
#         import re

#         words = re.findall(r"[a-zA-Z0-9_]+", text.lower())
#         cjk = [c for c in text if "一" <= c <= "鿿"]
#         bigrams = [a + b for a, b in zip(cjk, cjk[1:], strict=False)]
#         return words + cjk + bigrams

#     def _ensure_bm25(self) -> None:
#         """构建 BM25 索引（懒加载 + 缓存）。"""
#         if self._bm25 is not None:
#             return

#         from rank_bm25 import BM25Okapi

#         # 把整个 collection 拉出来。
#         # ⚠️ 库很大时这会很占内存 —— 十万级 chunk 就该换 Elasticsearch 了
#         collection = self.store.get_collection()
#         raw = collection.get(include=["documents", "metadatas"])

#         self._bm25_docs = [
#             Document(page_content=text, metadata=dict(meta or {}))
#             for text, meta in zip(
#                 raw.get("documents") or [],
#                 raw.get("metadatas") or [],
#                 strict=False,
#             )
#         ]

#         if not self._bm25_docs:
#             self._bm25 = None
#             return

#         corpus = [self._tokenize(d.page_content) for d in self._bm25_docs]
#         self._bm25 = BM25Okapi(corpus)

#     def invalidate(self) -> None:
#         """索引更新后调用，强制下次重建 BM25。"""
#         self._bm25 = None
#         self._bm25_docs = []

#     # ---------- 检索 ----------
#     def _bm25_search(self, query: str, k: int) -> list[Document]:
#         self._ensure_bm25()
#         if self._bm25 is None:
#             return []

#         scores = self._bm25.get_scores(self._tokenize(query))

#         # 取分数最高的 k 个下标：
#         # enumerate 配对成 (下标, 分数)，按分数降序排，取前 k
#         ranked = sorted(enumerate(scores), key=lambda pair: pair[1], reverse=True)
#         return [self._bm25_docs[i] for i, s in ranked[:k] if s > 0]

#     def retrieve(self, query: str, k: int = 5, **kwargs: Any) -> list[Document]:
#         """混合检索。"""
#         # 两路各取 2k 条，融合后再截断到 k
#         vec_docs = self.vector.retrieve(query, k=k * 2, **kwargs)
#         bm25_docs = self._bm25_search(query, k=k * 2)

#         # ---------- RRF 融合 ----------
#         fused: dict[str, float] = {}
#         by_id: dict[str, Document] = {}

#         for rank, doc in enumerate(vec_docs, start=1):
#             cid = str(doc.metadata.get("chunk_id", id(doc)))
#             fused[cid] = fused.get(cid, 0.0) + 1.0 / (self.RRF_K + rank)
#             by_id[cid] = doc

#         for rank, doc in enumerate(bm25_docs, start=1):
#             cid = str(doc.metadata.get("chunk_id", id(doc)))
#             fused[cid] = fused.get(cid, 0.0) + 1.0 / (self.RRF_K + rank)
#             by_id.setdefault(cid, doc)

#         ordered = sorted(fused.items(), key=lambda pair: pair[1], reverse=True)

#         results: list[Document] = []
#         for cid, score in ordered[:k]:
#             doc = by_id[cid]
#             doc.metadata["rrf_score"] = round(score, 6)
#             results.append(doc)

#         return results
