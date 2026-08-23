"""
向量库封装（Chroma）
====================

【一个 collection = 一门课】
collection 名字用 "course_{course_id}"，每门课的向量物理隔离在自己的目录里。
好处：删课程 = 删一个目录；一门课的索引损坏不会影响其他课。

【为什么用 upsert 而不是 add】
add：id 已存在会抛异常
upsert：id 已存在则覆盖
我们的 chunk_id 是内容稳定的（见 §4），所以重复索引同一个文件时，
upsert 会原地更新而不是产生重复向量 —— 这是增量索引能正确工作的基础。

【距离与相似度】
Chroma 默认用 L2（欧氏距离），我们显式改成 cosine（余弦距离）：
  - 文本 embedding 的行业惯例就是余弦相似度
  - 余弦距离范围 [0, 2]，相似度 = 1 - 距离，范围 [-1, 1]，直观好解释
  - ⚠️ space 是**建 collection 时**定死的，之后改不了。改了就得重建库。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from loguru import logger

from private_teacher.loaders.base import META_CHUNK_ID, META_SOURCE_PATH, sanitize_metadata
from private_teacher.rag.embeddings import Embedder, embed_in_batches
from private_teacher.utils.exceptions import RAGError


class VectorStoreManager:
    """管理某一门课的 Chroma collection。

    典型用法：
        mgr = VectorStoreManager("a1b2c3d4", Path("data"))
        mgr.add_documents(embedder, chunks)
        hits = mgr.similarity_search(embedder, "机器学习", k=5)
    """

    def __init__(self, course_id: str, data_dir: Path) -> None:
        """
        Args:
            course_id: 课程 id
            data_dir: 数据根目录（PathSettings.data_dir）
        """
        self.course_id = course_id
        self.persist_dir = data_dir / "knowledge_bases" / course_id / "chroma"
        # PersistentClient 要求目录存在
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.collection_name = f"course_{course_id}"
        # 延迟初始化：只有真正用到时才创建 client，
        # 因为 chromadb 首次 import + 建 client 有几百毫秒开销
        self._client: Any = None

    # ============================================================
    # 底层 client / collection
    # ============================================================
    @property
    def client(self) -> Any:
        """惰性创建 Chroma 持久化客户端。"""
        if self._client is None:
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            self._client = chromadb.PersistentClient(
                path=str(self.persist_dir),
                settings=ChromaSettings(
                    # 关掉匿名遥测 —— 本项目是 local-first，不该往外发任何数据
                    anonymized_telemetry=False,
                    # 允许 reset()，测试里要用
                    allow_reset=True,
                ),
            )
        return self._client

    def get_collection(self) -> Any:
        """取（或创建）collection。

        get_or_create 是幂等的：存在就返回，不存在就建。
        比 try get / except create 干净。
        """
        return self.client.get_or_create_collection(
            name=self.collection_name,
            # ⚠️ hnsw:space 只在**创建时**生效。
            #    如果 collection 已存在且是 l2，这里传 cosine 也不会改变它。
            #    换距离度量的唯一办法是删掉重建。
            metadata={"hnsw:space": "cosine"},
        )

    # ============================================================
    # 写入
    # ============================================================
    def add_documents(
        self,
        embedder: Embedder,
        chunks: list[Document],
        batch_size: int = 64,
        on_progress: Any = None,
    ) -> list[str]:
        """把 chunk 写入向量库（upsert 语义）。

        Args:
            embedder: 向量化器
            chunks: split_documents 的产物（metadata 里必须有 chunk_id）
            batch_size: 向量化批大小
            on_progress: 进度回调 (done, total)

        Returns:
            实际写入的 id 列表

        Raises:
            RAGError: 缺少 chunk_id / 向量化失败 / 写入失败
        """
        if not chunks:
            return []

        # ---------- 1. 取出三份平行数组 ----------
        ids: list[str] = []
        texts: list[str] = []
        metadatas: list[dict[str, Any]] = []

        for chunk in chunks:
            chunk_id = chunk.metadata.get(META_CHUNK_ID)
            if not chunk_id:
                # 没有 chunk_id 说明没走 split_documents，是调用方的 bug。
                # 与其静默生成随机 id（后患无穷），不如立刻报错
                raise RAGError(
                    "chunk 缺少 chunk_id，请先经过 split_documents 处理",
                    source=chunk.metadata.get(META_SOURCE_PATH, "?"),
                )

            ids.append(str(chunk_id))
            texts.append(chunk.page_content)
            # sanitize：把 None / Path 等 Chroma 不认的值清洗掉
            metadatas.append(sanitize_metadata(chunk.metadata))

        # ---------- 2. 去重 ----------
        # 同一批里出现重复 id，Chroma 会报错。
        # 正常不该发生，但如果调用方把同一个文件加载了两次就会 —— 这里兜一下底
        seen: set[str] = set()
        keep: list[int] = []
        for i, cid in enumerate(ids):
            if cid not in seen:
                seen.add(cid)
                keep.append(i)
        if len(keep) != len(ids):
            logger.warning(f"发现 {len(ids) - len(keep)} 个重复 chunk_id，已去重")
            ids = [ids[i] for i in keep]
            texts = [texts[i] for i in keep]
            metadatas = [metadatas[i] for i in keep]

        # ---------- 3. 向量化 ----------
        vectors = embed_in_batches(embedder, texts, batch_size, on_progress)

        # ---------- 4. 写入 ----------
        collection = self.get_collection()
        try:
            # upsert：id 已存在则覆盖，不存在则插入
            collection.upsert(
                ids=ids,
                documents=texts,
                metadatas=metadatas,
                embeddings=vectors,
            )

        except Exception as exc:
            raise RAGError(
                f"写入向量库失败: {exc}", course_id=self.course_id, count=len(ids)
            ) from exc

        logger.info(f"向量库写入 {len(ids)} 条（course={self.course_id}）")
        return ids

    # ============================================================
    # 检索
    # ============================================================
    def query(
        self,
        embedder: Embedder,
        query_text: str,
        k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[tuple[Document, float]]:
        """相似度检索，返回 (Document, 相似度) 列表。

        Args:
            embedder: 向量化器（用它的 embed_query，不是 embed_documents！）
            query_text: 查询语句
            k: 返回条数
            where: metadata 过滤条件，例如 {"file_type": "pdf"}。
                多条件要用 Chroma 的语法：
                    {"$and": [{"file_type": "pdf"}, {"page": {"$gte": 10}}]}

        Returns:
            [(Document, similarity), ...]，按相似度从高到低。
            similarity = 1 - cosine_distance，范围约 [-1, 1]，越大越像。

        为什么返回相似度而不只是 Document？
          - UI 上要显示"匹配度 87%"
          - Phase 6 的 rerank / 阈值过滤需要分数
        """
        # ---------- 防御性检查 ----------
        # 课程刚被删 / 从没建过索引：persist_dir 可能是空的或不存在，
        # 这时调 chromadb 的 get_or_create 会触发底层 sqlite 的
        # "readonly database" 错误（macOS 文件系统偶发）。
        # 这种情况直接返回空，让上层优雅处理。
        if not self.persist_dir.exists() or not any(self.persist_dir.iterdir()):
            return []

        collection = self.get_collection()

        # 空库直接返回，否则 Chroma 在某些版本会报错
        if collection.count() == 0:
            logger.warning(f"知识库为空（course={self.course_id}）")
            return []

        query_vector = embedder.embed_query(query_text)

        try:
            raw = collection.query(
                query_embeddings=[query_vector],
                # n_results 超过库里实际条数时 Chroma 会自动截断，不会报错
                n_results=k,
                where=where,
                # include 决定返回哪些字段。
                # 不写 "embeddings" —— 向量很大，传回来纯属浪费内存
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise RAGError(
                f"向量检索失败: {exc}",
                course_id=self.course_id,
                query=query_text[:50],
            ) from exc

        # ---------- 解析返回结构 ----------
        # Chroma 支持一次查多个 query，所以每个字段都是"二维"的：
        #   raw["documents"] = [[第1个query的结果...]]
        # 我们只查了一个，取 [0]
        docs_list = raw.get("documents") or [[]]
        metas_list = raw.get("metadatas") or [[]]
        dists_list = raw.get("distances") or [[]]

        results: list[tuple[Document, float]] = []
        for text, meta, dist in zip(docs_list[0], metas_list[0], dists_list[0], strict=False):
            # 余弦距离 → 相似度
            similarity = 1.0 - float(dist)
            results.append(
                (
                    Document(page_content=text, metadata=dict(meta or {})),
                    similarity,
                )
            )

        return results

    def similarity_search(
        self,
        embedder: Embedder,
        query_text: str,
        k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[Document]:
        """只要 Document、不要分数的简化版（分数会塞进 metadata["score"]）。"""
        results = self.query(embedder, query_text, k, where)
        docs: list[Document] = []
        for doc, score in results:
            # round 到 4 位：分数是给人看的，不需要 17 位小数
            doc.metadata["score"] = round(score, 4)
            docs.append(doc)
        return docs

    # ============================================================
    # 删除与统计
    # ============================================================
    def count(self) -> int:
        """库里有多少条向量。UI 显示"知识库规模"用。"""
        try:
            return int(self.get_collection().count())
        except Exception:
            # collection 还没建 / 目录被手删了 —— 都当作 0
            return 0

    def delete_by_source(self, source_path: str | Path) -> int:
        """删除某个源文件产生的所有向量。

        场景：用户删了一个课件，它的向量必须同步清掉，
        否则检索还会返回"已经不存在的文件"的内容。

        Returns:
            删除前该文件的向量条数（Chroma 的 delete 不返回删除数量，
            所以先查后删）
        """
        collection = self.get_collection()
        where = {META_SOURCE_PATH: str(Path(source_path).resolve())}

        # get 只查不返回向量，加 limit 也行；这里只要数量
        existing = collection.get(where=where, include=[])
        n = len(existing.get("ids", []))

        if n:
            collection.delete(where=where)
            logger.info(f"删除 {n} 条向量（source={Path(source_path).name}）")

        return n

    def delete_collection(self) -> None:
        """删除整个 collection（全量重建前调用）。

        ⚠️ collection 不存在时，Chroma 会抛异常。
           这里吞掉它 —— "删一个不存在的东西"应当是幂等成功，而不是失败。
        """
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"删除 collection: {self.collection_name}")
        except Exception as exc:
            logger.debug(f"collection 不存在或已删除：{exc}")

    def reset(self) -> None:
        """删除并立刻重建一个空 collection。"""
        self.delete_collection()
        self.get_collection()

    # ============================================================
    # 调试辅助
    # ============================================================
    def peek(self, n: int = 5) -> list[Document]:
        """随便看几条，确认库里到底存了什么。

        排查"检索不到"的问题时，第一件事就是 peek 一下 ——
        很多时候会发现库根本是空的，或者存的是乱码。
        """
        collection = self.get_collection()
        raw = collection.peek(limit=n)

        docs: list[Document] = []
        for text, meta in zip(
            raw.get("documents") or [],
            raw.get("metadatas") or [],
            strict=False,
        ):
            docs.append(Document(page_content=text, metadata=dict(meta or {})))
        return docs

    def list_sources(self) -> list[str]:
        """列出库里所有不同的源文件路径。

        用途：校验"数据库里登记的文档"和"向量库里实际有的文档"是否一致。
        """
        collection = self.get_collection()
        # include=["metadatas"] 只取元数据，不取正文和向量，省内存
        raw = collection.get(include=["metadatas"])

        sources = {str(m.get(META_SOURCE_PATH, "")) for m in (raw.get("metadatas") or []) if m}
        return sorted(s for s in sources if s)
