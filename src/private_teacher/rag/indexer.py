"""
索引构建器
==========

一条完整的流水线：

    documents 表
        ↓  load()            加载 → Document（带页码/行号）
        ↓  split_documents() 切分 → chunk（带 chunk_id）
        ↓  embed + upsert    向量化并写入 Chroma
        ↓  mark_indexed()    回写数据库状态
        ↓  manifest.save()   记录哈希，供下次增量判断

两种模式：
  build_full        —— 清空重来。用于"我怀疑索引坏了"
  build_incremental —— 只处理新增/修改/删除的文件。日常用这个
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from sqlite3 import Connection
from typing import Any

from loguru import logger

from private_teacher.config import LLMSettings, PathSettings
from private_teacher.loaders import load
from private_teacher.rag.embeddings import Embedder, get_cached_embedder
from private_teacher.rag.manifest import IndexManifest, ManifestEntry
from private_teacher.rag.splitters import ChunkerConfig, split_documents
from private_teacher.rag.vector_store import VectorStoreManager
from private_teacher.storage import document_repo
from private_teacher.storage.models import Document as DocRecord, now_utc, to_iso
from private_teacher.utils.exceptions import DocumentLoadError


@dataclass(slots=True)
class IndexResult:
    """一次索引操作的结果报告。

    为什么要返回这么细的统计？
      - UI 上要显示"新增 3 个文件 / 42 个片段"
      - 出问题时，errors 列表能直接告诉用户是哪个文件失败了，
        而不是让他去翻日志
    """

    documents_indexed: int = 0  # 成功建索引的文件数
    documents_skipped: int = 0  # 没变化、跳过的文件数
    documents_removed: int = 0  # 清理掉的已删除文件数
    chunks_added: int = 0  # 新增的 chunk 数
    errors: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return not self.errors

    def summary(self) -> str:
        """给 UI 用的一句话摘要。"""
        parts = [
            f"索引 {self.documents_indexed} 个文件",
            f"新增 {self.chunks_added} 个片段",
        ]
        if self.documents_skipped:
            parts.append(f"跳过 {self.documents_skipped} 个未变更文件")
        if self.documents_removed:
            parts.append(f"清理 {self.documents_removed} 个已删除文件")
        if self.errors:
            parts.append(f"⚠️ {len(self.errors)} 个失败")
        return "，".join(parts) + f"（耗时 {self.duration_seconds:.1f}s）"

class Indexer:
    """某一门课的知识库构建器。"""

    def __init__(
        self,
        course_id: str,
        llm_settings: LLMSettings,
        path_settings: PathSettings,
        chunker_config: ChunkerConfig | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        """
        Args:
            course_id: 课程 id
            llm_settings: 用来构造 embedder
            path_settings: 数据目录
            chunker_config: 切分参数；不传用默认
            embedder: 直接注入 embedder（测试用）。
                传了就不再从 settings 构造 —— 这叫"依赖注入"，
                让测试可以塞一个假的进来，不需要 mock 全局函数。
        """
        self.course_id = course_id
        self.llm = llm_settings
        self.paths = path_settings
        self.chunker_config = chunker_config or ChunkerConfig()

        # get_cached_embedder：本地模型只加载一次
        self._embedder = embedder or get_cached_embedder(llm_settings)
        self.store = VectorStoreManager(course_id, path_settings.data_dir)

    # ============================================================
    # 单个文件的处理
    # ============================================================
    def _index_one(
        self,
        doc: DocRecord,
        on_progress: Any = None,
    ) -> int:
        """加载 → 切分 → 写向量库。返回写入的 chunk 数。

        Raises:
            DocumentLoadError: 文件读取/解析失败
            RAGError: 向量化或写入失败
        """
        # ---------- 加载 ----------
        raw_docs = load(doc.path)
        if not raw_docs:
            # 空文件 / 扫描版 PDF —— 不算错误，但要如实告知
            logger.warning(f"{doc.path.name} 没有可提取的文本，跳过")
            return 0

        # ---------- 切分 ----------
        chunks = split_documents(raw_docs, self.chunker_config)
        if not chunks:
            return 0

        # 给每个 chunk 补上课程/文档/分类信息。
        # 为什么在这里补而不是在 loader 里？
        #   → loader 不该知道"课程"这种业务概念，它只管解析文件（分层原则）
        for chunk in chunks:
            chunk.metadata["course_id"] = self.course_id
            chunk.metadata["document_id"] = doc.id
            chunk.metadata["category"] = doc.category

        # ---------- 写入 ----------
        # 先删这个文件的旧向量：文件改小了的话，
        # 旧的多余 chunk（比如原来有 10 块现在只有 6 块）不会被 upsert 覆盖到，
        # 不删就会变成"幽灵内容"永远留在库里
        self.store.delete_by_source(doc.path)

        ids = self.store.add_documents(
            self._embedder,
            chunks,
            batch_size=self.llm.embedding_batch_size,
            on_progress=on_progress,
        )
        return len(ids)

    # ============================================================
    # 增量构建
    # ============================================================
    def build_incremental(
        self,
        conn: Connection,
        on_file: Any = None,
    ) -> IndexResult:
        """只处理新增 / 修改 / 删除的文件。

        Args:
            conn: 数据库连接
            on_file: 可选回调 (当前序号, 总数, 文件名)，给进度条用

        Returns:
            IndexResult 统计报告
        """
        started = time.perf_counter()
        result = IndexResult()

        manifest = IndexManifest.load(self.course_id, self.paths.data_dir)
        docs = document_repo.list_by_course(conn, self.course_id)

        # ---------- 1. 清理已删除的文件 ----------
        current_paths = {str(d.path.resolve()) for d in docs}
        for stale in manifest.stale_sources(current_paths):
            self.store.delete_by_source(stale)
            manifest.remove(stale)
            result.documents_removed += 1
            logger.info(f"清理已删除文件的向量：{Path(stale).name}")

        # ---------- 2. 处理新增/修改 ----------
        total = len(docs)
        for idx, doc in enumerate(docs, start=1):
            source_key = str(doc.path.resolve())

            if on_file is not None:
                on_file(idx, total, doc.path.name)

            # 物理文件被用户手动删了
            if not doc.path.exists():
                result.errors.append(f"{doc.path.name}: 文件不存在")
                continue

            # 内容没变 → 跳过（增量的意义所在）
            if not manifest.needs_index(source_key, doc.sha256):
                result.documents_skipped += 1
                continue

            try:
                n_chunks = self._index_one(doc)
            except DocumentLoadError as exc:
                # 单个文件失败不中断整批 —— 用户可能上传了 50 个文件，
                # 不能因为第 7 个是加密 PDF 就让前 6 个的工作白费
                result.errors.append(f"{doc.path.name}: {exc}")
                logger.warning(f"索引失败 {doc.path.name}: {exc}")
                continue
            except Exception as exc:
                result.errors.append(f"{doc.path.name}: {exc}")
                logger.exception(f"索引 {doc.path.name} 时发生未预期错误")
                continue

            # ---------- 成功：更新三处状态 ----------
            document_repo.mark_indexed(conn, doc.id)  # ① 数据库
            manifest.upsert(  # ② 清单
                ManifestEntry(
                    source_path=source_key,
                    sha256=doc.sha256,
                    chunk_count=n_chunks,
                    indexed_at=to_iso(now_utc()) or "",
                    document_id=doc.id,
                )
            )
            result.documents_indexed += 1
            result.chunks_added += n_chunks

        # ③ 落盘
        manifest.save(self.paths.data_dir)

        result.duration_seconds = time.perf_counter() - started
        logger.info(f"增量索引完成：{result.summary()}")
        return result

    # ============================================================
    # 全量构建
    # ============================================================
    def build_full(
        self,
        conn: Connection,
        on_file: Any = None,
    ) -> IndexResult:
        """推倒重来：清空向量库和清单，然后全部重新索引。

        什么时候用？
          - 换了 embedding 模型（向量维度/语义空间都变了，旧向量全部作废）
          - 改了 chunk_size（chunk 边界变了，chunk_id 也会变）
          - 怀疑索引状态不一致

        实现上就是"先清空，再走一遍增量" —— 清空之后所有文件都变成"新增"，
        增量逻辑自然会处理全部文件。**不要为全量单独写一套流程**，
        两套代码迟早会出现行为不一致的 bug。
        """
        logger.info(f"开始全量重建知识库（course={self.course_id}）")

        # ① 删向量库（连 collection 一起删，确保 hnsw 配置也重置）
        self.store.reset()

        # ② 清空清单
        manifest = IndexManifest(course_id=self.course_id)
        manifest.save(self.paths.data_dir)

        # ③ 把数据库里的 indexed_at 全部清空
        document_repo.mark_all_unindexed(conn, self.course_id)

        # ④ 走增量流程（此时所有文件都是"新增"）
        return self.build_incremental(conn, on_file=on_file)

    # ============================================================
    # 单文件维护
    # ============================================================
    def remove_document(self, conn: Connection, document: DocRecord) -> int:
        """从知识库里移除一个文档的所有向量（用户删除课件时调用）。

        Returns:
            删除的向量条数
        """
        source_key = str(document.path.resolve())

        n = self.store.delete_by_source(source_key)

        manifest = IndexManifest.load(self.course_id, self.paths.data_dir)
        manifest.remove(source_key)
        manifest.save(self.paths.data_dir)

        return n

    # ============================================================
    # 状态查询
    # ============================================================
    def stats(self, conn: Connection) -> dict[str, Any]:
        """知识库现状。UI 顶部的指标卡片用。"""
        manifest = IndexManifest.load(self.course_id, self.paths.data_dir)
        docs = document_repo.list_by_course(conn, self.course_id)

        return {
            "documents_total": len(docs),
            "documents_indexed": len(manifest.entries),
            "documents_pending": sum(
                1
                for d in docs
                if manifest.needs_index(str(d.path.resolve()), d.sha256)
            ),
            "chunks_in_manifest": manifest.total_chunks,
            "vectors_in_store": self.store.count(),
        }
