"""
课程服务
========

管理"课程 + 课件文件"的完整生命周期。

【核心职责：保持三处状态一致】
    ① SQLite 记录（documents 表）
    ② 物理文件（data/courseware/{course_id}/{category}/）
    ③ 向量库（data/knowledge_bases/{course_id}/）

任何一处漏掉，就会出现"检索到了但打不开"或者"删了还能搜到"这类灵异现象。
"""

from __future__ import annotations

import shutil
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from private_teacher.config import PathSettings
from private_teacher.loaders import is_supported
from private_teacher.rag.vector_store import VectorStoreManager
from private_teacher.storage import course_repo, document_repo
from private_teacher.storage.models import Course, Document
from private_teacher.utils.exceptions import (
    DocumentLoadError,
    DuplicateDocumentError,
    ServiceError,
)
from private_teacher.utils.hashing import compute_sha256, compute_sha256_bytes


@dataclass(slots=True)
class CourseStats:
    """课程概览，UI 卡片用。"""

    course: Course
    total_documents: int
    main_documents: int
    auxiliary_documents: int
    indexed_documents: int
    total_size: int  # 字节

    @property
    def total_size_mb(self) -> float:
        return self.total_size / 1024 / 1024

    @property
    def index_progress(self) -> float:
        """已索引比例 0.0~1.0（进度条用）。"""
        if self.total_documents == 0:
            return 0.0
        return self.indexed_documents / self.total_documents


class CourseService:
    """课程与课件管理。"""

    def __init__(self, paths: PathSettings) -> None:
        self.paths = paths

    # ============================================================
    # 课程 CRUD
    # ============================================================
    def create_course(
        self,
        conn: sqlite3.Connection,
        name: str,
        description: str = "",
    ) -> Course:
        """创建课程，并建好物理目录。

        Raises:
            ServiceError: 课程名为空
        """

        name = name.strip()
        if not name:
            raise ServiceError("课程名不能为空")

        course = course_repo.create(conn, name, description)

        # 立刻建目录：晚建不如早建，
        # 否则上传时才发现建不了（权限/磁盘满）会让用户丢文件
        for category in ("main", "auxiliary"):
            (self.paths.course_dir(course.id) / category).mkdir(parents=True, exist_ok=True)

        logger.info(f"课程已创建：{name}（{course.id}）")
        return course

    def list_courses(self, conn: sqlite3.Connection) -> list[Course]:
        return course_repo.list_all(conn)

    def get_course(self, conn: sqlite3.Connection, course_id: str) -> Course:
        """取课程，不存在就抛异常（UI 上是"页面不该出现"的状态）。"""
        course = course_repo.get_by_id(conn, course_id)
        if course is None:
            raise ServiceError("课程不存在", course_id=course_id)
        return course

    def update_course(
        self,
        conn: sqlite3.Connection,
        course_id: str,
        name: str | None = None,
        description: str | None = None,
    ) -> Course:
        updated = course_repo.update(conn, course_id, name, description)
        if updated is None:
            raise ServiceError("课程不存在", course_id=course_id)
        return updated

    def delete_course(self, conn: sqlite3.Connection, course_id: str) -> None:
        """彻底删除课程：数据库 + 文件 + 向量库。

        ⚠️ 删除顺序很讲究：**先删外部资源，最后删数据库记录**。
           如果先删数据库，中途崩溃就再也找不到该删哪些文件了（孤儿数据）。
           反过来，先删文件后崩溃，数据库记录还在，用户可以重试删除。
           这个原则叫"先清理难恢复的引用，最后清理索引"。
        """

        # ① 向量库
        VectorStoreManager(course_id, self.paths.data_dir).delete_collection()

        # ② 知识库目录（chroma + manifest）
        kb_dir = self.paths.data_dir / "knowledge_bases" / course_id
        shutil.rmtree(kb_dir, ignore_errors=True)

        # ③ 课件文件
        shutil.rmtree(self.paths.course_dir(course_id), ignore_errors=True)

        # ④ 数据库（documents 靠 CASCADE 自动删）
        course_repo.delete(conn, course_id)

        logger.info(f"课程已彻底删除：{course_id}")

    # ============================================================
    # 文档上传
    # ============================================================
    def upload_document(
        self,
        conn: sqlite3.Connection,
        course_id: str,
        source_path: Path,
        category: str = "main",
    ) -> Document:
        """把一个已存在于磁盘的文件导入课程。

        流程：校验 → 算哈希 → 查重 → 复制 → 登记

        Args:
            source_path: 源文件（会被复制，不是移动）
            category: "main" 或 "auxiliary"

        Returns:
            登记好的 Document

        Raises:
            DocumentLoadError: 文件不存在 / 格式不支持
            DuplicateDocumentError: 内容重复（同时也是 FileExistsError）
            ServiceError: 课程不存在 / category 非法
        """
        # ---------- 1. 前置校验 ----------
        self.get_course(conn, course_id)  # 课程必须存在

        if category not in ("main", "auxiliary"):
            raise ServiceError(f"category 必须是 main/auxiliary，收到 {category!r}")

        if not source_path.exists() or not source_path.is_file():
            raise DocumentLoadError("文件不存在", path=str(source_path))

        if not is_supported(source_path):
            raise DocumentLoadError(
                f"不支持的格式: {source_path.suffix}",
                path=str(source_path),
            )

        # ---------- 2. 内容去重 ----------
        # 先算哈希再复制：重复文件根本不该占磁盘
        sha = compute_sha256(source_path)
        existing = document_repo.get_by_sha256(conn, course_id, sha)
        if existing is not None:
            raise DuplicateDocumentError(
                f"文件内容重复（已存在：{existing.path.name}）",
                sha256=sha[:12],
                existing_path=str(existing.path),
            )

        # ---------- 3. 复制到课件目录 ----------
        dest_dir = self.paths.course_dir(course_id) / category
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = self._unique_dest(dest_dir, source_path.name)

        # copy2 会连同修改时间/权限一起复制（copy 只复制内容）
        shutil.copy2(source_path, dest)

        # ---------- 4. 登记 ----------
        try:
            doc = document_repo.create(conn, course_id, dest, category, sha, dest.stat().st_size)
        except Exception:
            # 写库失败要把刚复制的文件删掉，否则留下无主的孤儿文件
            dest.unlink(missing_ok=True)
            raise

        course_repo.touch(conn, course_id)  # 把课程顶到列表最前
        logger.info(f"上传成功：{dest.name} → {course_id}/{category}")
        return doc

    def upload_bytes(
        self,
        conn: sqlite3.Connection,
        course_id: str,
        filename: str,
        data: bytes,
        category: str = "main",
    ) -> Document:
        """直接从内存字节流上传（Streamlit file_uploader 用）。

        比"先写临时文件再调 upload_document"好在哪？
          - 重复文件根本不落盘（先用字节算哈希查重）
          - 少一次磁盘往返
        """
        self.get_course(conn, course_id)

        if category not in ("main", "auxiliary"):
            raise ServiceError(f"category 必须是 main/auxiliary，收到 {category!r}")

        if not is_supported(Path(filename)):
            raise DocumentLoadError(f"不支持的格式: {Path(filename).suffix}")

        sha = compute_sha256_bytes(data)
        existing = document_repo.get_by_sha256(conn, course_id, sha)
        if existing is not None:
            raise DuplicateDocumentError(
                f"文件内容重复（已存在：{existing.path.name}）",
                sha256=sha[:12],
            )

        dest_dir = self.paths.course_dir(course_id) / category
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = self._unique_dest(dest_dir, filename)
        dest.write_bytes(data)

        try:
            doc = document_repo.create(conn, course_id, dest, category, sha, len(data))
        except Exception:
            dest.unlink(missing_ok=True)
            raise

        course_repo.touch(conn, course_id)
        return doc

    @staticmethod
    def _unique_dest(dest_dir: Path, filename: str) -> Path:
        """避免同名文件互相覆盖：a.pdf → a_1.pdf → a_2.pdf

        为什么会同名但不重复？
          内容不同、文件名相同（比如两个学期的"讲义.pdf"）。
          sha256 查重放行了，但直接复制会覆盖掉前一个 —— 必须改名。
        """
        dest = dest_dir / filename
        if not dest.exists():
            return dest

        stem, suffix = Path(filename).stem, Path(filename).suffix
        for i in range(1, 1000):
            candidate = dest_dir / f"{stem}_{i}{suffix}"
            if not candidate.exists():
                return candidate

        raise ServiceError(f"同名文件过多，无法生成唯一文件名: {filename}")

    # ============================================================
    # 文档删除与查询
    # ============================================================
    def delete_document(
        self,
        conn: sqlite3.Connection,
        document_id: str,
    ) -> None:
        """删除一个课件：向量 → 文件 → 数据库。"""
        doc = document_repo.get_by_id(conn, document_id)
        if doc is None:
            raise ServiceError("文档不存在", document_id=document_id)

        # ① 向量
        VectorStoreManager(doc.course_id, self.paths.data_dir).delete_by_source(doc.path)

        # ② 从 manifest 里移除（否则增量索引会以为它还在）
        from private_teacher.rag.manifest import IndexManifest

        manifest = IndexManifest.load(doc.course_id, self.paths.data_dir)
        manifest.remove(str(doc.path.resolve()))
        manifest.save(self.paths.data_dir)

        # ③ 物理文件
        doc.path.unlink(missing_ok=True)

        # ④ 数据库
        document_repo.delete(conn, document_id)

        logger.info(f"文档已删除：{doc.path.name}")

    def list_documents(
        self,
        conn: sqlite3.Connection,
        course_id: str,
        category: str | None = None,
    ) -> list[Document]:
        return document_repo.list_by_course(conn, course_id, category)

    def get_stats(self, conn: sqlite3.Connection, course_id: str) -> CourseStats:
        """课程概览统计。"""
        course = self.get_course(conn, course_id)
        docs = document_repo.list_by_course(conn, course_id)

        return CourseStats(
            course=course,
            total_documents=len(docs),
            main_documents=sum(1 for d in docs if d.category == "main"),
            auxiliary_documents=sum(1 for d in docs if d.category == "auxiliary"),
            indexed_documents=sum(1 for d in docs if d.is_indexed),
            total_size=sum(d.size for d in docs),
        )

    def import_directory(
        self,
        conn: sqlite3.Connection,
        course_id: str,
        dir_path: Path,
        category: str = "main",
    ) -> tuple[list[Document], list[str]]:
        """批量导入一个目录。

        Returns:
            (成功导入的文档列表, 跳过原因列表)

        为什么返回"跳过原因"而不是抛异常？
          批量操作里"部分失败"是常态。用户需要的是一份报告，
          而不是"第 3 个文件出错所以全部回滚"。
        """
        from private_teacher.loaders import iter_loadable_files

        imported: list[Document] = []
        skipped: list[str] = []

        for file in iter_loadable_files(dir_path):
            try:
                imported.append(self.upload_document(conn, course_id, file, category))
            except DuplicateDocumentError:
                skipped.append(f"{file.name}: 内容重复")
            except (DocumentLoadError, ServiceError) as exc:
                skipped.append(f"{file.name}: {exc}")

        logger.info(f"目录导入完成：成功 {len(imported)}，跳过 {len(skipped)}")
        return imported, skipped
