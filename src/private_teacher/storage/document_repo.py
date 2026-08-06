"""
文档仓储
========

与 course_repo 同构。多出来的两个关键能力：
  1. get_by_sha256  → 上传去重
  2. mark_indexed / list_unindexed → 增量索引的状态追踪
"""

from __future__ import annotations

import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger

from private_teacher.storage.models import Document, now_utc, to_iso

# 合法的分类值。放成常量，避免各处硬编码字符串打错字
VALID_CATEGORIES = ("main", "auxiliary")


def _new_id() -> str:
    return uuid.uuid4().hex[:8]


def create(
    conn: sqlite3.Connection,
    course_id: str,
    path: Path,
    category: str,
    sha256: str,
    size: int,
    mtime: datetime | None = None,
) -> Document:
    """登记一个文档。

    Args:
        conn: 连接
        course_id: 所属课程
        path: 文件路径（会被转成 str 存库）
        category: "main" 或 "auxiliary"
        sha256: 文件哈希（调用方用 compute_sha256 算好传进来）
        size: 字节数
        mtime: 文件修改时间；不传则尝试从文件读，读不到用当前时间

    Returns:
        新建的 Document

    Raises:
        ValueError: category 非法
        sqlite3.IntegrityError: (course_id, sha256) 重复 —— 即重复上传
    """
    # 在 Python 侧先校验一次，报错信息比 SQLite 的 CHECK constraint failed 友好得多
    # DB 层的 CHECK 约束是最后一道防线（防止有人绕过这个函数直接写 SQL）
    if category not in VALID_CATEGORIES:
        raise ValueError(f"category 必须是 {VALID_CATEGORIES} 之一，收到 {category!r}")

    if mtime is None:
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC) if path.exists() else now_utc()

    doc_id = _new_id()

    conn.execute(
        """
        INSERT INTO documents
            (id, course_id, path, category, sha256, size, mtime, indexed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, NULL)
        """,
        #                                          ⬆ 新文档一定还没建索引
        (doc_id, course_id, str(path), category, sha256, size, to_iso(mtime)),
    )
    conn.commit()

    logger.info(f"登记文档：{path.name}（course={course_id}, category={category}）")

    return Document(
        id=doc_id,
        course_id=course_id,
        path=path,
        category=category,
        sha256=sha256,
        size=size,
        mtime=mtime,
        indexed_at=None,
    )


def get_by_id(conn: sqlite3.Connection, document_id: str) -> Document | None:
    row = conn.execute(
        "SELECT * FROM documents WHERE id = ?",
        (document_id,),
    ).fetchone()
    return Document.from_row(row) if row else None


def get_by_sha256(
    conn: sqlite3.Connection,
    course_id: str,
    sha256: str,
) -> Document | None:
    """按内容哈希查文档 —— **这是去重的核心**。

    上传流程：
        sha = compute_sha256(f)
        if get_by_sha256(conn, course_id, sha):
            raise FileExistsError("这个文件已经传过了")

    注意查询条件带 course_id：不同课程可以各自持有同一份 PDF。
    """
    row = conn.execute(
        "SELECT * FROM documents WHERE course_id = ? AND sha256 = ?",
        (course_id, sha256),
    ).fetchone()
    return Document.from_row(row) if row else None


def list_by_course(
    conn: sqlite3.Connection,
    course_id: str,
    category: str | None = None,
) -> list[Document]:
    """列出某门课的文档。

    Args:
        category: 传 "main" / "auxiliary" 过滤；None = 全要

    实现技巧：动态拼 WHERE 子句时，
    **拼的是 SQL 结构（列名/关键字），值永远走参数**，这样依然安全。
    """

    sql = "SELECT * FROM documents WHERE course_id = ?"
    params: list[object] = [course_id]

    if category is not None:
        sql += " AND category = ?"
        params.append(category)

    sql += " ORDER BY category, path "

    rows = conn.execute(sql, params).fetchall()
    return [Document.from_row(r) for r in rows]


def list_unindexed(conn: sqlite3.Connection, course_id: str) -> list[Document]:
    """列出还没建索引的文档（indexed_at IS NULL）—— 增量索引用。

    ⚠️ SQL 里判空必须用 `IS NULL`，写成 `= NULL` 永远返回 false（三值逻辑）。
    """
    rows = conn.execute(
        """
        SELECT * FROM documents
         WHERE course_id = ? AND indexed_at IS NULL
         ORDER BY path
        """,
        (course_id,),
    ).fetchall()
    return [Document.from_row(r) for r in rows]


def get_indexed(conn: sqlite3.Connection, course_id: str) -> list[Document]:
    """列出已经建好索引的文档。UI 上显示"知识库覆盖了哪些文件"。"""
    rows = conn.execute(
        """
        SELECT * FROM documents
         WHERE course_id = ? AND indexed_at IS NOT NULL
         ORDER BY path
        """,
        (course_id,),
    ).fetchall()
    return [Document.from_row(r) for r in rows]


def mark_indexed(conn: sqlite3.Connection, document_id: str) -> None:
    """标记为"已建索引"，写入当前时间。"""
    conn.execute(
        "UPDATE documents SET indexed_at = ? WHERE id = ?",
        (to_iso(now_utc()), document_id),
    )
    conn.commit()


def mark_all_unindexed(conn: sqlite3.Connection, course_id: str) -> None:
    """把整门课的文档全部标记为"未索引"。

    用途：全量重建知识库时先调它，再逐个建索引 —— 保证状态和向量库一致。
    """
    conn.execute(
        "UPDATE documents SET indexed_at = NULL WHERE course_id = ?",
        (course_id,),
    )
    conn.commit()


def delete(conn: sqlite3.Connection, document_id: str) -> None:
    """删除文档记录（不删物理文件、不删向量）。"""
    conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))
    conn.commit()
    logger.info(f"删除文档记录：id={document_id}")
