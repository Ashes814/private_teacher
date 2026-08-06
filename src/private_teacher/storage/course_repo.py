"""
课程仓储（Repository）
======================

设计选择：**模块级函数** 而不是类。

为什么？
  - 这些函数没有共享状态（conn 每次都是参数传进来的）
  - 类只会多一层 self，没有任何收益
  - 需要替换实现时，Python 的模块本身就可以被 monkeypatch

命名冲突提醒：
  course_repo.create 和 document_repo.create 同名，
  导入时用别名区分：
      from private_teacher.storage.course_repo import create as create_course
"""

from __future__ import annotations

import sqlite3
import uuid

from loguru import logger

from private_teacher.storage.models import Course, now_utc, to_iso


def _new_id() -> str:
    """生成 8 位短 id。

    uuid4().hex 是 32 位十六进制，取前 8 位：
      - 碰撞概率：16^8 ≈ 43 亿种组合，个人项目几百门课，碰撞概率约等于 0
      - 短 id 可以直接当目录名（data/courseware/a1b2c3d4/），比 36 位 uuid 好看太多
    真要严谨，可以在 INSERT 失败（PRIMARY KEY 冲突）时重试一次。
    """
    return uuid.uuid4().hex[:8]


def create(conn: sqlite3.Connection, name: str, description: str = "") -> Course:
    """新建课程。

    Args:
        conn: 数据库连接
        name: 课程名（不做唯一性约束，允许同名——用 id 区分）
        description: 课程描述

    Returns:
        新建的 Course 对象（含生成的 id 和时间戳）
    """

    course_id = _new_id()
    now = now_utc()

    conn.execute(
        # ⚠️ 永远用 ? 占位符，永远不要用 f-string 拼 SQL
        #    f"INSERT ... VALUES ('{name}')" 会被 name = "'); DROP TABLE courses; --" 打穿
        """
        INSERT INTO courses (id, name, description, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (course_id, name, description, to_iso(now), to_iso(now)),
    )
    conn.commit()

    logger.info(f"创建课程: {name} (id={course_id}) ")

    # 直接用已知数据构造对象，省一次 SELECT
    return Course(
        id=course_id,
        name=name,
        description=description,
        created_at=now,
        updated_at=now,
    )


def get_by_id(conn: sqlite3.Connection, course_id: str) -> Course | None:
    """按 id 查课程。找不到返回 None（而不是抛异常）。

    为什么返回 None 而不抛异常？
      - "查不到"是正常业务分支（比如用户删了课但浏览器还开着旧页面）
      - 调用方用 `if course is None:` 处理，比 try/except 清晰
    """
    row = conn.execute(
        "SELECT * FROM courses WHERE id = ?",
        (course_id,),
    ).fetchone()

    return Course.from_row(row) if row else None


def list_all(conn: sqlite3.Connection) -> list[Course]:
    """列出所有课程，最近更新的排前面。"""
    rows = conn.execute("SELECT * FROM courses ORDER BY updated_at DESC").fetchall()

    return [Course.from_row(r) for r in rows]


def update(
    conn: sqlite3.Connection,
    course_id: str,
    name: str | None = None,
    description: str | None = None,
) -> Course | None:
    """更新课程。传 None 的字段保持不变。

    Returns:
        更新后的 Course；课程不存在时返回 None
    """
    course = get_by_id(conn, course_id)
    if course is None:
        return None

    # None 表示"不改这个字段"，所以用旧值兜底
    new_name = name if name is not None else course.name
    new_desc = description if description is not None else course.description
    now = now_utc()

    conn.execute(
        """
        UPDATE courses
           SET name = ?, description = ?, updated_at = ?
         WHERE id = ?
        """,
        (new_name, new_desc, to_iso(now), course_id),
    )
    conn.commit()

    return Course(
        id=course_id,
        name=new_name,
        description=new_desc,
        created_at=course.created_at,
        updated_at=now,
    )


def delete(conn: sqlite3.Connection, course_id: str) -> None:
    """删除课程。

    documents 表上声明了 ON DELETE CASCADE，
    所以这里删课程会**自动**删掉它名下所有文档记录。

    ⚠️ 前提：连接必须执行过 PRAGMA foreign_keys = ON
       （get_connection 里已经做了；如果你自己 sqlite3.connect，级联不会生效！）

    注意：这里只删数据库记录，**不删物理文件和向量库**。
    那是 CourseService.delete_course 的职责（见 §10）——
    Repository 层只管数据库，不碰文件系统，边界要清楚。
    """

    conn.execute("DELETE FROM courses WHERE id = ?", (course_id,))
    conn.commit()
    logger.info(f"删除课程：id={course_id}")


def touch(conn: sqlite3.Connection, course_id: str) -> None:
    """只更新 updated_at。

    用途：上传文档、重建知识库之后，把课程"顶"到列表最前面。
    """

    conn.execute(
        "UPDATE courses SET updated_at = ? WHERE id = ?",
        (to_iso(now_utc()), course_id),
    )
    conn.commit()
