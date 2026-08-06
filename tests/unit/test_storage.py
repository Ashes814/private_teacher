"""storage 层单元测试。

测试组织原则：
  - 一个 class 测一个模块，class 名 = Test + 被测对象
  - 一个 test 方法只断言一件事（失败时一眼看出是哪里坏了）
  - 方法名写成句子：test_<做了什么>_<期望什么>
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

import pytest

from private_teacher.storage import course_repo, document_repo
from private_teacher.storage.db import Database, init_db
from private_teacher.storage.models import Course, Document, from_iso, to_iso


# ============================================================
# db.py
# ============================================================
class TestDatabase:
    def test_init_db_is_idempotent(self, tmp_path: Path) -> None:
        """重复 init_db 不应该报错（IF NOT EXISTS 生效）。"""
        db_path = tmp_path / "a.db"
        with Database(db_path) as conn:
            init_db(conn)
            init_db(conn)  # 第二次，不该炸
            # sqlite_master 是 SQLite 的系统表，存了所有表结构
            tables = {
                r["name"] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
        assert {"courses", "documents"} <= tables

    def test_foreign_keys_enabled(self, tmp_db: sqlite3.Connection) -> None:
        """外键必须是开启状态，否则 CASCADE 不生效。"""
        # PRAGMA 查询会返回一行一列，1 = 开启
        assert tmp_db.execute("PRAGMA foreign_keys").fetchone()[0] == 1

    def test_row_factory_allows_dict_access(self, tmp_db: sqlite3.Connection) -> None:
        """row_factory=Row 让结果可以按列名访问。"""
        course_repo.create(tmp_db, name="X")
        row = tmp_db.execute("SELECT * FROM courses").fetchone()
        assert row["name"] == "X"  # 不是 row[1]

    def test_rollback_on_exception(self, tmp_path: Path) -> None:
        """with 块里抛异常时应该回滚，数据不落库。"""
        db_path = tmp_path / "b.db"
        with Database(db_path) as conn:
            init_db(conn)

        with pytest.raises(RuntimeError), Database(db_path) as conn:
            conn.execute(
                "INSERT INTO courses VALUES ('x','n','',?,?)",
                (to_iso(datetime.now()), to_iso(datetime.now())),
            )
            raise RuntimeError("模拟业务异常")

        # 重新打开检查：那条数据不该存在
        with Database(db_path) as conn:
            assert conn.execute("SELECT COUNT(*) FROM courses").fetchone()[0] == 0


# ============================================================
# models.py
# ============================================================
class TestModels:
    def test_iso_roundtrip(self) -> None:
        """datetime → str → datetime 应该无损。"""
        from private_teacher.storage.models import now_utc

        dt = now_utc()
        assert from_iso(to_iso(dt)) == dt

    def test_iso_handles_none(self) -> None:
        assert to_iso(None) is None
        assert from_iso(None) is None
        assert from_iso("") is None

    def test_document_is_indexed_property(self, tmp_path: Path) -> None:
        doc = Document(
            id="d1",
            course_id="c1",
            path=tmp_path / "a.txt",
            category="main",
            sha256="abc",
            size=1,
            mtime=datetime.now(),
            indexed_at=None,
        )
        assert doc.is_indexed is False
        doc.indexed_at = datetime.now()
        assert doc.is_indexed is True


# ============================================================
# course_repo.py
# ============================================================
class TestCourseRepo:
    def test_create_and_get(self, tmp_db: sqlite3.Connection) -> None:
        course = course_repo.create(tmp_db, name="Python 入门")

        assert course.id  # 非空
        assert len(course.id) == 8
        assert course.name == "Python 入门"

        fetched = course_repo.get_by_id(tmp_db, course.id)
        assert fetched is not None
        assert fetched.name == "Python 入门"

    def test_get_missing_returns_none(self, tmp_db: sqlite3.Connection) -> None:
        assert course_repo.get_by_id(tmp_db, "nonexist") is None

    def test_list_all(self, tmp_db: sqlite3.Connection) -> None:
        course_repo.create(tmp_db, name="A")
        course_repo.create(tmp_db, name="B")
        assert len(course_repo.list_all(tmp_db)) == 2

    def test_update_partial(self, tmp_db: sqlite3.Connection) -> None:
        """只传 name 时 description 应保持不变。"""
        c = course_repo.create(tmp_db, name="A", description="原描述")
        updated = course_repo.update(tmp_db, c.id, name="B")

        assert updated is not None
        assert updated.name == "B"
        assert updated.description == "原描述"

    def test_update_missing_returns_none(self, tmp_db: sqlite3.Connection) -> None:
        assert course_repo.update(tmp_db, "nope", name="X") is None

    def test_delete_cascades_documents(self, tmp_db: sqlite3.Connection) -> None:
        """删课程时它名下的文档应该一起消失。"""
        course = course_repo.create(tmp_db, name="A")
        document_repo.create(tmp_db, course.id, Path("a.pdf"), "main", "abc123", 100)
        assert len(document_repo.list_by_course(tmp_db, course.id)) == 1

        course_repo.delete(tmp_db, course.id)

        assert course_repo.get_by_id(tmp_db, course.id) is None
        assert document_repo.list_by_course(tmp_db, course.id) == []


# ============================================================
# document_repo.py
# ============================================================
class TestDocumentRepo:
    def test_create_returns_unindexed(self, sample_course, tmp_db) -> None:
        doc = document_repo.create(tmp_db, sample_course.id, Path("a.pdf"), "main", "h1", 100)
        assert doc.indexed_at is None
        assert doc.is_indexed is False

    def test_invalid_category_raises(self, sample_course, tmp_db) -> None:
        with pytest.raises(ValueError, match="category"):
            document_repo.create(tmp_db, sample_course.id, Path("a.pdf"), "bad", "h1", 100)

    def test_sha256_dedup(self, sample_course, tmp_db) -> None:
        """同一门课里，相同 sha256 不允许插两次。"""
        document_repo.create(tmp_db, sample_course.id, Path("a.pdf"), "main", "same", 100)
        with pytest.raises(sqlite3.IntegrityError):
            document_repo.create(tmp_db, sample_course.id, Path("b.pdf"), "main", "same", 100)

    def test_same_sha_allowed_across_courses(self, tmp_db) -> None:
        """不同课程可以各自持有同一份文件。"""
        c1 = course_repo.create(tmp_db, name="A")
        c2 = course_repo.create(tmp_db, name="B")
        document_repo.create(tmp_db, c1.id, Path("a.pdf"), "main", "same", 1)
        document_repo.create(tmp_db, c2.id, Path("a.pdf"), "main", "same", 1)
        # 没抛异常就算过

    def test_get_by_sha256(self, sample_course, tmp_db) -> None:
        document_repo.create(tmp_db, sample_course.id, Path("a.pdf"), "main", "hhh", 1)
        assert document_repo.get_by_sha256(tmp_db, sample_course.id, "hhh") is not None
        assert document_repo.get_by_sha256(tmp_db, sample_course.id, "xxx") is None

    def test_list_by_category(self, sample_course, tmp_db) -> None:
        document_repo.create(tmp_db, sample_course.id, Path("a"), "main", "h1", 1)
        document_repo.create(tmp_db, sample_course.id, Path("b"), "auxiliary", "h2", 1)

        assert len(document_repo.list_by_course(tmp_db, sample_course.id)) == 2
        mains = document_repo.list_by_course(tmp_db, sample_course.id, "main")
        assert len(mains) == 1
        assert mains[0].category == "main"

    def test_mark_indexed_flow(self, sample_course, tmp_db) -> None:
        """未索引 → mark → 已索引 → mark_all_unindexed → 又回到未索引。"""
        doc = document_repo.create(tmp_db, sample_course.id, Path("a"), "main", "h1", 1)
        assert len(document_repo.list_unindexed(tmp_db, sample_course.id)) == 1

        document_repo.mark_indexed(tmp_db, doc.id)
        assert document_repo.list_unindexed(tmp_db, sample_course.id) == []
        assert len(document_repo.get_indexed(tmp_db, sample_course.id)) == 1

        document_repo.mark_all_unindexed(tmp_db, sample_course.id)
        assert len(document_repo.list_unindexed(tmp_db, sample_course.id)) == 1

    def test_path_roundtrip(self, sample_course, tmp_db) -> None:
        """路径存进去是 str，取出来必须还是 Path。"""
        p = Path("data") / "courseware" / "x.pdf"
        doc = document_repo.create(tmp_db, sample_course.id, p, "main", "h1", 1)
        fetched = document_repo.get_by_id(tmp_db, doc.id)
        assert isinstance(fetched.path, Path)
        assert fetched.path == p
