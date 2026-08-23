"""services 层单元测试（全部离线）。"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from private_teacher.config import PathSettings
from private_teacher.services import CourseService, KBService
from private_teacher.storage import course_repo, document_repo
from private_teacher.utils.exceptions import (
    DocumentLoadError,
    DuplicateDocumentError,
    ServiceError,
)

@pytest.fixture
def paths(tmp_path: Path) -> PathSettings:
    return PathSettings(data_dir=tmp_path / "data")

@pytest.fixture
def service(paths) -> CourseService:
    return CourseService(paths)


@pytest.fixture
def kb(hash_llm_settings, paths, fake_embedder) -> KBService:
    return KBService(hash_llm_settings, paths, embedder=fake_embedder)


@pytest.fixture
def src_file(tmp_path: Path) -> Path:
    """一个待上传的源文件。"""
    f = tmp_path / "src" / "lecture.txt"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(
        "机器学习是人工智能的核心分支，研究如何从数据中学习规律。",
        encoding="utf-8",
    )
    return f

# ============================================================
# 课程 CRUD
# ============================================================
class TestCourseCrud:
    def test_create_makes_dirs(self, tmp_db, service, paths) -> None:
        course = service.create_course(tmp_db, "Python 入门")

        assert (paths.course_dir(course.id) / "main").is_dir()
        assert (paths.course_dir(course.id) / "auxiliary").is_dir()

    def test_empty_name_rejected(self, tmp_db, service) -> None:
        with pytest.raises(ServiceError, match="不能为空"):
            service.create_course(tmp_db, "   ")

    def test_get_missing_raises(self, tmp_db, service) -> None:
        with pytest.raises(ServiceError, match="不存在"):
            service.get_course(tmp_db, "nope")

    def test_update(self, tmp_db, service) -> None:
        course = service.create_course(tmp_db, "旧名字")
        updated = service.update_course(tmp_db, course.id, name="新名字")
        assert updated.name == "新名字"


# ============================================================
# 上传
# ============================================================
class TestUpload:
    def test_copies_into_course_dir(self, tmp_db, service, src_file, paths) -> None:
        course = service.create_course(tmp_db, "Test")

        doc = service.upload_document(tmp_db, course.id, src_file, "main")

        assert doc.path.exists()
        assert doc.path.parent == paths.course_dir(course.id) / "main"
        assert src_file.exists()  # 是复制不是移动，源文件还在

    def test_dedup_by_sha256(self, tmp_db, service, src_file) -> None:
        """⭐ 同内容文件不能重复上传。"""
        course = service.create_course(tmp_db, "Test")
        service.upload_document(tmp_db, course.id, src_file, "main")

        with pytest.raises(DuplicateDocumentError):
            service.upload_document(tmp_db, course.id, src_file, "main")

    def test_duplicate_is_also_fileexistserror(
        self, tmp_db, service, src_file
    ) -> None:
        """多重继承生效：用内置异常也能捕获。"""
        course = service.create_course(tmp_db, "Test")
        service.upload_document(tmp_db, course.id, src_file, "main")

        with pytest.raises(FileExistsError):
            service.upload_document(tmp_db, course.id, src_file, "main")

    def test_same_name_different_content_renamed(
        self, tmp_db, service, tmp_path
    ) -> None:
        """⭐ 同名但内容不同 → 自动改名，不能互相覆盖。"""
        course = service.create_course(tmp_db, "Test")

        a = tmp_path / "d1" / "note.txt"
        b = tmp_path / "d2" / "note.txt"
        a.parent.mkdir(parents=True)
        b.parent.mkdir(parents=True)
        a.write_text("第一份内容", encoding="utf-8")
        b.write_text("第二份完全不同的内容", encoding="utf-8")

        doc_a = service.upload_document(tmp_db, course.id, a)
        doc_b = service.upload_document(tmp_db, course.id, b)

        assert doc_a.path != doc_b.path
        assert doc_a.path.exists() and doc_b.path.exists()
        assert doc_b.path.name == "note_1.txt"

    def test_unsupported_format_rejected(self, tmp_db, service, tmp_path) -> None:
        course = service.create_course(tmp_db, "Test")
        bad = tmp_path / "a.exe"
        bad.write_text("x", encoding="utf-8")

        with pytest.raises(DocumentLoadError, match="不支持"):
            service.upload_document(tmp_db, course.id, bad)

    def test_missing_source_rejected(self, tmp_db, service, tmp_path) -> None:
        course = service.create_course(tmp_db, "Test")
        with pytest.raises(DocumentLoadError):
            service.upload_document(tmp_db, course.id, tmp_path / "nope.txt")

    def test_invalid_category(self, tmp_db, service, src_file) -> None:
        course = service.create_course(tmp_db, "Test")
        with pytest.raises(ServiceError, match="category"):
            service.upload_document(tmp_db, course.id, src_file, "bad")

    def test_upload_bytes(self, tmp_db, service) -> None:
        course = service.create_course(tmp_db, "Test")

        doc = service.upload_bytes(
            tmp_db, course.id, "note.md", "# 标题\n内容".encode(), "auxiliary"
        )

        assert doc.path.exists()
        assert doc.category == "auxiliary"

    def test_upload_bytes_dedup(self, tmp_db, service) -> None:
        course = service.create_course(tmp_db, "Test")
        data = b"same content"
        service.upload_bytes(tmp_db, course.id, "a.txt", data)

        with pytest.raises(DuplicateDocumentError):
            service.upload_bytes(tmp_db, course.id, "b.txt", data)

    def test_import_directory(self, tmp_db, service, tmp_path) -> None:
        course = service.create_course(tmp_db, "Test")
        src = tmp_path / "batch"
        src.mkdir()
        (src / "a.txt").write_text("内容甲", encoding="utf-8")
        (src / "b.md").write_text("内容乙", encoding="utf-8")
        (src / "c.exe").write_text("忽略", encoding="utf-8")

        imported, skipped = service.import_directory(tmp_db, course.id, src)

        assert len(imported) == 2  # .exe 在扫描阶段就被过滤了
        assert skipped == []


# ============================================================
# 删除
# ============================================================
class TestDelete:
    def test_delete_document_removes_everything(
        self, tmp_db, service, kb, src_file
    ) -> None:
        course = service.create_course(tmp_db, "Test")
        doc = service.upload_document(tmp_db, course.id, src_file)
        kb.build_kb(tmp_db, course.id)
        assert kb.is_ready(course.id)

        service.delete_document(tmp_db, doc.id)

        assert not doc.path.exists()  # 文件没了
        assert document_repo.get_by_id(tmp_db, doc.id) is None  # 记录没了
        assert kb.search(course.id, "机器学习") == []  # 向量也没了

    def test_delete_course_removes_everything(
        self, tmp_db, service, kb, src_file, paths
    ) -> None:
        """⭐ 删课程后：文件、数据库、向量库三处都要干净。"""
        course = service.create_course(tmp_db, "Test")
        service.upload_document(tmp_db, course.id, src_file)
        kb.build_kb(tmp_db, course.id)

        service.delete_course(tmp_db, course.id)

        assert course_repo.get_by_id(tmp_db, course.id) is None
        assert document_repo.list_by_course(tmp_db, course.id) == []
        assert not paths.course_dir(course.id).exists()
        assert not (paths.data_dir / "knowledge_bases" / course.id).exists()

    def test_delete_missing_document(self, tmp_db, service) -> None:
        with pytest.raises(ServiceError, match="不存在"):
            service.delete_document(tmp_db, "nope")


# ============================================================
# 统计
# ============================================================
class TestStats:
    def test_counts(self, tmp_db, service, kb, tmp_path) -> None:
        course = service.create_course(tmp_db, "Test")
        service.upload_bytes(tmp_db, course.id, "a.txt", b"main content", "main")
        service.upload_bytes(
            tmp_db, course.id, "b.txt", b"aux content", "auxiliary"
        )

        stats = service.get_stats(tmp_db, course.id)

        assert stats.total_documents == 2
        assert stats.main_documents == 1
        assert stats.auxiliary_documents == 1
        assert stats.indexed_documents == 0
        assert stats.index_progress == 0.0

        kb.build_kb(tmp_db, course.id)
        assert service.get_stats(tmp_db, course.id).index_progress == 1.0


# ============================================================
# KBService
# ============================================================
class TestKBService:
    def test_build_and_search(self, tmp_db, service, kb, src_file) -> None:
        course = service.create_course(tmp_db, "Test")
        service.upload_document(tmp_db, course.id, src_file)

        result = kb.build_kb(tmp_db, course.id)

        assert result.ok
        assert result.chunks_added >= 1

        hits = kb.search(course.id, "机器学习", k=3)
        assert len(hits) >= 1
        assert "机器学习" in hits[0].content

    def test_search_with_sources(self, tmp_db, service, kb, src_file) -> None:
        """⭐ 检索结果必须能溯源到真实存在的文件。"""
        course = service.create_course(tmp_db, "Test")
        service.upload_document(tmp_db, course.id, src_file)
        kb.build_kb(tmp_db, course.id)

        hits = kb.search_with_sources(course.id, "机器学习", k=1)

        assert len(hits) == 1
        hit = hits[0]
        assert hit.location.exists is True
        assert hit.location.file_path.exists()
        assert hit.location.source_name == "lecture.txt"
        assert "lecture.txt" in hit.label

    def test_incremental_then_full(self, tmp_db, service, kb, src_file) -> None:
        course = service.create_course(tmp_db, "Test")
        service.upload_document(tmp_db, course.id, src_file)

        kb.build_kb(tmp_db, course.id, incremental=True)
        n1 = kb.stats(tmp_db, course.id)["vectors_in_store"]

        kb.build_kb(tmp_db, course.id, incremental=False)
        n2 = kb.stats(tmp_db, course.id)["vectors_in_store"]

        assert n1 == n2  # 全量重建不该改变向量数量

    def test_build_context(self, tmp_db, service, kb, src_file) -> None:
        course = service.create_course(tmp_db, "Test")
        service.upload_document(tmp_db, course.id, src_file)
        kb.build_kb(tmp_db, course.id)

        context, used = kb.build_context(course.id, "机器学习", k=2)

        assert "[1]" in context
        assert len(used) >= 1

    def test_clear_kb(self, tmp_db, service, kb, src_file) -> None:
        course = service.create_course(tmp_db, "Test")
        service.upload_document(tmp_db, course.id, src_file)
        kb.build_kb(tmp_db, course.id)

        kb.clear_kb(tmp_db, course.id)

        assert kb.is_ready(course.id) is False
        assert document_repo.list_unindexed(tmp_db, course.id) != []

    def test_search_empty_kb(self, kb) -> None:
        assert kb.search("nobody", "查询") == []
        assert kb.search_with_sources("nobody", "查询") == []