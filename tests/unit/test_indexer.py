"""indexer 单元测试（全部离线，用 HashEmbedder）。"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from private_teacher.config import PathSettings
from private_teacher.rag.indexer import Indexer
from private_teacher.rag.manifest import IndexManifest, ManifestEntry
from private_teacher.storage import course_repo, document_repo
from private_teacher.utils.hashing import compute_sha256

@pytest.fixture
def paths(tmp_path: Path) -> PathSettings:
    return PathSettings(data_dir=tmp_path / "data")


@pytest.fixture
def course(tmp_db: sqlite3.Connection):
    return course_repo.create(tmp_db, name="测试课程")

def add_file(
    conn: sqlite3.Connection,
    course_id: str,
    tmp_path: Path,
    name: str,
    content: str,
    category: str = "main",
):
    """在磁盘上造一个文件并登记到数据库。返回 Document 记录。"""
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")

    return document_repo.create(
        conn,
        course_id,
        path,
        category,
        compute_sha256(path),
        path.stat().st_size,
    )

@pytest.fixture
def indexer(course, hash_llm_settings, paths, fake_embedder) -> Indexer:
    return Indexer(
        course.id,
        hash_llm_settings,
        paths,
        embedder=fake_embedder,
    )  # 依赖注入，完全离线


# ============================================================
# Manifest
# ============================================================
class TestManifest:
    def test_load_missing_returns_empty(self, tmp_path: Path) -> None:
        m = IndexManifest.load("c1", tmp_path)
        assert m.entries == {}

    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        m = IndexManifest(course_id="c1")
        m.upsert(
            ManifestEntry(
                source_path="/a.txt",
                sha256="abc",
                chunk_count=3,
                indexed_at="2026-01-01T00:00:00+00:00",
            )
        )
        m.save(tmp_path)

        loaded = IndexManifest.load("c1", tmp_path)

        assert loaded.entries["/a.txt"].sha256 == "abc"
        assert loaded.total_chunks == 3

    def test_corrupted_file_returns_empty(self, tmp_path: Path) -> None:
        """损坏的 manifest 不该让程序崩溃，应降级为空清单。"""
        path = IndexManifest.path_for("c1", tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{ 这不是合法 JSON", encoding="utf-8")

        assert IndexManifest.load("c1", tmp_path).entries == {}

    def test_version_mismatch_returns_empty(self, tmp_path: Path) -> None:
        path = IndexManifest.path_for("c1", tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"version": 999, "entries": {}}', encoding="utf-8")

        assert IndexManifest.load("c1", tmp_path).entries == {}

    def test_needs_index_logic(self) -> None:
        m = IndexManifest(course_id="c1")

        assert m.needs_index("/a.txt", "sha1") is True  # 全新

        m.upsert(
            ManifestEntry("/a.txt", "sha1", 1, "2026-01-01T00:00:00+00:00")
        )
        assert m.needs_index("/a.txt", "sha1") is False  # 没变
        assert m.needs_index("/a.txt", "sha2") is True  # 变了

    def test_stale_sources(self) -> None:
        m = IndexManifest(course_id="c1")
        m.upsert(ManifestEntry("/a.txt", "s", 1, "t"))
        m.upsert(ManifestEntry("/b.txt", "s", 1, "t"))

        assert m.stale_sources({"/a.txt"}) == ["/b.txt"]

# ============================================================
# 全量构建
# ============================================================
class TestBuildFull:
    def test_creates_chunks(self, tmp_db, course, tmp_path, indexer) -> None:
        add_file(
            tmp_db,
            course.id,
            tmp_path,
            "doc.txt",
            "Python 是一种广泛使用的解释型编程语言，强调代码可读性。",
        )

        result = indexer.build_full(tmp_db)

        assert result.ok
        assert result.documents_indexed == 1
        assert result.chunks_added >= 1
        assert indexer.store.count() == result.chunks_added

    def test_marks_documents_indexed(self, tmp_db, course, tmp_path, indexer) -> None:
        doc = add_file(tmp_db, course.id, tmp_path, "a.txt", "一些课程内容文本")

        indexer.build_full(tmp_db)

        assert document_repo.get_by_id(tmp_db, doc.id).is_indexed is True
        assert document_repo.list_unindexed(tmp_db, course.id) == []

    def test_writes_manifest(self, tmp_db, course, tmp_path, indexer, paths) -> None:
        add_file(tmp_db, course.id, tmp_path, "a.txt", "课程内容文本内容")

        indexer.build_full(tmp_db)

        manifest = IndexManifest.load(course.id, paths.data_dir)
        assert len(manifest.entries) == 1
        assert manifest.total_chunks >= 1

    def test_rebuild_does_not_duplicate(
        self, tmp_db, course, tmp_path, indexer
    ) -> None:
        """⭐ 全量重建两次，向量数不应翻倍。"""
        add_file(tmp_db, course.id, tmp_path, "a.txt", "重建测试内容文本")

        indexer.build_full(tmp_db)
        n1 = indexer.store.count()
        indexer.build_full(tmp_db)
        n2 = indexer.store.count()

        assert n1 == n2

    def test_empty_course(self, tmp_db, course, indexer) -> None:
        """一个文件都没有时也要正常返回，不能崩。"""
        result = indexer.build_full(tmp_db)
        assert result.ok
        assert result.documents_indexed == 0


# ============================================================
# 增量构建
# ============================================================
class TestBuildIncremental:
    def test_only_processes_new(self, tmp_db, course, tmp_path, indexer) -> None:
        """⭐ 增量的核心：已索引的文件要被跳过。"""
        add_file(tmp_db, course.id, tmp_path, "a.txt", "第一个文件的内容")
        indexer.build_incremental(tmp_db)

        add_file(tmp_db, course.id, tmp_path, "b.txt", "第二个文件的内容")
        result = indexer.build_incremental(tmp_db)

        assert result.documents_indexed == 1  # 只处理了新的
        assert result.documents_skipped == 1  # 老的被跳过

    def test_second_run_skips_everything(
        self, tmp_db, course, tmp_path, indexer
    ) -> None:
        add_file(tmp_db, course.id, tmp_path, "a.txt", "内容没有任何变化")

        indexer.build_incremental(tmp_db)
        result = indexer.build_incremental(tmp_db)

        assert result.documents_indexed == 0
        assert result.documents_skipped == 1
        assert result.chunks_added == 0

    def test_modified_file_reindexed(self, tmp_db, course, tmp_path, indexer) -> None:
        """文件内容改了（sha256 变了）→ 必须重新索引。"""
        doc = add_file(tmp_db, course.id, tmp_path, "a.txt", "原始内容文本")
        indexer.build_incremental(tmp_db)

        # 模拟用户改了文件：改内容 + 更新数据库里的 sha256
        doc.path.write_text("修改后的全新内容文本", encoding="utf-8")
        tmp_db.execute(
            "UPDATE documents SET sha256 = ? WHERE id = ?",
            (compute_sha256(doc.path), doc.id),
        )
        tmp_db.commit()

        result = indexer.build_incremental(tmp_db)

        assert result.documents_indexed == 1
        assert result.documents_skipped == 0

    def test_deleted_file_cleaned_up(
        self, tmp_db, course, tmp_path, indexer
    ) -> None:
        """⭐ 数据库里删掉的文档，它的向量残留必须被清理。"""
        doc = add_file(tmp_db, course.id, tmp_path, "a.txt", "将要被删除的内容")
        add_file(tmp_db, course.id, tmp_path, "b.txt", "会保留下来的内容")
        indexer.build_incremental(tmp_db)
        assert indexer.store.count() >= 2

        document_repo.delete(tmp_db, doc.id)
        result = indexer.build_incremental(tmp_db)

        assert result.documents_removed == 1
        assert "/a.txt" not in " ".join(indexer.store.list_sources())

    def test_shrinking_file_removes_old_chunks(
        self, tmp_db, course, tmp_path, indexer
    ) -> None:
        """⚠️ 文件变短后，多余的旧 chunk 必须消失（不能留幽灵内容）。"""
        doc = add_file(
            tmp_db, course.id, tmp_path, "a.txt", "长内容段落。" * 500
        )
        indexer.build_incremental(tmp_db)
        n_before = indexer.store.count()
        assert n_before >= 3

        doc.path.write_text("变得很短的内容。", encoding="utf-8")
        tmp_db.execute(
            "UPDATE documents SET sha256 = ? WHERE id = ?",
            (compute_sha256(doc.path), doc.id),
        )
        tmp_db.commit()
        indexer.build_incremental(tmp_db)

        assert indexer.store.count() < n_before

    def test_missing_physical_file_reported(
        self, tmp_db, course, tmp_path, indexer
    ) -> None:
        """物理文件被手删 → 记进 errors，但不中断整批。"""
        doc = add_file(tmp_db, course.id, tmp_path, "a.txt", "会被删掉的文件")
        add_file(tmp_db, course.id, tmp_path, "b.txt", "正常存在的文件内容")
        doc.path.unlink()

        result = indexer.build_incremental(tmp_db)

        assert not result.ok
        assert len(result.errors) == 1
        assert "a.txt" in result.errors[0]
        assert result.documents_indexed == 1  # b.txt 仍然成功

    def test_unsupported_file_reported(
        self, tmp_db, course, tmp_path, indexer
    ) -> None:
        """不支持的格式记入 errors，不抛异常。"""
        add_file(tmp_db, course.id, tmp_path, "a.exe", "二进制内容")

        result = indexer.build_incremental(tmp_db)

        assert not result.ok
        assert result.documents_indexed == 0

    def test_progress_callback(self, tmp_db, course, tmp_path, indexer) -> None:
        add_file(tmp_db, course.id, tmp_path, "a.txt", "内容甲甲甲甲甲")
        add_file(tmp_db, course.id, tmp_path, "b.txt", "内容乙乙乙乙乙")

        seen: list[tuple[int, int, str]] = []
        indexer.build_incremental(
            tmp_db, on_file=lambda i, n, name: seen.append((i, n, name))
        )

        assert len(seen) == 2
        assert seen[0][1] == 2  # 总数正确

# ============================================================
# 元数据注入 & 统计
# ============================================================
class TestMetadataAndStats:
    def test_chunks_carry_course_metadata(
        self, tmp_db, course, tmp_path, indexer, fake_embedder
    ) -> None:
        """chunk 里必须带上 course_id / document_id / category。"""
        doc = add_file(
            tmp_db, course.id, tmp_path, "a.txt", "课程内容文本", "auxiliary"
        )
        indexer.build_full(tmp_db)

        hit = indexer.store.similarity_search(fake_embedder, "课程", k=1)[0]

        assert hit.metadata["course_id"] == course.id
        assert hit.metadata["document_id"] == doc.id
        assert hit.metadata["category"] == "auxiliary"

    def test_stats(self, tmp_db, course, tmp_path, indexer) -> None:
        add_file(tmp_db, course.id, tmp_path, "a.txt", "已索引的内容文本")
        indexer.build_full(tmp_db)
        add_file(tmp_db, course.id, tmp_path, "b.txt", "还没索引的内容")

        stats = indexer.stats(tmp_db)

        assert stats["documents_total"] == 2
        assert stats["documents_indexed"] == 1
        assert stats["documents_pending"] == 1
        assert stats["vectors_in_store"] >= 1

    def test_remove_document(self, tmp_db, course, tmp_path, indexer, paths) -> None:
        doc = add_file(tmp_db, course.id, tmp_path, "a.txt", "要被移除的内容文本")
        indexer.build_full(tmp_db)

        removed = indexer.remove_document(tmp_db, doc)

        assert removed >= 1
        assert indexer.store.count() == 0
        assert IndexManifest.load(course.id, paths.data_dir).entries == {}


class TestIndexResult:
    def test_summary_readable(self) -> None:
        from private_teacher.rag.indexer import IndexResult

        r = IndexResult(
            documents_indexed=3, chunks_added=42, documents_skipped=1,
            duration_seconds=1.234,
        )
        s = r.summary()

        assert "3 个文件" in s
        assert "42 个片段" in s
        assert r.ok is True