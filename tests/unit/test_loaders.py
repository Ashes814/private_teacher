"""loaders 层单元测试。"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import pytest

from private_teacher.loaders import (
    SUPPORTED_EXTENSIONS,
    is_supported,
    iter_loadable_files,
    load,
    load_directory,
)
from private_teacher.loaders.base import sanitize_metadata
from private_teacher.utils.exceptions import DocumentLoadError

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "courseware"


# ============================================================
# base.py
# ============================================================
class TestBase:
    def test_sanitize_drops_none(self) -> None:
        """None 值必须被整个 key 丢掉（Chroma 不接受 None）。"""
        clean = sanitize_metadata({"a": 1, "b": None, "c": "x"})
        assert clean == {"a": 1, "c": "x"}
        assert "b" not in clean

    def test_sanitize_converts_path(self) -> None:
        clean = sanitize_metadata({"p": Path("/a/b")})
        assert isinstance(clean["p"], str)

    def test_sanitize_keeps_primitives(self) -> None:
        meta = {"s": "x", "i": 1, "f": 1.5, "b": True}
        assert sanitize_metadata(meta) == meta


# ============================================================
# text_loader
# ============================================================
class TestTextLoader:
    def test_loads_text_file(self, tmp_path: Path) -> None:
        f = tmp_path / "test.txt"
        f.write_text("hello world", encoding="utf-8")

        docs = load(f)

        assert len(docs) == 1
        assert "hello" in docs[0].page_content
        # resolve() 后是绝对路径，所以和 str(f.resolve()) 比
        assert docs[0].metadata["source_path"] == str(f.resolve())
        assert docs[0].metadata["source_name"] == "test.txt"
        assert docs[0].metadata["file_type"] == "txt"
        assert docs[0].metadata["page"] == 1

    def test_empty_file_returns_empty_list(self, tmp_path: Path) -> None:
        """空文件不应该产生 Document。"""
        f = tmp_path / "empty.txt"
        f.write_text("   \n\n  ", encoding="utf-8")
        assert load(f) == []

    def test_gbk_encoding_fallback(self, tmp_path: Path) -> None:
        """GBK 编码的中文文件也要能正确读出来。"""
        f = tmp_path / "gbk.txt"
        f.write_bytes("机器学习".encode("gb18030"))

        docs = load(f)

        assert len(docs) == 1
        assert "机器学习" in docs[0].page_content


# ============================================================
# markdown_loader
# ============================================================
class TestMarkdownLoader:
    def test_strips_frontmatter(self, tmp_path: Path) -> None:
        f = tmp_path / "a.md"
        f.write_text(
            "---\ntitle: 我的标题\n---\n\n# 正文\n内容",
            encoding="utf-8",
        )

        docs = load(f)

        assert "title:" not in docs[0].page_content  # frontmatter 被剥掉
        assert "# 正文" in docs[0].page_content
        assert docs[0].metadata["title"] == "我的标题"

    def test_no_frontmatter_still_works(self, tmp_path: Path) -> None:
        f = tmp_path / "b.md"
        f.write_text("# 直接正文", encoding="utf-8")

        docs = load(f)

        assert "# 直接正文" in docs[0].page_content
        assert docs[0].metadata["line_start"] == 1

    def test_frontmatter_shifts_line_numbers(self, tmp_path: Path) -> None:
        """剥掉 frontmatter 后行号要相应偏移，不能还从 1 开始。"""
        f = tmp_path / "c.md"
        f.write_text("---\ntitle: T\n---\n正文", encoding="utf-8")

        docs = load(f)

        assert docs[0].metadata["line_start"] > 1


# ============================================================
# code_loader
# ============================================================
class TestCodeLoader:
    def test_preserves_line_numbers(self, tmp_path: Path) -> None:
        f = tmp_path / "test.py"
        f.write_text(
            "\n".join(f"line {i}" for i in range(1, 251)),
            encoding="utf-8",
        )

        docs = load(f, lines_per_chunk=100)

        assert len(docs) == 3  # 100 + 100 + 50
        assert docs[0].metadata["line_start"] == 1
        assert docs[0].metadata["line_end"] == 100
        assert docs[1].metadata["line_start"] == 101
        assert docs[2].metadata["line_end"] == 250

    def test_language_detected(self, tmp_path: Path) -> None:
        f = tmp_path / "a.py"
        f.write_text("x = 1", encoding="utf-8")
        assert load(f)[0].metadata["language"] == "python"

    def test_blank_chunks_skipped(self, tmp_path: Path) -> None:
        """整块空白的 chunk 不入库。"""
        f = tmp_path / "a.py"
        # 前 5 行有内容，接着 200 行空白
        f.write_text("x = 1\n" * 5 + "\n" * 200, encoding="utf-8")

        docs = load(f, lines_per_chunk=100)

        assert len(docs) == 1  # 只有第一块有内容

    def test_invalid_chunk_size_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "a.py"
        f.write_text("x = 1", encoding="utf-8")
        with pytest.raises(ValueError):
            load(f, lines_per_chunk=0)


# ============================================================
# registry
# ============================================================
class TestRegistry:
    def test_unsupported_format_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "test.exe"
        f.write_text("binary", encoding="utf-8")
        with pytest.raises(DocumentLoadError, match="不支持"):
            load(f)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(DocumentLoadError, match="不存在"):
            load(tmp_path / "nope.txt")

    def test_directory_path_raises(self, tmp_path: Path) -> None:
        with pytest.raises(DocumentLoadError, match="不是文件"):
            load(tmp_path)

    def test_extension_case_insensitive(self, tmp_path: Path) -> None:
        """.TXT 和 .txt 都要能认。"""
        f = tmp_path / "A.TXT"
        f.write_text("hi", encoding="utf-8")
        assert len(load(f)) == 1

    def test_is_supported(self, tmp_path: Path) -> None:
        assert is_supported(Path("a.pdf"))
        assert is_supported(Path("a.py"))
        assert not is_supported(Path("a.exe"))

    def test_supported_extensions_not_empty(self) -> None:
        assert ".pdf" in SUPPORTED_EXTENSIONS
        assert ".md" in SUPPORTED_EXTENSIONS


# ============================================================
# directory_loader
# ============================================================
class TestDirectoryLoader:
    def test_directory_recursive(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("A", encoding="utf-8")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "b.txt").write_text("B", encoding="utf-8")

        docs = load_directory(tmp_path)

        assert len(docs) == 2

    def test_ignores_junk_dirs(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("A", encoding="utf-8")
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "x.py").write_text("x=1", encoding="utf-8")
        (tmp_path / ".git").mkdir()
        (tmp_path / ".git" / "y.txt").write_text("y", encoding="utf-8")

        assert len(load_directory(tmp_path)) == 1

    def test_skips_unsupported_silently(self, tmp_path: Path) -> None:
        """不支持的格式在目录扫描时应该被安静跳过，不抛异常。"""
        (tmp_path / "a.txt").write_text("A", encoding="utf-8")
        (tmp_path / "b.exe").write_text("bin", encoding="utf-8")

        assert len(load_directory(tmp_path)) == 1

    def test_skips_oversized(self, tmp_path: Path) -> None:
        (tmp_path / "big.txt").write_text("x" * 5000, encoding="utf-8")
        assert load_directory(tmp_path, max_file_size=1000) == []

    def test_iter_returns_sorted(self, tmp_path: Path) -> None:
        """扫描结果必须有序，否则 chunk_id 每次都不一样。"""
        for name in ("c.txt", "a.txt", "b.txt"):
            (tmp_path / name).write_text("x", encoding="utf-8")

        files = iter_loadable_files(tmp_path)

        assert [f.name for f in files] == ["a.txt", "b.txt", "c.txt"]

    def test_missing_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises(DocumentLoadError):
            load_directory(tmp_path / "nope")


# ============================================================
# fixtures 真实文件（没生成样本就 skip）
# ============================================================
@pytest.mark.skipif(
    not (FIXTURES / "sample.pdf").exists(),
    reason="先跑 uv run python scripts/make_sample_courseware.py",
)
class TestRealFixtures:
    def test_pdf_pages(self) -> None:
        docs = load(FIXTURES / "sample.pdf")
        assert len(docs) == 3
        assert docs[0].metadata["page"] == 1
        assert docs[0].metadata["total_pages"] == 3
        assert "Machine Learning" in docs[0].page_content

    def test_python_multi_chunk(self) -> None:
        docs = load(FIXTURES / "sample.py", lines_per_chunk=50)
        assert len(docs) >= 2
        # 相邻 chunk 的行号必须连续，不能有洞
        for prev, curr in pairwise(docs):
            assert curr.metadata["line_start"] == prev.metadata["line_end"] + 1
