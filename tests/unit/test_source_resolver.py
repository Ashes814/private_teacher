"""source_resolver 单元测试。"""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.documents import Document

from private_teacher.config import PathSettings
from private_teacher.rag.source_resolver import SourceResolver, _as_int


@pytest.fixture
def paths(tmp_path: Path) -> PathSettings:
    return PathSettings(data_dir=tmp_path / "data")

@pytest.fixture
def resolver(paths) -> SourceResolver:
    return SourceResolver("c1", paths)

def make_chunk(content: str = "内容", **meta) -> Document:
    base = {
        "source_path": "/nowhere/a.txt",
        "source_name": "a.txt",
        "file_type": "txt",
        "page": 1,
        "total_pages": 1,
    }
    base.update(meta)
    return Document(page_content=content, metadata=base)

# ============================================================
# 解析
# ============================================================
class TestResolve:
    def test_resolve_lines(self, tmp_path, resolver) -> None:
        f = tmp_path / "test.py"
        f.write_text("line1\nline2\nline3\nline4\nline5", encoding="utf-8")

        loc = resolver.resolve(
            make_chunk(
                "line3\nline4",
                source_path=str(f),
                source_name="test.py",
                file_type="py",
                line_start=3,
                line_end=4,
            )
        )

        assert loc.line_start == 3
        assert loc.line_end == 4
        assert loc.exists is True
        assert loc.has_lines is True

    def test_resolve_page(self, resolver) -> None:
        loc = resolver.resolve(
            make_chunk(file_type="pdf", page=12, total_pages=50)
        )
        assert loc.page == 12
        assert loc.has_pages is True
        assert loc.has_lines is False

    def test_missing_file_marked(self, resolver) -> None:
        """文件不存在时不能抛异常，要用 exists=False 标记。"""
        loc = resolver.resolve(make_chunk(source_path="/definitely/not/here.txt"))
        assert loc.exists is False
        assert "文件已移除" in loc.label()

    def test_snippet_truncated(self, resolver) -> None:
        loc = resolver.resolve(make_chunk("x" * 500), snippet_chars=100)
        assert loc.snippet.endswith("...")
        assert len(loc.snippet) <= 103

    def test_snippet_not_truncated_when_short(self, resolver) -> None:
        loc = resolver.resolve(make_chunk("短内容"))
        assert loc.snippet == "短内容"

    def test_string_numbers_converted(self, resolver) -> None:
        """metadata 里存成字符串的数字也要能正确解析。"""
        loc = resolver.resolve(make_chunk(page="7", line_start="3"))
        assert loc.page == 7
        assert loc.line_start == 3

    def test_recover_moved_file(self, tmp_path, paths, resolver) -> None:
        """⭐ 路径失效但文件还在课件目录里 → 应该按文件名找回。"""
        course_dir = paths.course_dir("c1") / "main"
        course_dir.mkdir(parents=True)
        real = course_dir / "moved.txt"
        real.write_text("内容还在", encoding="utf-8")

        loc = resolver.resolve(
            make_chunk(source_path="/old/path/moved.txt", source_name="moved.txt")
        )

        assert loc.exists is True
        assert loc.file_path == real

    def test_resolve_many(self, resolver) -> None:
        chunks = [make_chunk(), make_chunk(), make_chunk()]
        assert len(resolver.resolve_many(chunks)) == 3


# ============================================================
# 标签与链接
# ============================================================
class TestLabelAndUri:
    def test_label_with_lines(self, tmp_path, resolver) -> None:
        f = tmp_path / "a.py"
        f.write_text("x", encoding="utf-8")

        loc = resolver.resolve(
            make_chunk(
                source_path=str(f), source_name="a.py", line_start=10, line_end=25
            )
        )
        assert loc.label() == "a.py · 第 10-25 行"

    def test_label_single_line(self, tmp_path, resolver) -> None:
        f = tmp_path / "a.py"
        f.write_text("x", encoding="utf-8")

        loc = resolver.resolve(
            make_chunk(
                source_path=str(f), source_name="a.py", line_start=7, line_end=7
            )
        )
        assert loc.label() == "a.py · 第 7 行"

    def test_label_with_page(self, resolver) -> None:
        loc = resolver.resolve(
            make_chunk(source_name="ml.pdf", file_type="pdf", page=3, total_pages=20)
        )
        assert "第 3/20 页" in loc.label()

    def test_uri_pdf_has_page_anchor(self, tmp_path, resolver) -> None:
        """PDF 链接要带 #page= 锚点，浏览器才能直接跳页。"""
        f = tmp_path / "a.pdf"
        f.write_bytes(b"%PDF-1.4")

        loc = resolver.resolve(
            make_chunk(
                source_path=str(f),
                source_name="a.pdf",
                file_type="pdf",
                page=5,
                total_pages=10,
            )
        )
        assert loc.uri().endswith("#page=5")
        assert loc.uri().startswith("file://")

    def test_uri_empty_when_missing(self, resolver) -> None:
        loc = resolver.resolve(make_chunk(source_path="/gone.txt"))
        assert loc.uri() == ""

# ============================================================
# 读取源文件
# ============================================================
class TestReadSnippet:
    def test_reads_with_context(self, tmp_path, resolver) -> None:
        f = tmp_path / "test.py"
        f.write_text("line1\nline2\nline3\nline4\nline5", encoding="utf-8")

        loc = resolver.resolve(
            make_chunk(
                "line3\nline4",
                source_path=str(f),
                file_type="py",
                line_start=3,
                line_end=4,
            )
        )
        snippet = resolver.read_snippet(loc, context_lines=1)

        assert "line2" in snippet  # 上文
        assert "line3" in snippet
        assert "line5" in snippet  # 下文
        assert "line1" not in snippet  # 超出上下文范围

    def test_context_clamped_at_boundaries(self, tmp_path, resolver) -> None:
        """上下文超出文件首尾时不能报错（切片越界）。"""
        f = tmp_path / "a.txt"
        f.write_text("only line", encoding="utf-8")

        loc = resolver.resolve(
            make_chunk(source_path=str(f), line_start=1, line_end=1)
        )
        assert resolver.read_snippet(loc, context_lines=100) == "only line"

    def test_falls_back_for_pdf(self, resolver) -> None:
        """PDF 没有行号，直接返回 chunk 预览。"""
        loc = resolver.resolve(
            make_chunk("PDF 的内容", file_type="pdf", page=3, total_pages=10)
        )
        assert resolver.read_snippet(loc) == "PDF 的内容"

    def test_falls_back_when_missing(self, resolver) -> None:
        loc = resolver.resolve(
            make_chunk("原始内容", source_path="/gone.txt", line_start=1, line_end=2)
        )
        assert resolver.read_snippet(loc) == "原始内容"

    def test_numbered_snippet_marks_hits(self, tmp_path, resolver) -> None:
        f = tmp_path / "a.py"
        f.write_text("\n".join(f"line{i}" for i in range(1, 11)), encoding="utf-8")

        loc = resolver.resolve(
            make_chunk(source_path=str(f), line_start=5, line_end=6)
        )
        out = resolver.read_numbered_snippet(loc, context_lines=1)

        lines = out.splitlines()
        assert lines[0].startswith("  ")  # 第 4 行：上下文
        assert any(line.startswith("▶") for line in lines)
        assert "   5 |" in out


# ============================================================
# 分组 & 工具
# ============================================================
class TestGrouping:
    def test_group_by_file(self, resolver) -> None:
        chunks = [
            make_chunk(source_name="a.pdf", page=3),
            make_chunk(source_name="a.pdf", page=1),
            make_chunk(source_name="b.md", page=1),
        ]
        grouped = resolver.group_by_file(chunks)

        assert set(grouped) == {"a.pdf", "b.md"}
        assert len(grouped["a.pdf"]) == 2
        # 组内按页码排序
        assert [loc.page for loc in grouped["a.pdf"]] == [1, 3]


class TestAsInt:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (5, 5),
            ("5", 5),
            (None, None),
            ("", None),
            ("abc", None),
            (3.7, 3),
        ],
    )
    def test_conversion(self, value, expected) -> None:
        assert _as_int(value) == expected
