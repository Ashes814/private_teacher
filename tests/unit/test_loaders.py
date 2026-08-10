"""loaders 层单元测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

# from private_teacher.loaders import (
#     SUPPORTED_EXTENSIONS,
#     is_supported,
#     iter_loadable_files,
#     load,
#     load_directory,
# )
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