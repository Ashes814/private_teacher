"""哈希工具测试。"""

from __future__ import annotations

from pathlib import Path

from private_teacher.utils.hashing import compute_sha256, compute_sha256_bytes


def test_same_content_same_hash(tmp_path: Path) -> None:
    """内容相同、文件名不同 → 哈希必须相同（这是去重能生效的前提）。"""
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("hello", encoding="utf-8")
    b.write_text("hello", encoding="utf-8")
    assert compute_sha256(a) == compute_sha256(b)


def test_different_content_different_hash(tmp_path: Path) -> None:
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("hello", encoding="utf-8")
    b.write_text("world", encoding="utf-8")
    assert compute_sha256(a) != compute_sha256(b)


def test_matches_known_value(tmp_path: Path) -> None:
    """对照 sha256 的标准值，确认实现没写错。

    空串的 sha256 是一个人人都能查到的常量，
    这种"黄金值测试"能保证你不是在测"自己的 bug 和自己的 bug 相等"。
    """
    f = tmp_path / "empty.txt"
    f.write_bytes(b"")
    expected = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    assert compute_sha256(f) == expected
    assert compute_sha256_bytes(b"") == expected


def test_large_file_chunked(tmp_path: Path) -> None:
    """超过一个 chunk（1MB）的文件也要算对。"""
    f = tmp_path / "big.bin"
    f.write_bytes(b"x" * (3 * 1024 * 1024))  # 3MB
    assert len(compute_sha256(f)) == 64