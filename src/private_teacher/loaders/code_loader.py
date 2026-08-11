"""
代码加载器
==========

代码和散文不一样：**行号就是它的坐标系**。
用户看到检索结果时会问"这在哪个文件第几行"，所以行号必须精确到 1 行不差。

Phase 1 用最朴素的"每 N 行一块"策略。
它的缺点很明显 —— 会把一个函数拦腰切断。
Phase 2 会换成 tree-sitter 的 AST 切分（按函数/类边界切）。
先用简单方案跑通全链路，是刻意的工程取舍，不是偷懒。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from private_teacher.loaders.base import (
    META_LINE_END,
    META_LINE_START,
    base_metadata,
)
from private_teacher.loaders.text_loader import read_text_smart

# 扩展名 → 语言名。用于 metadata，UI 上可以做语法高亮
LANGUAGE_BY_EXT: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".jsx": "javascript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".scala": "scala",
    ".sh": "bash",
    ".sql": "sql",
    ".r": "r",
    ".m": "matlab",
    ".jl": "julia",
}

DEFAULT_LINES_PER_CHUNK = 100


def load_code(
    path: Path,
    lines_per_chunk: int = DEFAULT_LINES_PER_CHUNK,
    **_kwargs: Any,
) -> list[Document]:
    """按固定行数切分代码文件。

    Args:
        path: 代码文件
        lines_per_chunk: 每块多少行。100 行 ≈ 3000 字符 ≈ 750 token，
            对 embedding 模型来说是安全的长度

    Returns:
        每 lines_per_chunk 行一个 Document

    Example:
        250 行的文件、lines_per_chunk=100 → 3 个 Document：
            [1, 100] [101, 200] [201, 250]
    """

    if lines_per_chunk < 1:
        raise ValueError(f"lines_per_chunk 必须 >= 1, 收到 {lines_per_chunk}")

    text = read_text_smart(path)
    # splitlines() 会去掉行尾换行符，且能正确处理 \n / \r\n / \r 三种换行
    lines = text.splitlines()

    if not lines:
        return []

    language = LANGUAGE_BY_EXT.get(path.suffix.lower(), "text")
    docs: list[Document] = []

    # range(0, 250, 100) → 0, 100, 200
    for start_idx in range(0, len(lines), lines_per_chunk):
        # 切片自动处理越界：lines[200:300] 在只有 250 行时返回 50 行
        chunk_lines = lines[start_idx : start_idx + lines_per_chunk]

        # 整块都是空白行 → 跳过，别往向量库塞垃圾
        if not any(line.strip() for line in chunk_lines):
            continue

        meta = base_metadata(path)
        meta["language"] = language
        # ⚠️ 下标从 0 开始，行号从 1 开始，所以要 +1
        meta[META_LINE_START] = start_idx + 1
        # line_end 是「包含」的最后一行号
        meta[META_LINE_END] = min(start_idx + lines_per_chunk, len(lines))

        docs.append(
            Document(
                page_content="\n".join(chunk_lines),
                metadata=meta,
            )
        )

    return docs
