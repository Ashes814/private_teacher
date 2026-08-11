"""
Markdown 加载器
===============

比纯文本多做两件事：
  1. 剥掉 YAML frontmatter（--- 包裹的元信息），避免它污染语义检索
  2. 把 frontmatter 里的 title 提取到 metadata（UI 显示更友好）
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from private_teacher.loaders.base import (
    META_LINE_END,
    META_LINE_START,
    base_metadata,
)
from private_teacher.loaders.text_loader import read_text_smart

# 匹配文件开头的 YAML frontmatter：
#   \A        字符串开头（比 ^ 严格，^ 在 MULTILINE 下会匹配每行开头）
#   ---\s*\n  三个短横线 + 换行
#   (.*?)     非贪婪捕获中间内容
#   \n---\s*\n  结束的三个短横线
#   re.S      让 . 也能匹配换行符
_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.S)

# 从 frontmatter 里抠 title: xxx
_TITLE_RE = re.compile(r"^title:\s*(.+)$", re.M)


def load_markdown(path: Path, **_kwargs: Any) -> list[Document]:
    """加载 Markdown 文件。

    Returns:
        0 或 1 个 Document
    """
    raw = read_text_smart(path)

    title: str | None = None
    body = raw
    # frontmatter 占了多少行 —— 剥掉之后行号要相应偏移，否则跳转会错位
    offset = 0

    match = _FRONTMATTER_RE.match(raw)
    if match:
        frontmatter = match.group(1)
        # match.end() 是 frontmatter 结束位置（字符下标）
        # 数一下前面有多少个换行，就是偏移了多少行
        offset = raw[: match.end()].count("\n")
        body = raw[match.end() :]

        title_match = _TITLE_RE.search(frontmatter)
        if title_match:
            # strip 掉可能的引号：title: "机器学习" → 机器学习
            title = title_match.group(1).strip().strip("\"'")

    if not body.strip():
        return []

    meta = base_metadata(path)
    # +1 因为行号从 1 开始
    meta[META_LINE_START] = offset + 1
    meta[META_LINE_END] = offset + len(body.splitlines())
    if title:
        meta["title"] = title

    return [Document(page_content=body, metadata=meta)]
