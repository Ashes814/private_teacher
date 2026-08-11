"""
纯文本加载器（.txt / .log / .csv 等）
=====================================

最简单的 loader，但有两个真实世界的坑必须处理：
  1. 编码：中文 Windows 上的 txt 大概率是 GBK，直接 utf-8 解码会炸
  2. 空文件：读出来是空串，不应该产生一个空 Document 污染向量库
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from loguru import logger

from private_teacher.loaders.base import (
    META_LINE_END,
    META_LINE_START,
    base_metadata,
)
from private_teacher.utils.exceptions import DocumentLoadError

# 尝试的编码顺序：先 utf-8（最通用），再中文 Windows 常见编码
# gb18030 是 gbk 的超集，能覆盖更多生僻字，所以放在 gbk 前面
_ENCODINGS = ("utf-8", "gb18030", "utf-16", "latin-1")


def read_text_smart(path: Path) -> str:
    """按多种编码依次尝试读取文本。

    为什么不直接 errors="replace"？
      - replace 会把无法解码的字节变成 "�"，中文文档会变成一片乱码
      - 先尝试正确的编码，全都失败了再用 replace 兜底，能救回绝大多数文件

    latin-1 永远不会失败（它把任意字节映射成字符），所以它是天然的兜底项。
    但它排在最后 —— 如果前面的编码能成功，就不会走到它。
    """
    for encoding in _ENCODINGS:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue  # 换下一种编码
        except OSError as exc:
            # 文件不存在 / 没权限 —— 换编码也没用，直接报错
            raise DocumentLoadError(
                f"读取文件失败: {exc}",
                path=str(path),
            ) from exc

    # 所有编码都失败（几乎不可能，因为 latin-1 不会失败）
    logger.warning(f"{path.name} 编码识别失败，改用 replace 模式")
    return path.read_text(encoding="utf-8", errors="replace")


def load_text(path: Path, **_kwargs: Any) -> list[Document]:
    """加载纯文本文件。

    Args:
        path: 文件路径
        **_kwargs: 吞掉调用方多传的参数（如 lines_per_chunk）。
            下划线开头表示"我知道有这个参数但故意不用"，ruff 不会报未使用告警。

    Returns:
        长度为 0 或 1 的列表 —— 空文件返回 []，不产生垃圾 Document

    为什么整个文件只产生 1 个 Document？
      - Document 是"逻辑单位"，切分是 splitter 的活（单一职责）
      - 纯文本没有天然的页边界，整体交给 splitter 按语义切更合理
    """
    text = read_text_smart(path)

    # strip() 后为空 → 说明是空文件或只有空白字符
    if not text.strip():
        logger.warning(f"跳过空文件：{path.name}")
        return []

    meta = base_metadata(path)
    # 文本文件天然有行号，全部记下来，方便 §9 跳转
    meta[META_LINE_START] = 1
    meta[META_LINE_END] = len(text.splitlines())

    return [Document(page_content=text, metadata=meta)]
