"""
Loader 注册表
=============

整个 loaders 包对外的**唯一入口**：

    from private_teacher.loaders import load
    docs = load(Path("xxx.pdf"))

调用方永远不需要知道 pdf 用什么库解析。
以后要加 .docx 支持，只需要写一个 load_docx 再往 LOADERS 里注册一行。
这就是"注册表模式"（registry pattern）——用数据（dict）代替 if/elif 分支。
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from private_teacher.loaders.code_loader import LANGUAGE_BY_EXT, load_code
from private_teacher.loaders.markdown_loader import load_markdown
from private_teacher.loaders.pdf_loader import load_pdf
from private_teacher.loaders.pptx_loader import load_pptx
from private_teacher.loaders.text_loader import load_text
from private_teacher.utils.exceptions import DocumentLoadError

# loader 的统一签名：(path, **kwargs) -> list[Document]
# 用 Callable[..., ...] 是因为各 loader 的关键字参数不同（lines_per_chunk 等）

LoaderFn = Callable[..., list[Document]]


def _build_registry() -> dict[str, LoaderFn]:
    """构造扩展名 → loader 的映射表。

    写成函数而不是直接写字面量，是为了能用循环把 20 多种代码扩展名
    一次性注册进去，避免手写 20 行重复代码（也就少 20 个打错字的机会）。
    """

    registry: dict[str, LoaderFn] = {
        # ---------- 文档 ----------
        ".pdf": load_pdf,
        ".pptx": load_pptx,
        # ---------- 文本 ----------
        ".txt": load_text,
        ".log": load_text,
        ".csv": load_text,
        ".tsv": load_text,
        # ---------- Markdown ----------
        ".md": load_markdown,
        ".markdown": load_markdown,
        # ---------- 结构化文本（当代码处理，保留行号） ----------
        ".json": load_code,
        ".yaml": load_code,
        ".yml": load_code,
        ".toml": load_code,
        ".ini": load_code,
        ".xml": load_code,
        ".html": load_code,
    }

    # 所有编程语言扩展名统一交给 load_code
    for ext in LANGUAGE_BY_EXT:
        registry[ext] = load_code

    return registry


LOADERS: dict[str, LoaderFn] = _build_registry()

# 对外暴露"我们支持哪些格式"，Streamlit 的 file_uploader 要用（type= 参数）
SUPPORTED_EXTENSIONS: tuple[str, ...] = tuple(sorted(LOADERS))


def is_supported(path: Path) -> bool:
    """这个文件能不能加载。用于目录扫描时过滤。"""
    return path.suffix.lower() in LOADERS


def get_loader(path: Path) -> LoaderFn:
    """按扩展名取 loader。

    Raises:
        DocumentLoadError: 扩展名不在注册表里
    """
    ext = path.suffix.lower()
    loader = LOADERS.get(ext)
    if loader is None:
        raise DocumentLoadError(
            f"不支持的文件格式: {ext or '(无扩展名)'}",
            path=str(path),
            supported=list(SUPPORTED_EXTENSIONS),
        )
    return loader


def load(path: Path, **kwargs: Any) -> list[Document]:
    """加载任意支持的文件 —— **整个 loaders 包的统一入口**。

    Args:
        path: 文件路径
        **kwargs: 透传给具体 loader，如 lines_per_chunk=50、include_notes=False。
            不认识的参数会被 loader 的 **_kwargs 安静吞掉，所以这里很宽松。

    Returns:
        Document 列表（可能为空 —— 空文件 / 扫描版 PDF）

    Raises:
        DocumentLoadError: 文件不存在 / 格式不支持 / 解析失败
    """
    # 先检查存在性，报错信息比让 open() 抛 FileNotFoundError 清楚
    if not path.exists():
        raise DocumentLoadError("文件不存在", path=str(path))
    if not path.is_file():
        raise DocumentLoadError("不是文件（是目录？）", path=str(path))

    loader = get_loader(path)
    return loader(path, **kwargs)
