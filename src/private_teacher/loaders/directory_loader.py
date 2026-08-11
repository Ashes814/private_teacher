"""
目录加载器
==========

递归扫描一个目录，把所有支持的文件都加载进来。

核心设计：**单个文件失败不能中断整批**。
用户上传一个 50 个文件的课件包，第 7 个是加密 PDF，
正确行为是"跳过它并告诉用户"，而不是整批失败。
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from loguru import logger

from private_teacher.loaders.registry import is_supported, load
from private_teacher.utils.exceptions import DocumentLoadError

# 扫描时直接跳过的目录名（不进去看）
# 这些目录里的文件对学习毫无价值，还可能有几万个小文件拖慢扫描
DEFAULT_IGNORE_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".svn",
        ".hg",
        "__pycache__",
        ".venv",
        "venv",
        "node_modules",
        ".idea",
        ".vscode",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "dist",
        "build",
        ".ipynb_checkpoints",
    }
)

# 单个文件大小上限，超过就跳过（默认 50MB）
# 防止误传一个 2GB 的数据集把内存吃爆
DEFAULT_MAX_FILE_SIZE = 50 * 1024 * 1024


def iter_loadable_files(
    dir_path=Path,
    ignore_dirs: Iterable[str] = DEFAULT_IGNORE_DIRS,
    max_file_size: int = DEFAULT_MAX_FILE_SIZE,
) -> list[Path]:
    """列出目录下所有可加载的文件（已过滤）。

    单独拆成一个函数的好处：
      - UI 上可以先"预览将要导入哪些文件"，让用户确认后再真的加载
      - 测试起来比 load_directory 容易得多（不用真的解析文件）
    """

    ignore = set(ignore_dirs)
    results: list[Path] = []

    # rglob("*") 递归遍历所有层级；返回顺序不保证，最后统一排序
    for item in dir_path.rglob("*"):
        if not item.is_file():
            continue

        # item.parts 是路径的每一段，例如 ("a", ".git", "config")
        # 只要任意一段命中忽略名单，整个文件跳过

        if ignore & set(item.parts):
            continue

        # 隐藏文件（.DS_Store、.env 之类）
        if item.name.startswith("."):
            continue

        if not is_supported(item):
            continue

        if item.stat().st_size > max_file_size:
            logger.warning(f"跳过大文件 {item.name}（{item.stat().st_size / 1024 / 1024:.1f}MB）")
            continue

        results.append(item)

    # 排序保证结果稳定 —— 否则每次扫描顺序不同，chunk_id 会跟着变
    return sorted(results)


def load_directory(
    dir_path: Path,
    ignore_dirs: Iterable[str] = DEFAULT_IGNORE_DIRS,
    max_file_size: int = DEFAULT_MAX_FILE_SIZE,
    **kwargs: Any,
) -> list[Document]:
    """递归加载整个目录。

    Args:
        dir_path: 目录路径
        ignore_dirs: 要跳过的目录名集合
        max_file_size: 单文件大小上限（字节）
        **kwargs: 透传给各 loader

    Returns:
        所有文件的 Document 合并列表

    Raises:
        DocumentLoadError: dir_path 不存在或不是目录
    """
    if not dir_path.exists():
        raise DocumentLoadError("目录不存在", path=str(dir_path))
    if not dir_path.is_dir():
        raise DocumentLoadError("不是目录", path=str(dir_path))

    files = iter_loadable_files(dir_path, ignore_dirs, max_file_size)
    all_docs: list[Document] = []
    failed: list[str] = []

    for file in files:
        try:
            all_docs.extend(load(file, **kwargs))
        except DocumentLoadError as exc:
            # 已知的加载失败 —— 记下来继续，不中断整批
            logger.warning(f"跳过 {file.name}: {exc}")
            failed.append(file.name)
        except Exception as exc:
            # 未预料的异常也不能让整批挂掉，但要用 exception 打完整堆栈，
            # 因为这说明我们的错误处理有漏洞，需要修
            logger.exception(f"加载 {file.name} 时发生未预期错误: {exc}")
            failed.append(file.name)

    logger.info(
        f"目录加载完成：{len(files)} 个文件 → {len(all_docs)} 个 Document"
        + (f"，{len(failed)} 个失败" if failed else "")
    )
    return all_docs
