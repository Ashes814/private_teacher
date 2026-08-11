"""课件加载层。

对外只暴露 4 个东西，其余都是实现细节：
    load                  加载单个文件
    load_directory        递归加载目录
    is_supported          判断格式是否支持
    SUPPORTED_EXTENSIONS  支持的扩展名元组（UI 用）
"""

from private_teacher.loaders.directory_loader import (
    iter_loadable_files,
    load_directory,
)
from private_teacher.loaders.registry import (
    LOADERS,
    SUPPORTED_EXTENSIONS,
    get_loader,
    is_supported,
    load,
)

__all__ = [
    "LOADERS",
    "SUPPORTED_EXTENSIONS",
    "get_loader",
    "is_supported",
    "iter_loadable_files",
    "load",
    "load_directory",
]
