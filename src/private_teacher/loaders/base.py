"""
Loader 公共契约
===============

这个文件不做任何加载工作，它只定义**所有 loader 必须遵守的约定**。

为什么值得单独开一个文件？
  - metadata 的 key 如果散落在 6 个 loader 里各写各的，
    早晚会出现 "source_path" / "source" / "src_path" 三种写法
  - 集中定义成常量后，改名字只改一处，IDE 还能自动补全

【metadata 标准字段】
  source_path : str  必填。文件绝对路径（str 形式，因为 Chroma 只接受基础类型）
  source_name : str  必填。文件名，UI 显示用
  file_type   : str  必填。扩展名（不含点），如 "pdf" / "py"
  page        : int  必填。页码/幻灯片号，从 1 开始；非分页文件恒为 1
  total_pages : int  必填。总页数；非分页文件为 1
  line_start  : int  选填。起始行号（从 1 开始），代码/文本类才有
  line_end    : int  选填。结束行号（含）

⚠️ 铁律：metadata 的值只能是 str / int / float / bool。
   不能是 None、list、dict、Path —— Chroma 会直接拒绝写入并抛
   ValueError: Expected metadata value to be a str, int, float or bool。
   所以"没有值"的字段要**整个 key 都不放**，而不是放 None。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# ---------- metadata key 常量 ----------
META_SOURCE_PATH = "source_path"
META_SOURCE_NAME = "source_name"
META_FILE_TYPE = "file_type"
META_PAGE = "page"
META_TOTAL_PAGES = "total_pages"
META_LINE_START = "line_start"
META_LINE_END = "line_end"
META_CHUNK_ID = "chunk_id" 


# Chroma 允许的 metadata值类型
_ALLOWED_META_TYPES = (str, int, float, bool)

def base_metadata(path: Path, page: int = 1, total_pages: int = 1) -> dict[str, Any]:
    """构造所有 loader 共用的基础 metadata。

    Args:
        path: 源文件路径
        page: 页码（从 1 开始）
        total_pages: 总页数

    Returns:
        含 5 个必填字段的 dict

    为什么用 resolve()？
      - 把相对路径变成绝对路径，避免"换个工作目录就找不到源文件"
      - 注意：resolve() 会解析软链接，一般来说这正是我们想要的
    """

    return {
        META_SOURCE_PATH: str(path.resolve()),
        META_SOURCE_NAME: path.name,
        # suffix 是 ".pdf"，lstrip(".") 去掉点；lower() 统一小写
        META_FILE_TYPE: path.suffix.lower().lstrip("."),
        META_PAGE: page,
        META_TOTAL_PAGES: total_pages,
    }

def sanitize_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    """清洗 metadata，保证能被 Chroma 接受。

    规则：
      - 值是 None      → 丢掉这个 key
      - 值是 Path      → 转 str
      - 值是 list/dict → 转 str（保底，不丢信息）
      - 其他不认识的   → 转 str

    这个函数在写入向量库之前调用（见 §6），是最后一道防线。
    """
    clean: dict[str, Any] = {}
    for key, value in meta.items():
        if value is None:
            continue # 关键：整个 key 都不要，而不是写 None
        if isinstance(value, _ALLOWED_META_TYPES):
            clean[key] = value
        else:
            clean[key] = str(value)

    return clean