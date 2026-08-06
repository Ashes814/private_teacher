"""
文件哈希工具
============

为什么需要哈希？
  - 上传去重：同一个文件改个名字再传一次，不应该重复入库
  - 增量索引：文件内容变了 → sha256 变了 → 需要重建这个文件的向量

为什么用 sha256 而不是 md5？
  - md5 已被证明可构造碰撞（虽然对本项目无所谓）
  - sha256 是标准库自带、速度足够（几百 MB/s），没有理由不用

为什么要分块读？
  - 一个 500MB 的 PDF 如果 read_bytes() 一次读进内存，会瞬间吃掉 500MB
  - 分块（每次 1MB）读，内存占用恒定
"""
from __future__ import annotations

import hashlib
from pathlib import Path

# 每次从磁盘读多少字节。1MB 是经验值：
# 太小 → 系统调用次数多，慢；太大 → 内存占用高，收益递减
_CHUNK_SIZE = 1024 * 1024


def compute_sha256(path: Path) -> str:
    """计算文件的 sha256 十六进制字符串。

    Args:
        path: 文件路径

    Returns:
        64 个字符的十六进制字符串，例如 "a1b2c3..."

    Raises:
        FileNotFoundError: 文件不存在（直接透传标准库异常，语义已经足够清楚）
    """
    hasher = hashlib.sha256()

    # "rb" = 二进制读。文本模式会做换行符转换，导致 Windows/Linux 哈希不一致
    with path.open("rb") as fp:
        # walrus 运算符 :=  ——  边赋值边判断，等价于：
        #   while True:
        #       chunk = fp.read(_CHUNK_SIZE)
        #       if not chunk: break
        #       hasher.update(chunk)
        while chunk := fp.read(_CHUNK_SIZE):
            hasher.update(chunk)

    return hasher.hexdigest()


def compute_sha256_bytes(data: bytes) -> str:
    """对内存中的字节串算 sha256。

    用途：Streamlit 的 file_uploader 给的是内存字节流，
    可以先算哈希判重，再决定要不要真的落盘（省一次磁盘写）。
    """
    return hashlib.sha256(data).hexdigest()
