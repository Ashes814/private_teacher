"""
PDF 加载器
==========

策略：pypdf 优先，提取为空时自动降级到 pdfplumber。

为什么要两套？
  - pypdf 快（纯 Python，无外部依赖），但对复杂排版/特殊字体的提取质量一般
  - pdfplumber 基于 pdfminer.six，慢 3~10 倍，但版面还原更好
  - 扫描版 PDF（整页是图片）两个都提不出字 —— 那需要 OCR，Phase 1 不做，
    只在日志里明确告诉用户"这个文件没提取到文字"
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from loguru import logger

from private_teacher.loaders.base import base_metadata
from private_teacher.utils.exceptions import DocumentLoadError


def _extract_with_pypdf(path: Path) -> list[str]:
    """用 pypdf 提取每页文本。返回值下标 i 对应第 i+1 页。"""
    from pypdf import PdfReader

    reader = PdfReader(str(path))

    # 加密 PDF：先尝试空密码解密（很多 PDF 只是设了"禁止复制"，空密码能开）
    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception as exc:
            raise DocumentLoadError(
                "PDF 已加密且无法用空密码解开",
                path=str(path),
            ) from exc

    # extract_text() 可能返回 None（空页），统一成空串
    return [(page.extract_text() or "") for page in reader.pages]


def _extract_with_pdfplumber(path: Path) -> list[str]:
    """用 pdfplumber 提取（兜底方案，更慢但更准）。"""
    import pdfplumber

    texts: list[str] = []
    # with 保证文件句柄被释放（pdfplumber 会一直持有文件）
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            texts.append(page.extract_text() or "")
    return texts


def load_pdf(path: Path, **_kwargs: Any) -> list[Document]:
    """加载 PDF，**每页一个 Document**。

    为什么按页拆而不是整本一个 Document？
      - 检索结果要能告诉用户"在第 12 页"，页码必须在 metadata 里
      - 如果整本合成一个 Document，splitter 切完之后就分不清哪块在哪页了

    Returns:
        非空页的 Document 列表（空页会被跳过，但页码保持原样不重排）

    Raises:
        DocumentLoadError: 文件损坏、加密无法打开
    """
    try:
        page_texts = _extract_with_pypdf(path)
    except DocumentLoadError:
        raise  # 已经是我们的异常，原样抛出
    except Exception as exc:
        raise DocumentLoadError(
            f"pypdf 解析失败: {exc}",
            path=str(path),
        ) from exc

    # ---------- 降级判断 ----------
    # 所有页加起来都没几个字 → 大概率 pypdf 没提出来，换 pdfplumber 试试
    total_chars = sum(len(t.strip()) for t in page_texts)
    if total_chars < 20 and page_texts:
        logger.warning(f"{path.name}: pypdf 几乎没提取到文字，尝试 pdfplumber")
        try:
            page_texts = _extract_with_pdfplumber(path)
            total_chars = sum(len(t.strip()) for t in page_texts)
        except ImportError:
            logger.warning("未安装 pdfplumber，跳过降级（uv add pdfplumber）")
        except Exception as exc:
            logger.warning(f"pdfplumber 也失败了：{exc}")

    if total_chars == 0:
        # 不抛异常 —— 扫描版 PDF 是合法输入，只是我们暂时处理不了。
        # 返回空列表让上层"跳过这个文件"，比中断整批上传体验好得多
        logger.warning(f"{path.name}: 没有提取到任何文字，可能是扫描版 PDF（需要 OCR）")
        return []

    total = len(page_texts)
    docs: list[Document] = []

    # start=1：页码从 1 开始，和 PDF 阅读器显示的一致
    for page_no, text in enumerate(page_texts, start=1):
        if not text.strip():
            continue  # 空白页不入库
        docs.append(
            Document(
                page_content=text,
                metadata=base_metadata(path, page=page_no, total_pages=total),
            )
        )

    logger.debug(f"{path.name}: 提取 {len(docs)}/{total} 页")
    return docs
