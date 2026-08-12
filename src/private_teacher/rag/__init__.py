"""RAG 核心层。

模块职责一览：
    splitters       文本切分
    embeddings      向量化
    vector_store    向量存储（Chroma）
    indexer         构建知识库（编排 loader → splitter → embedder → store）
    retriever       检索
    source_resolver 结果溯源
"""

from private_teacher.rag.splitters import (
    ChunkerConfig,
    split_documents,
    split_text,
)

__all__ = [
    "ChunkerConfig",
    "split_documents",
    "split_text",
]
