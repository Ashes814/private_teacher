"""
Embedding 工厂
==============

对外接口：
    embedder = get_embedder(settings)
    vecs = embedder.embed_documents(["文本1", "文本2"])   # 批量（建索引用）
    vec  = embedder.embed_query("查询文本")                # 单条（检索用）

【为什么 embed_documents 和 embed_query 要分开？】
不是多此一举。有些模型（如 bge、e5）在训练时给查询加了前缀
（"query: xxx" / "为这个句子生成表示："），文档则不加。
用错接口会让检索准确率明显下降。所以哪怕两者实现相同，接口也要分开。

【四种 provider】
  openai      : 质量最好，要花钱，要联网
  ollama      : 本地跑，免费，需先 `ollama pull nomic-embed-text`
  huggingface : 本地跑，首次会下载模型（几百 MB），之后完全离线
  hash        : 我们自己写的确定性哈希 embedder —— **不是真的语义模型**，
                但零依赖、零延迟、结果可复现，专门给单元测试和离线 demo 用
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from itertools import pairwise
from typing import Protocol, runtime_checkable

from loguru import logger

from private_teacher.config import LLMSettings
from private_teacher.utils.exceptions import ConfigurationError, RAGError


# ============================================================
# 接口定义
# ============================================================
@runtime_checkable
class Embedder(Protocol):
    """Embedding 模型的最小借口。

    用 Protocol （结构化类型/鸭子类型）而不是抽象基类ABC:
     - langchain 的 OpenAIEmbeddings 不可能继承我们的基类
     - Protocol 只要求“长得像”就行，不要求继承关系
     - runtime_checkable 让 isinstance(x, Embedder) 也能用 （只检查方法名）
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量向量化文档。返回 len(texts) 个向量。"""
        ...

    def embed_query(self, text: str) -> list[float]:
        """向量化单条查询。返回 1 个向量。"""
        ...


# ============================================================
# 内置的确定性哈希 Embedder（测试 / 离线用）
# ============================================================
class HashEmbedder:
    """基于字符/词哈希的"伪 embedding"。

    ⚠️ 它**不理解语义**，只能捕捉字面重合度（本质是稀疏的 bag-of-words）。
    但它有三个测试场景下无可替代的优点：
      1. 完全确定性 —— 同样输入永远同样输出，测试不会 flaky
      2. 零依赖、零网络、微秒级 —— CI 里跑几千条也不心疼
      3. **有真实的区分能力** —— "机器学习"和"机器学习是AI分支"确实会更接近

    对比一下常见的错误写法：
        def embed_query(self, t): return [hash(t) % 100 / 100] * 1536
    这个写法有两个致命问题：
      a) 所有维度的值都相同 → 任意两个向量都平行 → 余弦相似度恒为 1，
         检索结果等于随机排序，测试断言"最相关的排第一"会随机失败
      b) 内置 hash() 对 str 加了随机盐，重启进程结果就变（PYTHONHASHSEED）

    【算法】
      1. 分词：中文按"单字 + 相邻二字组合"，英文按单词
      2. 每个 token 用 sha1 映射到 [0, dim) 的某个桶，桶内计数 +1
      3. L2 归一化，让余弦相似度 = 点积
    """

    # 匹配英文单词/数字（\w 在 Python 3 默认匹配 Unicode，所以要排除中文）
    _WORD_RE = re.compile(r"[a-zA-Z0-9_]+")

    def __init__(self, dimension: int = 256) -> None:
        self.dimension = dimension

    # ---------- 内部工具 ----------
    def _tokenize(self, text: str) -> list[str]:
        """把文本拆成 token 列表。

        中文没有空格分词，用"单字 + bigram（相邻两字）"是最简单有效的方案：
          "机器学习" → ["机","器","学","习","机器","器学","学习"]
        bigram 让"机器学习"和"深度学习"能通过共享的"学习"产生相似度。
        """

        lower = text.lower()
        tokens: list[str] = self._WORD_RE.findall(lower)

        # 提取所有中日韩统一表意文字
        cjk = [ch for ch in lower if "一" <= ch <= "鿿"]
        tokens.extend(cjk)
        # bigram: 把 [a, b, c] 配成 [(a, b), (b, c)]
        tokens.extend(a + b for a, b in pairwise(cjk))

        return tokens

    def _bucket(self, token: str) -> int:
        """把 token 稳定地映射到某个维度下标。

        用 sha1 而不是 hash()：确定性（见上面的说明）。
        取前 8 个十六进制字符 = 32 bit，够分散了。
        """

        digest = hashlib.sha1(token.encode("utf-8")).hexdigest()[:8]
        return int(digest, 16) % self.dimension

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dimension

        for token in self._tokenize(text):
            vec[self._bucket(token)] += 1.0

        # ---------- L2 归一化 ----------
        # 归一化后所有向量长度为 1，此时 余弦相似度 == 点积，
        # 且长文本不会仅仅因为"词多"就和所有查询都更相似
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0:
            # 全空文本：返回一个固定的非零向量，
            # 避免零向量导致余弦相似度出现 0/0 = NaN
            vec[0] = 1.0
            return vec

        return [v / norm for v in vec]

    # ---------- Embedder 协议实现 ----------
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


# ============================================================
# HuggingFace 本地模型包装
# ============================================================
class SentenceTransformerEmbedder:
    """sentence-transformers 的轻量包装。

    为什么自己包一层，不用 langchain_huggingface？
      - 少装一个包（langchain-huggingface 会连带装一堆东西）
      - 我们只需要 encode() 一个方法，包装成本极低

    推荐模型（中文场景）：
      BAAI/bge-small-zh-v1.5   512 维，约 100MB，速度快，质量够用
      BAAI/bge-base-zh-v1.5    768 维，约 400MB，质量更好
      BAAI/bge-m3              1024 维，约 2GB，多语言最强，但慢
    """

    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ConfigurationError(
                "使用 huggingface embedding 需要安装 sentence-transformers",
                hint="uv add sentence-transformers",
            ) from exc

        # 首次调用会从 HuggingFace 下载模型到 ~/.cache/huggingface
        # 国内网络可能很慢，可设置环境变量 HF_ENDPOINT=https://hf-mirror.com
        logger.info(f"加载本地 embedding 模型：{model_name} (首次会下载)")
        self._model = SentenceTransformer(model_name)

        # bge 系列要求给查询加指令前缀才能发挥最佳效果
        self._query_prefix = (
            "为这个句子生成表示以用于检索相关文章：" if "bge" in model_name.lower() else ""
        )

    def embed_documents(self, texts: list[str]) -> list[float]:
        # normalize_embeddings = True 让输出已经是单位向量，
        # 这样余弦相似度可以直接用点积算，也和 Chroma 的 cosine 距离匹配
        vectors = self._model.encode(texts, normalize_embeddings=True)
        # numpy array → 纯 Python list（Chroma 接受 list，也便于 JSON 序列化）
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        vector = self._model.encode(
            self._query_prefix + text,
            normalize_embeddings=True,
        )

        return list(vector.tolist())


# ============================================================
# 工厂函数
# ============================================================
def get_embedder(settings: LLMSettings) -> Embedder:
    """根据配置构造 Embedder。

    Args:
        settings: LLM 配置（读它的 embedding_* 字段）

    Returns:
        满足 Embedder 协议的对象

    Raises:
        ConfigurationError: 缺 API key / provider 非法 / 依赖没装
    """
    provider = settings.embedding_provider

    match provider:
        # ---------- OpenAI（含中转） ----------
        case "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                # 和 llm/factory.py 保持一致的错误风格：
                # 明确告诉用户缺什么、去哪儿配
                raise ConfigurationError(
                    "缺少 OPENAI_API_KEY（embedding 需要）",
                    provider="openai",
                    env_var="OPENAI_API_KEY",
                    hint="在 .env 设置 OPENAI_API_KEY=sk-...，"
                    "或改用 PRIVATE_TEACHER_LLM_EMBEDDING_PROVIDER=ollama",
                )

            from langchain_openai import OpenAIEmbeddings

            return OpenAIEmbeddings(
                model=settings.embedding_model,
                api_key=api_key,
                base_url=settings.embedding_base_url,
                chunk_size=settings.embedding_batch_size,
                # ⚠️ text-embedding-3-* 支持"降维"，但 dimensions 参数
                #    只对 3 系模型有效。老模型（ada-002）传这个参数会报错，
                #    所以这里做个判断
                **(
                    {"dimensions": settings.embedding_dimension}
                    if settings.embedding_model.startswith("text-embedding-3")
                    else {}
                ),
            )

        # ---------- Ollama 本地 ----------
        case "ollama":
            from langchain_ollama import OllamaEmbeddings

            return OllamaEmbeddings(
                model=settings.embedding_model,
                base_url=settings.embedding_base_url or settings.ollama_base_url,
            )

        # ---------- HuggingFace 本地 ----------
        case "huggingface":
            return SentenceTransformerEmbedder(settings.embedding_model)

        # ---------- 哈希（测试 / 离线） ----------
        case "hash":
            logger.warning("使用 HashEmbedder —— 它不理解语义，仅供测试/离线 demo 使用")

            return HashEmbedder(dimension=settings.embedding_dimension)

        case _:
            raise ConfigurationError(
                f"未知的 embedding provider: {provider!r}",
                valid_options=["openai", "ollama", "huggingface", "hash"],
            )


# ============================================================
# 带缓存的工厂（避免重复加载本地模型）
# ============================================================
# 模块级缓存：key 是配置的关键字段元组，value 是 Embedder 实例
# 为什么不用 @lru_cache 直接装饰 get_embedder？
#   → LLMSettings 是 pydantic 模型，默认不可哈希，不能当 lru_cache 的 key
_EMBEDDER_CACHE: dict[tuple[str, str, int], Embedder] = {}


def get_cached_embedder(settings: LLMSettings) -> Embedder:
    """带进程内缓存的 get_embedder。

    为什么需要缓存？
      - SentenceTransformer 加载一次要 3~10 秒，每次检索都重载完全不可接受
      - OpenAIEmbeddings 内部持有 HTTP 连接池，复用能省掉 TCP/TLS 握手

    ⚠️ 缓存 key 不包含 API key —— 换 key 需要重启进程（可接受的取舍）。
    """

    key = (
        settings.embedding_provider,
        settings.embedding_model,
        settings.embedding_dimension,
    )
    if key not in _EMBEDDER_CACHE:
        _EMBEDDER_CACHE[key] = get_embedder(settings)
    return _EMBEDDER_CACHE[key]


def clear_embedder_cache() -> None:
    """清空缓存。测试里换配置时要调用，否则会拿到上个测试的实例。"""
    _EMBEDDER_CACHE.clear()


# ============================================================
# 批量向量化（带进度与错误包装）
# ============================================================
def embed_in_batches(
    embedder: Embedder,
    texts: list[str],
    batch_size: int = 64,
    on_progress: "callable | None" = None,  # noqa: UP037
) -> list[list[float]]:
    """分批向量化，避免一次性把几千条文本塞给 API。

    Args:
        embedder: Embedder 实例
        texts: 待向量化文本
        batch_size: 每批条数
        on_progress: 可选回调 (已完成数, 总数)，Streamlit 进度条用

    Returns:
        与 texts 一一对应的向量列表

    Raises:
        RAGError: 任何一批失败都会中断并抛出（含批次信息，便于定位）
    """
    if not texts:
        return []

    vectors: list[list[float]] = []
    total = len(texts)

    for start in range(0, total, batch_size):
        batch = texts[start : start + batch_size]

        try:
            vectors.extend(embedder.embed_documents(batch))
        except Exception as exc:
            # 把底层异常统一包装成 RAGError，并带上"第几批"信息 ——
            # 排查时能立刻定位到是哪几条文本导致的（比如某条超长）
            raise RAGError(
                f"向量化失败（第 {start // batch_size + 1} 批）: {exc}",
                batch_start=start,
                batch_size=len(batch),
                total=total,
            ) from exc

        if on_progress is not None:
            on_progress(min(start + batch_size, total), total)

    # 防御性检查：数量对不上说明 embedder 实现有问题，
    # 这种 bug 如果不当场发现，会在后面表现为"向量和文本错位"，极难排查
    if len(vectors) != total:
        raise RAGError(
            f"向量数量不匹配：期望 {total}，实际 {len(vectors)}",
        )

    return vectors
