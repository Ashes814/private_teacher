"""
文本切分器
==========

【为什么要切】
Embedding 模型有输入长度上限（512 ~ 8192 token）。更重要的是：
一个 5000 字的 chunk 向量化后，语义被"平均"掉了，检索时什么都匹配不上。
chunk 太小又会丢上下文。所以切分是 RAG 效果的第一个关键旋钮。

【递归切分算法】
给一组按"语义强度"从强到弱排序的分隔符：
    ["\\n\\n", "\\n", "。", "！", "？", "；", " ", ""]
                段落   行   中文句子结束      词  字符

1. 用最强的分隔符切一刀
2. 切出来的片段如果还超长，就用次强的分隔符继续切它
3. 一直递归到 "" —— 即按字符硬切（保底，保证一定能切到足够小）
4. 最后把这些小片段**贪心地合并**回接近 chunk_size 的块，并加上 overlap

这样做的效果：优先保持段落完整，实在不行才拆句子，最后才拆字。

【chunk_size 怎么选】
  - 中文：1 汉字 ≈ 1 字符 ≈ 0.7 token，800 字符 ≈ 560 token
  - 英文：1 token ≈ 4 字符，800 字符 ≈ 200 token
  - 800 对两种语言都在安全区，且检索粒度合适（约等于一个自然段）

【overlap 怎么选】
  chunk_size 的 10%~15%。作用是防止关键句正好落在两块的接缝上。
  例："...模型的准确率达到了" | "95%，远超基线。"
  没有 overlap 的话，问"准确率多少"两块都答不上来。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from langchain_core.documents import Document

from private_teacher.loaders.base import META_CHUNK_ID, META_PAGE, META_SOURCE_PATH

# ============================================================
# 默认分隔符（顺序 = 语义强度，从强到弱）
# ============================================================
DEFAULT_SEPARATORS: tuple[str, ...] = (
    "\n\n",  # 段落：最强的语义边界
    "\n",  # 换行
    "。",  # 中文句号
    "！",  # 中文感叹号
    "？",  # 中文问号
    "；",  # 中文分号
    ". ",  # 英文句号（带空格，避免切断 "3.14" 和 "Mr. Smith"）
    "! ",
    "? ",
    "; ",
    "，",  # 中文逗号（比句子弱，但比空格强）
    " ",  # 空格（英文词边界）
    "",  # 空串 = 按字符硬切，最后的保底
)


@dataclass(frozen=True, slots=True)
class ChunkerConfig:
    """切分参数。

    frozen=True 让它不可变 —— 配置对象被意外修改是很难查的 bug。
    不可变还带来一个好处：可以当 dict 的 key、可以安全地当默认参数。
    """

    chunk_size: int = 800
    chunk_overlap: int = 100
    separators: tuple[str, ...] = DEFAULT_SEPARATORS
    # 小于这个长度的 chunk 直接丢弃（通常是切分产生的碎屑，没有检索价值）
    # 默认 5：低于 5 字符的片段几乎没检索意义（中文 2-3 字就够）
    min_chunk_size: int = 5

    def __post_init__(self) -> None:
        """dataclass 的构造后钩子，用来做参数校验。

        frozen 的 dataclass 里不能直接赋值，但可以读、可以抛异常。
        """
        if self.chunk_size < 1:
            raise ValueError(f"chunk_size 必须 > 0，收到 {self.chunk_size}")
        if self.chunk_overlap < 0:
            raise ValueError(f"chunk_overlap 不能为负，收到 {self.chunk_overlap}")
        if self.chunk_overlap >= self.chunk_size:
            # overlap >= chunk_size 会导致合并循环无法前进（死循环）
            raise ValueError(
                f"chunk_overlap({self.chunk_overlap}) 必须小于 chunk_size({self.chunk_size})"
            )


# 模块级默认配置。因为 frozen，共享同一个实例是安全的
DEFAULT_CONFIG = ChunkerConfig()


# ============================================================
# 第一步：把长文本切成"原子"（每个都 <= chunk_size）
# ============================================================
def _split_keep_separator(text: str, separator: str) -> list[str]:
    """用 separator 切分，但把分隔符**保留在前一段的末尾**。

    为什么要保留？
      "今天下雨。明天晴。" 用 "。" 切：
        丢掉分隔符 → ["今天下雨", "明天晴"]     ← 句号没了，读起来别扭
        保留分隔符 → ["今天下雨。", "明天晴。"]  ← 原文可还原
      保留分隔符还有个隐性好处：拼回去等于原文，方便做溯源定位。
    """

    parts = text.split(separator)
    result: list[str] = []

    for i, part in enumerate(parts):
        # 最后一段后面没有分隔符（split 的性质）
        if i < len(parts) - 1:
            result.append(part + separator)
        else:
            result.append(part)

    # 过滤掉纯空串（连续分隔符会产生空串）
    return [p for p in result if p]


def _atomize(text: str, separators: tuple[str, ...], chunk_size: int) -> list[str]:
    """递归切分，直到每个片段都不超过 chunk_size。

    Args:
        text: 待切文本
        separators: 剩余可用的分隔符（越靠前语义越强）
        chunk_size: 单块上限

    Returns:
        每个元素长度都 <= chunk_size 的片段列表

    递归的三个出口：
      1. 文本本身就够短 → 直接返回
      2. 分隔符用完了   → 按字符硬切
      3. 当前分隔符是 "" → 按字符硬切
    """
    # 出口 1：已经够短
    if len(text) <= chunk_size:
        return [text] if text else []

    # 出口 2：没有分隔符可用了，只能硬切
    if not separators:
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    separator, rest = separators[0], separators[1:]

    # 出口 3：空串分隔符 = 约定的"硬切"标记
    if separator == "":
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    # 当前分隔符在文本里根本不存在 → 跳过它，用下一个
    if separator not in text:
        return _atomize(text, rest, chunk_size)

    pieces = _split_keep_separator(text, separator)
    atoms: list[str] = []

    for piece in pieces:
        if len(piece) <= chunk_size:
            atoms.append(piece)
        else:
            # 这一片还是太长 → 用更弱的分隔符继续切它
            atoms.extend(_atomize(piece, rest, chunk_size))

    return atoms


# ============================================================
# 第二步：把原子贪心合并成接近 chunk_size 的块（并加 overlap）
# ============================================================
def _merge_atoms(atoms: list[str], chunk_size: int, overlap: int) -> list[str]:
    """把小片段合并成大块。

    算法（贪心 + 滑动窗口）：
      维护一个"当前缓冲区" buffer 和它的总长度 total。
      每来一个原子：
        - 如果加进去会超长，就先把 buffer 输出成一个 chunk，
          然后从 buffer **头部**弹出元素，直到剩余长度 <= overlap，
          这些残留的尾巴就成了下一块的开头 —— 这就是 overlap 的实现。
        - 然后把新原子放进 buffer。

    Example:
        atoms = ["A"*500, "B"*400, "C"*300], chunk_size=800, overlap=100
        → 第 1 块: "A"*500 + "B"*400 会超（900>800），
          所以先输出 "A"*500，弹出直到 <=100（弹光了），
          第 2 块从 "B" 开始……
    """

    if not atoms:
        return []

    chunks: list[str] = []
    buffer: list[str] = []
    total = 0

    for atom in atoms:
        # 放不下了 → 先结算当前 buffer
        if buffer and total + len(atom) > chunk_size:
            chunks.append("".join(buffer))

            # 从头部弹出，制造 overlap 尾巴
            # 两个退出条件：
            #   total <= overlap             → overlap 长度够了
            #   total + len(atom) <= size    → 新原子放得下了
            # 注意 `and buffer` 防止弹空后无限循环
            while buffer and (total > overlap or (total + len(atom) > chunk_size and total > 0)):
                total -= len(buffer[0])
                buffer.pop(0)

        buffer.append(atom)
        total += len(atom)

    # 循环结束后 buffer 里还有残留，别忘了输出
    if buffer:
        chunks.append("".join(buffer))

    return chunks


def split_text(text: str, config: ChunkerConfig = DEFAULT_CONFIG) -> list[str]:
    """把一段纯文本切成 chunk 列表（对外可单独使用）。"""
    # 短文本优化：输入本身就不需要切，直接原样返回
    stripped = text.strip()
    if not stripped:
        return []
    if len(stripped) <= config.chunk_size:
        # 短输入：如果连 min_chunk_size 都不够（比如 1~4 字符的"短"），
        # 那确实没检索价值，丢弃
        return [stripped] if len(stripped) >= config.min_chunk_size else []

    # 长输入走正常流程
    atoms = _atomize(text, config.separators, config.chunk_size)
    chunks = _merge_atoms(atoms, config.chunk_size, config.chunk_overlap)

    # 清洗：去掉首尾空白 + 丢弃太短的碎屑
    return [c.strip() for c in chunks if len(c.strip()) >= config.min_chunk_size]


# ============================================================
# 第三步：对 Document 列表切分并生成 chunk_id
# ============================================================
def _source_key(source_path: str) -> str:
    """把长路径压成 8 位稳定短码，用于拼 chunk_id。

    为什么不直接用路径？
      - 路径可能包含冒号（Windows 的 "C:\\..."），会和 chunk_id 的分隔符冲突
      - 路径可能很长，chunk_id 会变得难看且占空间

    为什么用 sha1 而不是内置 hash()？
      ⚠️ 关键坑：Python 的 hash("abc") 对 str 类型是**每个进程随机加盐的**
        （PYTHONHASHSEED），同一个字符串在两次运行中 hash 值不同！
        用它做 id 会导致"重启程序后所有 chunk_id 全变了"，增量索引彻底失效。
        hashlib 的哈希是确定性的，跨进程、跨机器、跨版本都一致。
    """
    return hashlib.sha1(source_path.encode("utf-8")).hexdigest()[:8]


def split_documents(
    docs: list[Document],
    config: ChunkerConfig = DEFAULT_CONFIG,
) -> list[Document]:
    """把 Document 列表切分成 chunk（保留并扩展 metadata）。

    Args:
        docs: loader 产出的 Document 列表
        config: 切分参数

    Returns:
        切分后的 Document 列表，每个都带唯一的 chunk_id

    chunk_id 格式：
        {源文件短码}:p{页码}:{该页内序号}
        例如 "a3f21b09:p12:3" = 某文件第 12 页的第 4 个 chunk

    为什么这么设计 chunk_id？
      1. **稳定**：只要文件内容和切分参数不变，重新索引得到的 id 完全一致
         → 可以用 upsert 覆盖写入，不会产生重复向量
      2. **可读**：出问题时一眼看出是哪个文件哪一页
      3. **无冲突**：短码 + 页码 + 序号三段联合唯一
    """
    chunks: list[Document] = []
    # 计数器：key 是 (源文件短码, 页码)，value 是该页已经产生了几个 chunk
    # 为什么按 (文件, 页) 分组计数，而不是用全局序号？
    #   → 全局序号会受"文档处理顺序"影响。多加一个文件，后面所有 id 全变，
    #     增量索引就废了。按页分组计数则完全局部化。
    counters: dict[tuple[str, int], int] = {}
    for doc in docs:
        pieces = split_text(doc.page_content, config)

        source_path = str(doc.metadata.get(META_SOURCE_PATH, ""))
        page = int(doc.metadata.get(META_PAGE, 1))
        key = _source_key(source_path)

        for piece in pieces:
            # setdefault + 自增：该 (文件,页) 的第几个 chunk
            seq = counters.get((key, page), 0)
            counters[(key, page)] = seq + 1

            # ⚠️ 必须 copy！直接改 doc.metadata 会让同一页的所有 chunk
            #    共享同一个 dict 对象，最后全都变成最后一个 chunk 的 id
            meta = dict(doc.metadata)
            meta[META_CHUNK_ID] = f"{key}:p{page}:{seq}"
            meta["chunk_index"] = seq

            chunks.append(Document(page_content=piece, metadata=meta))

    return chunks


def estimate_tokens(text: str) -> int:
    """粗略估算 token 数（不依赖 tiktoken，够用即可）。

    经验公式：
      - 中日韩字符：约 1 字 = 1 token（保守估计）
      - 其他字符：约 4 字符 = 1 token

    用途：在 UI 上提示"这次索引大约消耗 X token"，让用户对 API 花费有预期。
    需要精确值时才用 tiktoken（慢很多）。
    """
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    other = len(text) - cjk
    return cjk + other // 4
