"""
索引清单（Manifest）
====================

manifest.json 记录"向量库里现在到底有哪些文件、它们的内容哈希是多少"。

【为什么不直接查数据库的 indexed_at？】
indexed_at 只能回答"这个文件建过索引吗"，回答不了"建索引之后文件改过吗"。
用户改了一个 PPT 再上传，sha256 变了但 indexed_at 还在 —— 光看它就会漏掉更新。

manifest 存了每个文件**建索引时的 sha256**，于是可以精确判断三种状态：
    数据库有 + manifest 没有        → 新增，要建索引
    数据库有 + manifest 有 + sha 不同 → 修改，要先删旧向量再重建
    数据库没有 + manifest 有        → 已删除，要清理残留向量

【为什么用 JSON 而不是存数据库？】
manifest 和 Chroma 目录是"一对孪生兄弟"，放在同一个目录下，
删课程时 rmtree 一把全清掉，不会出现"数据库清了但向量还在"的不一致。
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from loguru import logger

# manifest 的格式版本。以后字段变了就 +1，读到旧版本直接当空的处理（触发全量重建）
MANIFEST_VERSION = 1

@dataclass(slots=True)
class ManifestEntry:
    """一个已索引文件的记录。"""

    source_path: str  # 绝对路径（str）
    sha256: str  # 建索引时的内容哈希
    chunk_count: int  # 产生了多少个 chunk
    indexed_at: str  # ISO8601 时间
    document_id: str = ""  # 对应 documents 表的 id

@dataclass(slots=True)
class IndexManifest:
    """整个知识库的清单。"""

    course_id: str
    version: int = MANIFEST_VERSION
    # key = source_path，value = 记录
    # field(default_factory=dict)：可变默认值必须用 default_factory，
    # 直接写 = {} 会让所有实例共享同一个 dict（Python 经典陷阱）
    entries: dict[str, ManifestEntry] = field(default_factory=dict)

    # ---------- 路径 ----------
    @staticmethod
    def path_for(course_id: str, data_dir: Path) -> Path:
        """manifest.json 的位置：与 chroma/ 同级。"""
        return data_dir / "knowledge_bases" / course_id / "manifest.json"

    # ---------- 读写 ----------
    @classmethod
    def load(cls, course_id: str, data_dir: Path) -> IndexManifest:
        """读取 manifest；文件不存在 / 损坏 / 版本不匹配都返回空清单。

        为什么损坏时不抛异常？
          - manifest 是**可重建的缓存**，不是权威数据源
          - 返回空清单只会导致"多做一次全量索引"，代价可接受
          - 抛异常则会让用户卡死在一个无法自行修复的状态
        """
        path = cls.path_for(course_id, data_dir)
        if not path.exists():
            return cls(course_id=course_id)

        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(f"manifest 损坏，按空清单处理：{exc}")
            return cls(course_id=course_id)

        if raw.get("version") != MANIFEST_VERSION:
            logger.warning("manifest 版本不匹配，将触发全量重建")
            return cls(course_id=course_id)

        entries = {
            key: ManifestEntry(**value)
            for key, value in (raw.get("entries") or {}).items()
        }
        return cls(course_id=course_id, version=raw["version"], entries=entries)

    def save(self, data_dir: Path) -> None:
        """写回磁盘（原子写：先写临时文件再改名）。

        为什么要原子写？
          直接覆盖时，如果进程在写到一半被杀掉，manifest.json 会变成半截 JSON。
          先写 .tmp 再 rename —— rename 在同一文件系统内是原子操作，
          要么完全成功，要么文件完全不变，绝不会出现中间状态。
        """
        path = self.path_for(self.course_id, data_dir)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "course_id": self.course_id,
            "version": self.version,
            # asdict 把 dataclass 递归转成普通 dict，才能被 json 序列化
            "entries": {k: asdict(v) for k, v in self.entries.items()},
        }

        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(
            # ensure_ascii=False 让中文以原样存储，方便人眼调试
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp.replace(path)  # 原子替换

    # ---------- 查询 ----------
    def needs_index(self, source_path: str, sha256: str) -> bool:
        """这个文件需要（重新）建索引吗？"""
        entry = self.entries.get(source_path)
        if entry is None:
            return True  # 全新文件
        return entry.sha256 != sha256  # 内容变了

    def stale_sources(self, current_paths: set[str]) -> list[str]:
        """找出"清单里有、但当前文档列表里没有"的路径 —— 需要清理的残留。"""
        return sorted(set(self.entries) - current_paths)

    # ---------- 修改 ----------
    def upsert(self, entry: ManifestEntry) -> None:
        self.entries[entry.source_path] = entry

    def remove(self, source_path: str) -> None:
        self.entries.pop(source_path, None)  # pop 带默认值 = 不存在也不报错

    def clear(self) -> None:
        self.entries.clear()

    @property
    def total_chunks(self) -> int:
        return sum(e.chunk_count for e in self.entries.values())
