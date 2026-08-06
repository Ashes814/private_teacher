"""
领域模型（dataclass）
=====================

这一层的职责：把 sqlite3.Row（本质是 dict）转成有类型的 Python 对象。

为什么不用 pydantic BaseModel？
  - 这些对象来自我们自己的数据库，不是外部输入，不需要 pydantic 的校验
  - dataclass 创建开销比 pydantic 小一个数量级（检索一次可能造上千个对象）
  - dataclass 是标准库，减少依赖

为什么时间要手动转 ISO 字符串？
  - Python 3.12 起，sqlite3 的默认 datetime adapter 被标记为 deprecated
  - 手动转最明确：数据库里存的就是人眼可读的 "2026-07-29T10:30:00+00:00"
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

# ============================================================
# 时间序列化辅助函数
# ============================================================
def now_utc() -> datetime:
    """当前 UTC 时间（带时区信息）。

    为什么坚持用带时区的 UTC，而不是 datetime.now()？
      - datetime.now() 返回 naive（无时区）时间，跨机器/跨时区会算错
      - UTC 存储 + 展示时转本地时区，是所有严肃项目的标准做法
    """
    return datetime.now(UTC)

def to_iso(dt: datetime | None) -> str | None:
    """datetime → ISO8601 字符串（用于写库）。None 原样返回。"""
    return None if dt is None else dt.isoformat()

def from_iso(value: str | None) -> datetime | None:
    """ISO8601 字符串 → datetime（用于读库）。None / 空串 → None。"""
    if not value:
        return None
    
    return datetime.fromisoformat(value)

# ============================================================
# 领域对象
# ============================================================

@dataclass(slots=True)
class Course:
    """一门课程。

    slots=True 的作用：
      - 不生成 __dict__，内存占用更小、属性访问更快
      - 副作用：不能给实例动态加属性（这其实是好事，能挡住打错字的 bug）
    """

    id: str
    name: str
    description: str
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Course:
        """从数据库行构造对象。

        这个方法是"数据库表结构"和"Python 对象"之间的唯一翻译点——
        以后加字段只要改这里 + schema.sql，其他代码不用动。
        """

        return cls(
            id=row["id"],
            name=row["name"],
            description=row["description"] or "",
            # from_iso 返回 datetime | None，但 created_at 在 schema 里是 NOT NULL，
            # 所以这里断言它一定有值（mypy 需要这个提示）
            created_at=from_iso(row["created_at"]), #type: ignore[arg-type]
            updated_at=from_iso(row["updated_at"]), #type: ignore[arg-type]
            
        )
    

@dataclass(slots=True)
class Document:
    """一个课件文件。"""

    id:str
    course_id: str
    path: Path
    category: str # "main" | "auxiliary"
    sha256: str
    size: int
    mtime: datetime
    indexed_at: datetime | None # None = 还没建索引

    @property
    def is_indexed(self) -> bool:
        """是否已进入向量库。UI 上用它显示 ✅ / ⏳ 图标。"""
        return self.indexed_at is not None
    
    @property
    def exists(self) -> bool:
        """物理文件是否还在（可能被用户手动删了）。"""
        return self.path.exists()
    
    @classmethod
    def from_row(cls, row: sqlite3.Row) -> Document:
        return cls(
            id=row["id"],
            course_id=row["course_id"],
            # 数据库里存的是 str，取出来立刻转回 Path，
            # 保证业务层拿到的永远是 Path 对象（不用到处 Path(...) 包一层）
            path=Path(row["path"]),
            category=row["category"],
            sha256=row["sha256"],
            size=row["size"],
            mtime=from_iso(row["mtime"]), # type: ignore[arg-type]
            indexed_at=from_iso(row["indexed_at"]),

        )