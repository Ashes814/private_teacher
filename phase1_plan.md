# Phase 1 — 课件 + RAG 知识库 详细开发计划

> 本文是 [Build_plan.md](Build_plan.md) Phase 1 阶段的**完整开发教学**，与 [phase0_plan.md](phase0_plan.md) 同风格。  
> 学完之后，你将掌握：**SQLite 存储**、**多格式文件加载（PDF/PPTX/TXT/MD/Code）**、**文本切分**、**Embedding**、**Chroma 向量库**、**RAG 索引/检索**、**源文件溯源**、**Streamlit 课件与知识库 UI**。  
> **本文结构**：每一节先是「该建什么文件、文件里该有什么、为什么这样设计、怎么验证、哪里容易踩坑」的**设计指引**，
> 然后跟一个 **「完整代码实现」小节**（§2.7 / §3.7 / §4.7 / §5.6 / §6.6 / §7.6 / §8.6 / §9.4 / §10.4 / §11.4 / §12.1.1），
> 给出可直接运行、带详细中文注释的全部源码与测试。
> **推荐用法**：先照着设计指引自己写一遍，再对照完整实现查漏补缺 —— 直接抄代码学不到东西。
> 全部文件清单与落地顺序见 **§14 附录**。

---

## 0. 前置准备（5 分钟）

### 0.1 Phase 0 收尾确认

跑通这个命令，**全绿** 才能进 Phase 1：

```bash
make ci
```

预期：
```
✅ ruff check
✅ ruff format --check
✅ mypy
✅ pytest
```

### 0.2 安装 Phase 1 依赖

```bash
uv add chromadb                    # 向量库
uv add sentence-transformers       # 本地 embedding（可选）
uv add rank-bm25                   # 关键词检索（混合检索用）
uv add unstructured                # PDF/PPTX 加载
uv add pypdf                       # 备选 PDF 库
uv add python-pptx                 # PPTX 加载
uv add tiktoken                    # token 计数（切分用）
uv add nbformat                    # Phase 3+ 提前装

uv add --dev pytest-cov            # 覆盖率（已有）
```

> **补充（写完整代码时才发现还缺的依赖）**
>
> ```bash
> uv add streamlit                   # 第 11 节 UI 必需
> uv add pdfplumber                  # PDF 兜底解析（pypdf 提取为空时用）
> ```
>
> 注意：**本文后面的完整实现不使用 `langchain-chroma`**，而是直接用 `chromadb` 原生 API。
> 原因有三：
> 1. 少一层封装依赖，langchain 集成包版本变动频繁，容易在升级时炸掉；
> 2. 原生 API 可以**自己指定向量 id**（我们用 `chunk_id`），重复索引时能 `upsert` 覆盖而不是重复插入；
> 3. 学习价值更高——你会真正看懂"向量库到底存了什么"。
>
> 我们仍然复用 `langchain_core.documents.Document` 作为**统一数据结构**（loader / splitter / retriever 之间传递），
> 因为 Phase 2 的 Agent 链路要用它。

### 0.3 Phase 0 代码快速回顾

Phase 1 依赖 Phase 0 的：
- `LLMSettings.embedding_model` 字段需要加（OpenAI/HuggingFace 二选一）
- `PathSettings.course_dir(course_id)` 已有
- 自定义异常体系（`DocumentLoadError`、`RAGError`）已有

### 0.4 概念图（RAG 全貌）

```
PDF/PPTX/TXT/Code
       ↓
   [Loader]         ← 加载：文件 → Document（含 page/line 元数据）
       ↓
   [Splitter]       ← 切分：长 Document → 短 Chunk
       ↓
   [Embedder]       ← 向量化：Chunk → vector（浮点数数组）
       ↓
   [Chroma]         ← 存储：vector + metadata
       ↓
   [Retriever]      ← 检索：query → top-k Chunks
       ↓
   [SourceResolver] ← 溯源：chunk_id → 源文件路径/页码/行号
       ↓
   LLM 问答
```

---

## 1. 阶段目标与目录结构

### 1.1 阶段目标

**输入**：用户上传的任意格式课件文件  
**输出**：本地可检索的 RAG 知识库 + 源文件溯源能力  
**可演示**：「上传 PDF → 检索『机器学习』→ 给出 3 个相关片段 + 每个片段能跳到 PDF 哪一页」

### 1.2 目录结构（新增部分）

```
private_teacher/
├── data/
│   ├── courseware/                  # 【新】课件物理存储
│   │   └── {course_id}/
│   │       ├── main/                # 主干课程（PDF/PPTX）
│   │       └── auxiliary/           # 辅助资料（代码/TXT/扩展）
│   ├── knowledge_bases/             # 【新】向量库持久化
│   │   └── {course_id}/
│   │       ├── chroma/              # Chroma 持久化目录
│   │       └── manifest.json        # 文档清单 + 哈希
│   └── app.db                       # 【新】SQLite（之前没有数据）
│
├── src/private_teacher/
│   ├── storage/                     # 【新】持久化层
│   │   ├── __init__.py
│   │   ├── db.py                    # SQLite 连接 + 迁移
│   │   ├── models.py                # 领域 dataclass
│   │   ├── course_repo.py
│   │   └── document_repo.py
│   │
│   ├── loaders/                     # 【新】课件加载
│   │   ├── __init__.py
│   │   ├── registry.py             # 扩展名 → loader 分发
│   │   ├── pdf_loader.py
│   │   ├── pptx_loader.py
│   │   ├── text_loader.py
│   │   ├── markdown_loader.py
│   │   ├── code_loader.py           # 按语言切分，保留行号
│   │   └── directory_loader.py     # 递归扫描
│   │
│   ├── rag/                         # 【新】RAG 核心
│   │   ├── __init__.py
│   │   ├── embeddings.py           # Embedder 工厂
│   │   ├── vector_store.py          # ChromaStore 抽象
│   │   ├── splitters.py            # Recursive + CodeAware
│   │   ├── indexer.py              # 从课件构建 KB
│   │   ├── retriever.py            # 混合检索（向量 + BM25）
│   │   ├── source_resolver.py      # chunk_id → 源文件
│   │   └── reranker.py             # 可选 cross-encoder
│   │
│   ├── services/                    # 【新】业务编排
│   │   ├── __init__.py
│   │   ├── course_service.py       # 课程 CRUD
│   │   └── kb_service.py            # 知识库构建/检索
│   │
│   └── ui/                          # 【新】Streamlit 课件 + 知识库页
│       └── pages/
│           ├── 1_📚_courseware.py
│           └── 2_🧠_knowledge_base.py
│
└── tests/
    ├── fixtures/
    │   └── courseware/              # 【新】测试用样本文件
    │       ├── sample.pdf
    │       ├── sample.txt
    │       ├── sample.md
    │       └── sample.py
    ├── unit/
    │   ├── test_loaders.py          # 【新】
    │   ├── test_splitters.py        # 【新】
    │   ├── test_embeddings.py       # 【新】
    │   ├── test_vector_store.py     # 【新】
    │   ├── test_indexer.py          # 【新】
    │   ├── test_retriever.py        # 【新】
    │   ├── test_source_resolver.py  # 【新】
    │   ├── test_storage.py          # 【新】
    │   └── test_services.py         # 【新】
    └── integration/
        └── test_rag_pipeline.py     # 【新】端到端测试
```

### 1.3 7 天开发节奏

| Day | 任务 | 可演示 |
| --- | --- | --- |
| Day 1 | 存储层（SQLite + repos） | 创建/查询课程和文档 |
| Day 2 | 加载器（PDF/TXT/MD/PY/PPTX） | 上传文件成功入库 |
| Day 3 | 切分器 + Embedding | 把文档切成 chunk 并向量化 |
| Day 4 | Chroma 向量库 + Indexer | 完整构建知识库 |
| Day 5 | Retriever + SourceResolver | 检索 demo |
| Day 6 | Services + Streamlit UI | 上传 + 检索的 web 界面 |
| Day 7 | E2E 测试 + 收尾 | 完整流程跑通 |

---

## 2. 存储层（SQLite + Repos）

### 2.1 目标

把 Phase 0 在 [Build_plan.md 第 7.1 节](Build_plan.md#71-sqlite-schemadataappdb) 设计的 `courses` / `documents` 表落地为代码。

### 2.2 概念小课堂

- **SQLite**：Python 标准库自带 `sqlite3`，文件型数据库，零配置
- **Repository 模式**：把"怎么读写 DB"封装成类，业务层不直接写 SQL
- **dataclass**：Python 内置的「数据类」，比 dict 更结构化，比 ORM 更轻

### 2.3 文件内容指引

#### `src/private_teacher/storage/db.py`

**导入**：
- `sqlite3`（标准库）
- `pathlib.Path`
- `private_teacher.config.PathSettings`

**要实现**：
- `def get_connection(db_path: Path) -> sqlite3.Connection`：获取连接（带 row_factory=Row）
- `def init_db(conn: sqlite3.Connection) -> None`：执行 schema.sql 创建表
- `class Database` 上下文管理器：`with Database() as conn: ...` 自动 open/close

**schema.sql 文件**（放在 `src/private_teacher/storage/schema.sql`）：
- 包含 `Build_plan.md` 第 7.1 节的两张表
- 加上 `CREATE INDEX IF NOT EXISTS idx_documents_course_id ON documents(course_id)`

**思考题**：
- 为什么用 `row_factory=sqlite3.Row`？→ 让结果可以像 dict 一样用 `row["column"]` 访问
- 为什么 `init_db` 用 `IF NOT EXISTS`？→ 重复执行不报错，方便测试

#### `src/private_teacher/storage/models.py`

**用 `@dataclass` 定义**：

```python
@dataclass
class Course:
    id: str
    name: str
    description: str
    created_at: datetime
    updated_at: datetime

@dataclass
class Document:
    id: str
    course_id: str
    path: Path
    category: str          # "main" | "auxiliary"
    sha256: str
    size: int
    mtime: datetime
    indexed_at: datetime | None
```

**思考题**：
- 为什么用 dataclass 而不是 pydantic BaseModel？→ 性能更好、不依赖 pydantic、与 sqlite3.Row 转换简单
- `indexed_at: datetime | None` 的 None 含义？→ 还没建索引

#### `src/private_teacher/storage/course_repo.py`

**方法**（每个方法接 `conn` 作为参数）：
- `def create(conn, name: str, description: str = "") -> Course`
- `def get_by_id(conn, course_id: str) -> Course | None`
- `def list_all(conn) -> list[Course]`
- `def delete(conn, course_id: str) -> None`（级联删 documents）
- `def update(conn, course_id: str, name: str, description: str) -> None`

**关键**：
- `id` 用 `uuid.uuid4().hex[:8]` 生成 8 字符短 ID
- `created_at` / `updated_at` 用 `datetime.now(UTC)`
- 所有 SQL 用**参数化**（防止 SQL 注入）

#### `src/private_teacher/storage/document_repo.py`

**方法**：
- `def create(conn, course_id, path, category, sha256, size) -> Document`
- `def get_by_sha256(conn, course_id, sha256) -> Document | None`（**用于去重**）
- `def list_by_course(conn, course_id, category: str | None = None) -> list[Document]`
- `def delete(conn, document_id: str) -> None`
- `def mark_indexed(conn, document_id: str) -> None`（更新 indexed_at）
- `def get_indexed(conn, course_id) -> list[Document]`（找出已建索引的）

### 2.4 测试代码

**tests/unit/test_storage.py**

```python
"""测试 storage 层。"""
import pytest
from private_teacher.storage.db import Database, init_db
from private_teacher.storage.course_repo import (
    create as create_course, get_by_id, list_all, delete,
)
from private_teacher.storage.document_repo import (
    create as create_doc, get_by_sha256,
)


class TestCourseRepo:
    def test_create_and_get(self, tmp_db):
        course = create_course(tmp_db, name="Python 入门")
        assert course.id is not None
        assert course.name == "Python 入门"
        assert get_by_id(tmp_db, course.id).name == "Python 入门"

    def test_list_all(self, tmp_db):
        create_course(tmp_db, name="A")
        create_course(tmp_db, name="B")
        assert len(list_all(tmp_db)) == 2

    def test_delete_cascades_documents(self, tmp_db):
        course = create_course(tmp_db, name="A")
        create_doc(tmp_db, course.id, Path("a.pdf"), "main", "abc123", 100)
        delete(tmp_db, course.id)
        assert list_by_course(tmp_db, course.id) == []


class TestDocumentRepo:
    def test_sha256_dedup(self, tmp_db):
        course = create_course(tmp_db, name="A")
        create_doc(tmp_db, course.id, Path("a.pdf"), "main", "abc123", 100)
        # 同一 sha256 重复上传
        with pytest.raises(IntegrityError):
            create_doc(tmp_db, course.id, Path("a.pdf"), "main", "abc123", 100)
```

**conftest.py 加 `tmp_db` fixture**：
```python
@pytest.fixture
def tmp_db(tmp_path):
    db_path = tmp_path / "test.db"
    with Database(db_path) as conn:
        init_db(conn)
        yield conn
```

### 2.5 验证

```bash
uv run pytest tests/unit/test_storage.py -v
```

预期：**6+ 个测试全过**。

### 2.6 踩坑

| 现象 | 原因 | 解决 |
| --- | --- | --- |
| `sqlite3.IntegrityError: UNIQUE constraint failed` | 重复 sha256 插入 | `documents` 加 `UNIQUE(course_id, sha256)` 约束 |
| `conn` 没自动关闭 | 没 with | 用 `Database` 上下文管理器 |
| 路径在不同 OS 写法不同 | `/` vs `\` | 存数据库用 `str(path)`，显示用 `path.as_posix()` |

---

### 2.7 完整代码实现

> 下面是第 2 节全部文件的可运行实现。**建议先自己写一遍，再对照这里查漏补缺。**

#### 2.7.1 `src/private_teacher/utils/hashing.py`（新增工具，后面 §10 也要用）

```python
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
```

#### 2.7.2 `src/private_teacher/storage/schema.sql`

```sql
-- ============================================================
-- Private Teacher 数据库 Schema
-- ============================================================
-- 约定：
--   1. 所有建表语句都用 IF NOT EXISTS —— 重复执行不报错，方便测试反复 init
--   2. 时间统一存 ISO8601 字符串（如 "2026-07-29T10:30:00+00:00"）
--      不用 sqlite 的 TIMESTAMP 自动转换，因为 Python 3.12 已废弃默认的
--      datetime adapter，会打 DeprecationWarning。我们在 Python 侧手动转。
--   3. 主键用 TEXT（8 位 uuid 短 id），不用自增整数
--      —— 短 id 可以直接当目录名，且不暴露"你一共建了几门课"
-- ============================================================

-- ---------- 课程表 ----------
CREATE TABLE IF NOT EXISTS courses (
    id          TEXT PRIMARY KEY,          -- uuid4 前 8 位，如 "a1b2c3d4"
    name        TEXT NOT NULL,             -- 课程名，如 "机器学习导论"
    description TEXT NOT NULL DEFAULT '',  -- 课程描述，允许空串但不允许 NULL
    created_at  TEXT NOT NULL,             -- ISO8601 UTC
    updated_at  TEXT NOT NULL              -- ISO8601 UTC
);

-- ---------- 文档表 ----------
CREATE TABLE IF NOT EXISTS documents (
    id         TEXT PRIMARY KEY,
    course_id  TEXT NOT NULL
        REFERENCES courses(id) ON DELETE CASCADE,   -- 删课程时自动删文档
    path       TEXT NOT NULL,              -- 文件落盘后的绝对/相对路径（str 形式）
    category   TEXT NOT NULL
        CHECK (category IN ('main', 'auxiliary')),  -- 只允许这两个值，DB 层兜底
    sha256     TEXT NOT NULL,              -- 文件内容哈希，用于去重 + 增量索引
    size       INTEGER NOT NULL,           -- 文件字节数
    mtime      TEXT NOT NULL,              -- 文件修改时间（ISO8601）
    indexed_at TEXT,                       -- 建索引时间；NULL = 还没进向量库

    -- 关键约束：同一门课里，同样内容的文件只能存一份
    -- 注意是 (course_id, sha256) 组合唯一，不同课程可以各存一份同样的 PDF
    UNIQUE (course_id, sha256)
);

-- ---------- 索引 ----------
-- 场景：几乎每次查询都是 "把某门课的文档列出来"，course_id 上建索引避免全表扫
CREATE INDEX IF NOT EXISTS idx_documents_course_id
    ON documents (course_id);

-- 场景：增量索引时要快速找出 "还没建索引的文档"
CREATE INDEX IF NOT EXISTS idx_documents_indexed_at
    ON documents (course_id, indexed_at);
```

> **打包提醒**：`schema.sql` 不是 `.py` 文件，某些打包后端默认不会把它放进 wheel。
> `uv_build` 默认会打包 module 目录下的所有文件，所以本项目开箱即用；
> 如果以后换成 setuptools，记得在 `pyproject.toml` 加 `[tool.setuptools.package-data]`。

#### 2.7.3 `src/private_teacher/storage/db.py`

```python
"""
SQLite 连接管理
===============

三个对外接口：
  - get_connection(db_path)  : 拿一个配置好的连接
  - init_db(conn)            : 建表（幂等）
  - Database                 : with 语句上下文管理器（自动 commit / rollback / close）

为什么不用 SQLAlchemy / ORM？
  - 本项目表结构极简（Phase 1 只有 2 张表），ORM 的抽象成本大于收益
  - 直接写 SQL 能让你看清"到底发了什么语句"，学习价值更高
  - 需要的时候再迁移到 ORM 不难（Repository 模式已经把 SQL 隔离在一层里了）
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from types import TracebackType

from loguru import logger

from private_teacher.config import PathSettings

# schema.sql 与本文件同目录，用 __file__ 定位，避免依赖"当前工作目录"
SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def get_connection(
    db_path: Path,
    *,
    check_same_thread: bool = True,
) -> sqlite3.Connection:
    """创建并配置一个 SQLite 连接。

    Args:
        db_path: .db 文件路径；父目录不存在会自动创建
        check_same_thread:
            默认 True = 连接只能在创建它的线程里用（sqlite3 的安全默认值）。
            Streamlit 会在不同线程处理请求，那里要传 False。
            传 False 的前提是：你保证同一时刻只有一个线程在用这个连接。

    Returns:
        已设置 row_factory 和 PRAGMA 的连接
    """
    # 父目录不存在时 sqlite3.connect 会直接报 "unable to open database file"
    # 这里主动创建，让调用方少踩一个坑
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path), check_same_thread=check_same_thread)

    # ---------- row_factory ----------
    # 默认返回 tuple：row[0], row[1]... 可读性极差，且加字段就错位
    # 设成 sqlite3.Row 后可以 row["name"]，同时仍支持 row[0]
    conn.row_factory = sqlite3.Row

    # ---------- 外键约束 ----------
    # ⚠️ 大坑：SQLite 默认【不开启】外键约束，
    #    而且这是【连接级别】的设置——每个新连接都要重新执行一次
    #    不开的话 ON DELETE CASCADE 完全不生效，删课程后文档变成孤儿数据
    conn.execute("PRAGMA foreign_keys = ON")

    # ---------- WAL 模式 ----------
    # Write-Ahead Logging：读写可以并发（读不会被写阻塞）
    # Streamlit 多标签页 / 后台索引任务同时访问时能显著减少 "database is locked"
    conn.execute("PRAGMA journal_mode = WAL")

    # ---------- 锁等待 ----------
    # 拿不到锁时最多等 5 秒再报错，而不是立刻抛 OperationalError
    conn.execute("PRAGMA busy_timeout = 5000")

    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """执行 schema.sql 建表。幂等——重复调用安全。

    Args:
        conn: get_connection() 返回的连接
    """
    sql = SCHEMA_PATH.read_text(encoding="utf-8")

    # executescript 能一次执行多条以分号分隔的语句
    # 注意：executescript 会先隐式 commit 当前事务，所以别在事务中间调用它
    conn.executescript(sql)
    conn.commit()

    logger.debug("数据库 schema 初始化完成")


class Database:
    """SQLite 连接的上下文管理器。

    用法一（推荐，脚本 / 测试）：
        with Database(Path("data/app.db")) as conn:
            init_db(conn)
            create(conn, name="Python 入门")
        # 退出 with 时自动 commit + close；出异常则 rollback + close

    用法二（长生命周期，如 Streamlit）：
        db = Database(path, check_same_thread=False)
        conn = db.connect()      # 手动开
        ...
        db.close()               # 手动关

    为什么要写这个类？
      - 忘记 commit 是新手最常见的 bug（数据"写了"但重启就没了）
      - 忘记 close 会泄漏文件句柄
      - with 语句把这两件事变成语言层面的保证
    """

    def __init__(
        self,
        db_path: Path | str | None = None,
        *,
        check_same_thread: bool = True,
    ) -> None:
        """
        Args:
            db_path: 不传则用 PathSettings().data_dir / "app.db"
            check_same_thread: 见 get_connection
        """
        if db_path is None:
            db_path = PathSettings().data_dir / "app.db"

        self.db_path = Path(db_path)
        self.check_same_thread = check_same_thread
        # 类型注解写成 Optional，表示"还没连"
        self._conn: sqlite3.Connection | None = None

    # ---------- 显式接口 ----------
    def connect(self) -> sqlite3.Connection:
        """建立连接（幂等：已连接则直接返回）。"""
        if self._conn is None:
            self._conn = get_connection(
                self.db_path,
                check_same_thread=self.check_same_thread,
            )
        return self._conn

    def close(self) -> None:
        """关闭连接（幂等：未连接则什么也不做）。"""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # ---------- 上下文管理器协议 ----------
    def __enter__(self) -> sqlite3.Connection:
        """进入 with 块，返回值就是 as 后面拿到的东西。"""
        return self.connect()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        """退出 with 块。

        返回 False 表示"不吞掉异常"——异常会继续往外抛。
        返回 True 会把异常吃掉，那是极其危险的默认行为，别这么干。
        """
        if self._conn is not None:
            if exc_type is None:
                # 正常退出 → 提交
                self._conn.commit()
            else:
                # 出异常 → 回滚，保证不留半截数据
                self._conn.rollback()
                logger.warning(f"数据库事务回滚：{exc!r}")
            self.close()
        return False
```

#### 2.7.4 `src/private_teacher/storage/models.py`

```python
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
            created_at=from_iso(row["created_at"]),  # type: ignore[arg-type]
            updated_at=from_iso(row["updated_at"]),  # type: ignore[arg-type]
        )


@dataclass(slots=True)
class Document:
    """一个课件文件。"""

    id: str
    course_id: str
    path: Path
    category: str  # "main" | "auxiliary"
    sha256: str
    size: int
    mtime: datetime
    indexed_at: datetime | None  # None = 还没建索引

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
            mtime=from_iso(row["mtime"]),  # type: ignore[arg-type]
            indexed_at=from_iso(row["indexed_at"]),
        )
```

#### 2.7.5 `src/private_teacher/storage/course_repo.py`

```python
"""
课程仓储（Repository）
======================

设计选择：**模块级函数** 而不是类。

为什么？
  - 这些函数没有共享状态（conn 每次都是参数传进来的）
  - 类只会多一层 self，没有任何收益
  - 需要替换实现时，Python 的模块本身就可以被 monkeypatch

命名冲突提醒：
  course_repo.create 和 document_repo.create 同名，
  导入时用别名区分：
      from private_teacher.storage.course_repo import create as create_course
"""

from __future__ import annotations

import sqlite3
import uuid

from loguru import logger

from private_teacher.storage.models import Course, now_utc, to_iso


def _new_id() -> str:
    """生成 8 位短 id。

    uuid4().hex 是 32 位十六进制，取前 8 位：
      - 碰撞概率：16^8 ≈ 43 亿种组合，个人项目几百门课，碰撞概率约等于 0
      - 短 id 可以直接当目录名（data/courseware/a1b2c3d4/），比 36 位 uuid 好看太多
    真要严谨，可以在 INSERT 失败（PRIMARY KEY 冲突）时重试一次。
    """
    return uuid.uuid4().hex[:8]


def create(conn: sqlite3.Connection, name: str, description: str = "") -> Course:
    """新建课程。

    Args:
        conn: 数据库连接
        name: 课程名（不做唯一性约束，允许同名——用 id 区分）
        description: 课程描述

    Returns:
        新建的 Course 对象（含生成的 id 和时间戳）
    """
    course_id = _new_id()
    now = now_utc()

    conn.execute(
        # ⚠️ 永远用 ? 占位符，永远不要用 f-string 拼 SQL
        #    f"INSERT ... VALUES ('{name}')" 会被 name = "'); DROP TABLE courses; --" 打穿
        """
        INSERT INTO courses (id, name, description, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (course_id, name, description, to_iso(now), to_iso(now)),
    )
    conn.commit()

    logger.info(f"创建课程：{name}（id={course_id}）")

    # 直接用已知数据构造对象，省一次 SELECT
    return Course(
        id=course_id,
        name=name,
        description=description,
        created_at=now,
        updated_at=now,
    )


def get_by_id(conn: sqlite3.Connection, course_id: str) -> Course | None:
    """按 id 查课程。找不到返回 None（而不是抛异常）。

    为什么返回 None 而不抛异常？
      - "查不到"是正常业务分支（比如用户删了课但浏览器还开着旧页面）
      - 调用方用 `if course is None:` 处理，比 try/except 清晰
    """
    row = conn.execute(
        "SELECT * FROM courses WHERE id = ?",
        (course_id,),
    ).fetchone()

    return Course.from_row(row) if row else None


def list_all(conn: sqlite3.Connection) -> list[Course]:
    """列出所有课程，最近更新的排前面。"""
    rows = conn.execute(
        "SELECT * FROM courses ORDER BY updated_at DESC"
    ).fetchall()

    return [Course.from_row(r) for r in rows]


def update(
    conn: sqlite3.Connection,
    course_id: str,
    name: str | None = None,
    description: str | None = None,
) -> Course | None:
    """更新课程。传 None 的字段保持不变。

    Returns:
        更新后的 Course；课程不存在时返回 None
    """
    course = get_by_id(conn, course_id)
    if course is None:
        return None

    # None 表示"不改这个字段"，所以用旧值兜底
    new_name = name if name is not None else course.name
    new_desc = description if description is not None else course.description
    now = now_utc()

    conn.execute(
        """
        UPDATE courses
           SET name = ?, description = ?, updated_at = ?
         WHERE id = ?
        """,
        (new_name, new_desc, to_iso(now), course_id),
    )
    conn.commit()

    return Course(
        id=course_id,
        name=new_name,
        description=new_desc,
        created_at=course.created_at,
        updated_at=now,
    )


def delete(conn: sqlite3.Connection, course_id: str) -> None:
    """删除课程。

    documents 表上声明了 ON DELETE CASCADE，
    所以这里删课程会**自动**删掉它名下所有文档记录。

    ⚠️ 前提：连接必须执行过 PRAGMA foreign_keys = ON
       （get_connection 里已经做了；如果你自己 sqlite3.connect，级联不会生效！）

    注意：这里只删数据库记录，**不删物理文件和向量库**。
    那是 CourseService.delete_course 的职责（见 §10）——
    Repository 层只管数据库，不碰文件系统，边界要清楚。
    """
    conn.execute("DELETE FROM courses WHERE id = ?", (course_id,))
    conn.commit()
    logger.info(f"删除课程：id={course_id}")


def touch(conn: sqlite3.Connection, course_id: str) -> None:
    """只更新 updated_at。

    用途：上传文档、重建知识库之后，把课程"顶"到列表最前面。
    """
    conn.execute(
        "UPDATE courses SET updated_at = ? WHERE id = ?",
        (to_iso(now_utc()), course_id),
    )
    conn.commit()
```

#### 2.7.6 `src/private_teacher/storage/document_repo.py`

```python
"""
文档仓储
========

与 course_repo 同构。多出来的两个关键能力：
  1. get_by_sha256  → 上传去重
  2. mark_indexed / list_unindexed → 增量索引的状态追踪
"""

from __future__ import annotations

import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger

from private_teacher.storage.models import Document, now_utc, to_iso

# 合法的分类值。放成常量，避免各处硬编码字符串打错字
VALID_CATEGORIES = ("main", "auxiliary")


def _new_id() -> str:
    return uuid.uuid4().hex[:8]


def create(
    conn: sqlite3.Connection,
    course_id: str,
    path: Path,
    category: str,
    sha256: str,
    size: int,
    mtime: datetime | None = None,
) -> Document:
    """登记一个文档。

    Args:
        conn: 连接
        course_id: 所属课程
        path: 文件路径（会被转成 str 存库）
        category: "main" 或 "auxiliary"
        sha256: 文件哈希（调用方用 compute_sha256 算好传进来）
        size: 字节数
        mtime: 文件修改时间；不传则尝试从文件读，读不到用当前时间

    Returns:
        新建的 Document

    Raises:
        ValueError: category 非法
        sqlite3.IntegrityError: (course_id, sha256) 重复 —— 即重复上传
    """
    # 在 Python 侧先校验一次，报错信息比 SQLite 的 CHECK constraint failed 友好得多
    # DB 层的 CHECK 约束是最后一道防线（防止有人绕过这个函数直接写 SQL）
    if category not in VALID_CATEGORIES:
        raise ValueError(f"category 必须是 {VALID_CATEGORIES} 之一，收到 {category!r}")

    if mtime is None:
        if path.exists():
            # st_mtime 是 Unix 时间戳（float），转成带时区的 datetime
            mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
        else:
            # 测试里常传虚构路径，兜底成当前时间，不要因此崩掉
            mtime = now_utc()

    doc_id = _new_id()

    conn.execute(
        """
        INSERT INTO documents
            (id, course_id, path, category, sha256, size, mtime, indexed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, NULL)
        """,
        #                                          ⬆ 新文档一定还没建索引
        (doc_id, course_id, str(path), category, sha256, size, to_iso(mtime)),
    )
    conn.commit()

    logger.info(f"登记文档：{path.name}（course={course_id}, category={category}）")

    return Document(
        id=doc_id,
        course_id=course_id,
        path=path,
        category=category,
        sha256=sha256,
        size=size,
        mtime=mtime,
        indexed_at=None,
    )


def get_by_id(conn: sqlite3.Connection, document_id: str) -> Document | None:
    row = conn.execute(
        "SELECT * FROM documents WHERE id = ?",
        (document_id,),
    ).fetchone()
    return Document.from_row(row) if row else None


def get_by_sha256(
    conn: sqlite3.Connection,
    course_id: str,
    sha256: str,
) -> Document | None:
    """按内容哈希查文档 —— **这是去重的核心**。

    上传流程：
        sha = compute_sha256(f)
        if get_by_sha256(conn, course_id, sha):
            raise FileExistsError("这个文件已经传过了")

    注意查询条件带 course_id：不同课程可以各自持有同一份 PDF。
    """
    row = conn.execute(
        "SELECT * FROM documents WHERE course_id = ? AND sha256 = ?",
        (course_id, sha256),
    ).fetchone()
    return Document.from_row(row) if row else None


def list_by_course(
    conn: sqlite3.Connection,
    course_id: str,
    category: str | None = None,
) -> list[Document]:
    """列出某门课的文档。

    Args:
        category: 传 "main" / "auxiliary" 过滤；None = 全要

    实现技巧：动态拼 WHERE 子句时，
    **拼的是 SQL 结构（列名/关键字），值永远走参数**，这样依然安全。
    """
    sql = "SELECT * FROM documents WHERE course_id = ?"
    params: list[object] = [course_id]

    if category is not None:
        sql += " AND category = ?"
        params.append(category)

    sql += " ORDER BY category, path"

    rows = conn.execute(sql, params).fetchall()
    return [Document.from_row(r) for r in rows]


def list_unindexed(conn: sqlite3.Connection, course_id: str) -> list[Document]:
    """列出还没建索引的文档（indexed_at IS NULL）—— 增量索引用。

    ⚠️ SQL 里判空必须用 `IS NULL`，写成 `= NULL` 永远返回 false（三值逻辑）。
    """
    rows = conn.execute(
        """
        SELECT * FROM documents
         WHERE course_id = ? AND indexed_at IS NULL
         ORDER BY path
        """,
        (course_id,),
    ).fetchall()
    return [Document.from_row(r) for r in rows]


def get_indexed(conn: sqlite3.Connection, course_id: str) -> list[Document]:
    """列出已经建好索引的文档。UI 上显示"知识库覆盖了哪些文件"。"""
    rows = conn.execute(
        """
        SELECT * FROM documents
         WHERE course_id = ? AND indexed_at IS NOT NULL
         ORDER BY path
        """,
        (course_id,),
    ).fetchall()
    return [Document.from_row(r) for r in rows]


def mark_indexed(conn: sqlite3.Connection, document_id: str) -> None:
    """标记为"已建索引"，写入当前时间。"""
    conn.execute(
        "UPDATE documents SET indexed_at = ? WHERE id = ?",
        (to_iso(now_utc()), document_id),
    )
    conn.commit()


def mark_all_unindexed(conn: sqlite3.Connection, course_id: str) -> None:
    """把整门课的文档全部标记为"未索引"。

    用途：全量重建知识库时先调它，再逐个建索引 —— 保证状态和向量库一致。
    """
    conn.execute(
        "UPDATE documents SET indexed_at = NULL WHERE course_id = ?",
        (course_id,),
    )
    conn.commit()


def delete(conn: sqlite3.Connection, document_id: str) -> None:
    """删除文档记录（不删物理文件、不删向量）。"""
    conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))
    conn.commit()
    logger.info(f"删除文档记录：id={document_id}")
```

#### 2.7.7 `src/private_teacher/storage/__init__.py`

```python
"""持久化层。

这里做 re-export，让外部可以写：
    from private_teacher.storage import Database, Course, Document
而不必记住每个符号在哪个子模块里。

⚠️ 注意：course_repo / document_repo 故意**不**在这里 re-export 函数，
   因为两边都有 create / delete / get_by_id，扁平化后必然撞名。
   用的时候请显式导入模块：
       from private_teacher.storage import course_repo, document_repo
       course_repo.create(conn, name="...")
"""

from private_teacher.storage import course_repo, document_repo
from private_teacher.storage.db import Database, get_connection, init_db
from private_teacher.storage.models import Course, Document, now_utc

__all__ = [
    "Course",
    "Database",
    "Document",
    "course_repo",
    "document_repo",
    "get_connection",
    "init_db",
    "now_utc",
]
```

#### 2.7.8 `tests/conftest.py` 追加的 fixture

```python
# ============================================================
# Phase 1：数据库 fixture
# ============================================================
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture
def tmp_db(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    """一个建好表的临时数据库连接，测试结束自动销毁。

    tmp_path 是 pytest 内置 fixture：每个测试函数独享一个空目录，
    测试结束后 pytest 自动清理（默认保留最近 3 次运行，方便排查）。

    为什么用真数据库而不是 mock？
      - SQLite 建一个内存/临时库只要几毫秒，比写 mock 便宜
      - mock 数据库最容易出现"测试全过，上线全崩"——SQL 语法错误 mock 根本发现不了
    """
    from private_teacher.storage.db import Database, init_db

    db_path = tmp_path / "test.db"
    with Database(db_path) as conn:
        init_db(conn)
        # yield 之前的代码 = setup，之后的代码 = teardown
        # 这里 teardown 交给 with 语句（自动 commit + close）
        yield conn


@pytest.fixture
def sample_course(tmp_db: sqlite3.Connection):
    """预置一门课，省得每个测试都手写创建逻辑。"""
    from private_teacher.storage import course_repo

    return course_repo.create(tmp_db, name="测试课程", description="用于单元测试")
```

#### 2.7.9 `tests/unit/test_storage.py`（完整版）

```python
"""storage 层单元测试。

测试组织原则：
  - 一个 class 测一个模块，class 名 = Test + 被测对象
  - 一个 test 方法只断言一件事（失败时一眼看出是哪里坏了）
  - 方法名写成句子：test_<做了什么>_<期望什么>
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

import pytest

from private_teacher.storage import course_repo, document_repo
from private_teacher.storage.db import Database, init_db
from private_teacher.storage.models import Course, Document, from_iso, to_iso


# ============================================================
# db.py
# ============================================================
class TestDatabase:
    def test_init_db_is_idempotent(self, tmp_path: Path) -> None:
        """重复 init_db 不应该报错（IF NOT EXISTS 生效）。"""
        db_path = tmp_path / "a.db"
        with Database(db_path) as conn:
            init_db(conn)
            init_db(conn)  # 第二次，不该炸
            # sqlite_master 是 SQLite 的系统表，存了所有表结构
            tables = {
                r["name"]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
        assert {"courses", "documents"} <= tables

    def test_foreign_keys_enabled(self, tmp_db: sqlite3.Connection) -> None:
        """外键必须是开启状态，否则 CASCADE 不生效。"""
        # PRAGMA 查询会返回一行一列，1 = 开启
        assert tmp_db.execute("PRAGMA foreign_keys").fetchone()[0] == 1

    def test_row_factory_allows_dict_access(self, tmp_db: sqlite3.Connection) -> None:
        """row_factory=Row 让结果可以按列名访问。"""
        course_repo.create(tmp_db, name="X")
        row = tmp_db.execute("SELECT * FROM courses").fetchone()
        assert row["name"] == "X"  # 不是 row[1]

    def test_rollback_on_exception(self, tmp_path: Path) -> None:
        """with 块里抛异常时应该回滚，数据不落库。"""
        db_path = tmp_path / "b.db"
        with Database(db_path) as conn:
            init_db(conn)

        with pytest.raises(RuntimeError):
            with Database(db_path) as conn:
                conn.execute(
                    "INSERT INTO courses VALUES ('x','n','',?,?)",
                    (to_iso(datetime.now()), to_iso(datetime.now())),
                )
                raise RuntimeError("模拟业务异常")

        # 重新打开检查：那条数据不该存在
        with Database(db_path) as conn:
            assert conn.execute("SELECT COUNT(*) FROM courses").fetchone()[0] == 0


# ============================================================
# models.py
# ============================================================
class TestModels:
    def test_iso_roundtrip(self) -> None:
        """datetime → str → datetime 应该无损。"""
        from private_teacher.storage.models import now_utc

        dt = now_utc()
        assert from_iso(to_iso(dt)) == dt

    def test_iso_handles_none(self) -> None:
        assert to_iso(None) is None
        assert from_iso(None) is None
        assert from_iso("") is None

    def test_document_is_indexed_property(self, tmp_path: Path) -> None:
        doc = Document(
            id="d1",
            course_id="c1",
            path=tmp_path / "a.txt",
            category="main",
            sha256="abc",
            size=1,
            mtime=datetime.now(),
            indexed_at=None,
        )
        assert doc.is_indexed is False
        doc.indexed_at = datetime.now()
        assert doc.is_indexed is True


# ============================================================
# course_repo.py
# ============================================================
class TestCourseRepo:
    def test_create_and_get(self, tmp_db: sqlite3.Connection) -> None:
        course = course_repo.create(tmp_db, name="Python 入门")

        assert course.id  # 非空
        assert len(course.id) == 8
        assert course.name == "Python 入门"

        fetched = course_repo.get_by_id(tmp_db, course.id)
        assert fetched is not None
        assert fetched.name == "Python 入门"

    def test_get_missing_returns_none(self, tmp_db: sqlite3.Connection) -> None:
        assert course_repo.get_by_id(tmp_db, "nonexist") is None

    def test_list_all(self, tmp_db: sqlite3.Connection) -> None:
        course_repo.create(tmp_db, name="A")
        course_repo.create(tmp_db, name="B")
        assert len(course_repo.list_all(tmp_db)) == 2

    def test_update_partial(self, tmp_db: sqlite3.Connection) -> None:
        """只传 name 时 description 应保持不变。"""
        c = course_repo.create(tmp_db, name="A", description="原描述")
        updated = course_repo.update(tmp_db, c.id, name="B")

        assert updated is not None
        assert updated.name == "B"
        assert updated.description == "原描述"

    def test_update_missing_returns_none(self, tmp_db: sqlite3.Connection) -> None:
        assert course_repo.update(tmp_db, "nope", name="X") is None

    def test_delete_cascades_documents(self, tmp_db: sqlite3.Connection) -> None:
        """删课程时它名下的文档应该一起消失。"""
        course = course_repo.create(tmp_db, name="A")
        document_repo.create(
            tmp_db, course.id, Path("a.pdf"), "main", "abc123", 100
        )
        assert len(document_repo.list_by_course(tmp_db, course.id)) == 1

        course_repo.delete(tmp_db, course.id)

        assert course_repo.get_by_id(tmp_db, course.id) is None
        assert document_repo.list_by_course(tmp_db, course.id) == []


# ============================================================
# document_repo.py
# ============================================================
class TestDocumentRepo:
    def test_create_returns_unindexed(self, sample_course, tmp_db) -> None:
        doc = document_repo.create(
            tmp_db, sample_course.id, Path("a.pdf"), "main", "h1", 100
        )
        assert doc.indexed_at is None
        assert doc.is_indexed is False

    def test_invalid_category_raises(self, sample_course, tmp_db) -> None:
        with pytest.raises(ValueError, match="category"):
            document_repo.create(
                tmp_db, sample_course.id, Path("a.pdf"), "bad", "h1", 100
            )

    def test_sha256_dedup(self, sample_course, tmp_db) -> None:
        """同一门课里，相同 sha256 不允许插两次。"""
        document_repo.create(
            tmp_db, sample_course.id, Path("a.pdf"), "main", "same", 100
        )
        with pytest.raises(sqlite3.IntegrityError):
            document_repo.create(
                tmp_db, sample_course.id, Path("b.pdf"), "main", "same", 100
            )

    def test_same_sha_allowed_across_courses(self, tmp_db) -> None:
        """不同课程可以各自持有同一份文件。"""
        c1 = course_repo.create(tmp_db, name="A")
        c2 = course_repo.create(tmp_db, name="B")
        document_repo.create(tmp_db, c1.id, Path("a.pdf"), "main", "same", 1)
        document_repo.create(tmp_db, c2.id, Path("a.pdf"), "main", "same", 1)
        # 没抛异常就算过

    def test_get_by_sha256(self, sample_course, tmp_db) -> None:
        document_repo.create(
            tmp_db, sample_course.id, Path("a.pdf"), "main", "hhh", 1
        )
        assert document_repo.get_by_sha256(tmp_db, sample_course.id, "hhh") is not None
        assert document_repo.get_by_sha256(tmp_db, sample_course.id, "xxx") is None

    def test_list_by_category(self, sample_course, tmp_db) -> None:
        document_repo.create(tmp_db, sample_course.id, Path("a"), "main", "h1", 1)
        document_repo.create(tmp_db, sample_course.id, Path("b"), "auxiliary", "h2", 1)

        assert len(document_repo.list_by_course(tmp_db, sample_course.id)) == 2
        mains = document_repo.list_by_course(tmp_db, sample_course.id, "main")
        assert len(mains) == 1
        assert mains[0].category == "main"

    def test_mark_indexed_flow(self, sample_course, tmp_db) -> None:
        """未索引 → mark → 已索引 → mark_all_unindexed → 又回到未索引。"""
        doc = document_repo.create(
            tmp_db, sample_course.id, Path("a"), "main", "h1", 1
        )
        assert len(document_repo.list_unindexed(tmp_db, sample_course.id)) == 1

        document_repo.mark_indexed(tmp_db, doc.id)
        assert document_repo.list_unindexed(tmp_db, sample_course.id) == []
        assert len(document_repo.get_indexed(tmp_db, sample_course.id)) == 1

        document_repo.mark_all_unindexed(tmp_db, sample_course.id)
        assert len(document_repo.list_unindexed(tmp_db, sample_course.id)) == 1

    def test_path_roundtrip(self, sample_course, tmp_db) -> None:
        """路径存进去是 str，取出来必须还是 Path。"""
        p = Path("data") / "courseware" / "x.pdf"
        doc = document_repo.create(tmp_db, sample_course.id, p, "main", "h1", 1)
        fetched = document_repo.get_by_id(tmp_db, doc.id)
        assert isinstance(fetched.path, Path)
        assert fetched.path == p
```

#### 2.7.10 `tests/unit/test_hashing.py`

```python
"""哈希工具测试。"""

from __future__ import annotations

from pathlib import Path

from private_teacher.utils.hashing import compute_sha256, compute_sha256_bytes


def test_same_content_same_hash(tmp_path: Path) -> None:
    """内容相同、文件名不同 → 哈希必须相同（这是去重能生效的前提）。"""
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("hello", encoding="utf-8")
    b.write_text("hello", encoding="utf-8")
    assert compute_sha256(a) == compute_sha256(b)


def test_different_content_different_hash(tmp_path: Path) -> None:
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("hello", encoding="utf-8")
    b.write_text("world", encoding="utf-8")
    assert compute_sha256(a) != compute_sha256(b)


def test_matches_known_value(tmp_path: Path) -> None:
    """对照 sha256 的标准值，确认实现没写错。

    空串的 sha256 是一个人人都能查到的常量，
    这种"黄金值测试"能保证你不是在测"自己的 bug 和自己的 bug 相等"。
    """
    f = tmp_path / "empty.txt"
    f.write_bytes(b"")
    expected = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    assert compute_sha256(f) == expected
    assert compute_sha256_bytes(b"") == expected


def test_large_file_chunked(tmp_path: Path) -> None:
    """超过一个 chunk（1MB）的文件也要算对。"""
    f = tmp_path / "big.bin"
    f.write_bytes(b"x" * (3 * 1024 * 1024))  # 3MB
    assert len(compute_sha256(f)) == 64
```

---

## 3. 加载器（Loaders）

### 3.1 目标

让用户上传 PDF/PPTX/TXT/MD/Code → 返回 `Document` 列表（带 page/line 元数据）。

### 3.2 概念小课堂

- **统一接口**：所有 loader 返回 `langchain_core.documents.Document`，里面是 `page_content` + `metadata`
- **metadata 标准化**：每个 loader 都填 `source_path`、`page`、`line_start`、`line_end`
- **懒加载 vs 全加载**：大 PDF 懒加载（边读边切），小 TXT 全加载

### 3.3 文件内容指引

#### `src/private_teacher/loaders/registry.py`

**核心结构**：

```python
from pathlib import Path
from langchain_core.documents import Document

# 扩展名 → loader 函数的映射
LOADERS: dict[str, Callable] = {
    ".pdf": load_pdf,
    ".pptx": load_pptx,
    ".txt": load_text,
    ".md": load_markdown,
    ".py": load_code,   # + .js .ts .java .go .rs 等
    ".cpp": load_code,
    ...
}

def get_loader(path: Path) -> Callable:
    """根据扩展名返回 loader。"""
    ext = path.suffix.lower()
    if ext not in LOADERS:
        raise DocumentLoadError(f"不支持的格式: {ext}", path=str(path))
    return LOADERS[ext]

def load(path: Path) -> list[Document]:
    """统一入口。"""
    loader = get_loader(path)
    return loader(path)
```

#### `src/private_teacher/loaders/pdf_loader.py`

**用 `pypdf` 或 `unstructured`**：

```python
from pypdf import PdfReader
from langchain_core.documents import Document

def load_pdf(path: Path) -> list[Document]:
    reader = PdfReader(str(path))
    docs = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        docs.append(Document(
            page_content=text,
            metadata={
                "source_path": str(path),
                "page": i,
                "total_pages": len(reader.pages),
            }
        ))
    return docs
```

**思考题**：
- 为什么每页一个 Document？→ 后面 source 跳转要精确到页
- `pypdf` 提取乱码怎么办？→ 用 `pdfplumber` 或 `unstructured` 备选

#### `src/private_teacher/loaders/text_loader.py`

**最简单，直接读文件**：

```python
def load_text(path: Path) -> list[Document]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return [Document(
        page_content=text,
        metadata={"source_path": str(path), "page": 1, "line_start": 1}
    )]
```

#### `src/private_teacher/loaders/code_loader.py`

**按"行号"切分，每 N 行一个 Document**：

```python
def load_code(path: Path, lines_per_chunk: int = 100) -> list[Document]:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    docs = []
    for i in range(0, len(lines), lines_per_chunk):
        chunk_lines = lines[i : i + lines_per_chunk]
        docs.append(Document(
            page_content="\n".join(chunk_lines),
            metadata={
                "source_path": str(path),
                "page": 1,  # 代码不分页
                "line_start": i + 1,
                "line_end": min(i + lines_per_chunk, len(lines)),
            }
        ))
    return docs
```

**关键**：保留行号 → 后面能精确跳到代码某一行。

#### `src/private_teacher/loaders/markdown_loader.py`

**比 text loader 多识别 frontmatter**：

```python
import re

def load_markdown(path: Path) -> list[Document]:
    text = path.read_text(encoding="utf-8", errors="replace")
    # 提取 YAML frontmatter（如果有）
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            text = parts[2].lstrip()
    return [Document(
        page_content=text,
        metadata={"source_path": str(path), "page": 1, "line_start": 1}
    )]
```

#### `src/private_teacher/loaders/pptx_loader.py`

**用 `python-pptx` 按 slide 拆分**：

```python
from pptx import Presentation

def load_pptx(path: Path) -> list[Document]:
    prs = Presentation(str(path))
    docs = []
    for i, slide in enumerate(prs.slides, start=1):
        text = "\n".join(
            shape.text for shape in slide.shapes if shape.has_text_frame
        )
        docs.append(Document(
            page_content=text,
            metadata={"source_path": str(path), "page": i}
        ))
    return docs
```

#### `src/private_teacher/loaders/directory_loader.py`

**递归扫整个目录**：

```python
def load_directory(dir_path: Path) -> list[Document]:
    all_docs = []
    for file in dir_path.rglob("*"):
        if file.is_file():
            try:
                all_docs.extend(load(file))
            except DocumentLoadError as e:
                logger.warning(f"跳过 {file}: {e}")
    return all_docs
```

### 3.4 测试代码

**tests/unit/test_loaders.py**

```python
class TestTextLoader:
    def test_loads_text_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello world", encoding="utf-8")
        docs = load(f)
        assert len(docs) == 1
        assert "hello" in docs[0].page_content
        assert docs[0].metadata["source_path"] == str(f)

class TestCodeLoader:
    def test_preserves_line_numbers(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("\n".join(f"line {i}" for i in range(1, 251)))
        docs = load(f, lines_per_chunk=100)
        assert len(docs) == 3  # 100 + 100 + 50
        assert docs[0].metadata["line_start"] == 1
        assert docs[0].metadata["line_end"] == 100
        assert docs[2].metadata["line_end"] == 250

class TestRegistry:
    def test_unsupported_format_raises(self, tmp_path):
        f = tmp_path / "test.exe"
        f.write_text("binary")
        with pytest.raises(DocumentLoadError):
            load(f)

    def test_directory_recursive(self, tmp_path):
        (tmp_path / "a.txt").write_text("A")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "b.txt").write_text("B")
        docs = load_directory(tmp_path)
        assert len(docs) == 2
```

**测试用样本文件**（`tests/fixtures/courseware/`）：
- `sample.txt`：纯文本
- `sample.md`：带 frontmatter 的 markdown
- `sample.py`：100+ 行 Python 代码
- `sample.pdf`：用 `reportlab` 或下载的简单 PDF

**生成样本脚本**（`scripts/make_sample_courseware.py`）：
- 写一段代码生成所有样本文件
- 跑一次就有永久测试数据

### 3.5 验证

```bash
uv run pytest tests/unit/test_loaders.py -v
```

### 3.6 踩坑

| 现象 | 原因 | 解决 |
| --- | --- | --- |
| PDF 提取出空字符串 | 扫描版 PDF（图） | 用 OCR（unstructured 的 ocr 模式） |
| 中文乱码 | 编码不对 | `errors="replace"` + 强制 utf-8 |
| 代码 chunk 跨函数 | lines_per_chunk 切碎 | Phase 1 接受，Phase 2 改 AST-aware |
| PPTX 表格丢字 | python-pptx 不支持 table cell | 遍历 `cell.text` 拼接 |

---

### 3.7 完整代码实现

> **本节的设计主线**：所有 loader 都返回 `list[Document]`，且 **metadata 的 key 完全统一**。
> 统一之后，下游的 splitter / vector_store / source_resolver 就不需要 `if isinstance(pdf)` 这种分支了。

#### 3.7.1 `src/private_teacher/loaders/base.py`（元数据契约）

```python
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
META_CHUNK_ID = "chunk_id"  # 由 splitter 填（见 §4）

# Chroma 允许的 metadata 值类型
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
            continue  # 关键：整个 key 都不要，而不是写 None
        if isinstance(value, _ALLOWED_META_TYPES):
            clean[key] = value
        else:
            clean[key] = str(value)
    return clean
```

#### 3.7.2 `src/private_teacher/loaders/text_loader.py`

```python
"""
纯文本加载器（.txt / .log / .csv 等）
=====================================

最简单的 loader，但有两个真实世界的坑必须处理：
  1. 编码：中文 Windows 上的 txt 大概率是 GBK，直接 utf-8 解码会炸
  2. 空文件：读出来是空串，不应该产生一个空 Document 污染向量库
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from loguru import logger

from private_teacher.loaders.base import (
    META_LINE_END,
    META_LINE_START,
    base_metadata,
)
from private_teacher.utils.exceptions import DocumentLoadError

# 尝试的编码顺序：先 utf-8（最通用），再中文 Windows 常见编码
# gb18030 是 gbk 的超集，能覆盖更多生僻字，所以放在 gbk 前面
_ENCODINGS = ("utf-8", "gb18030", "utf-16", "latin-1")


def read_text_smart(path: Path) -> str:
    """按多种编码依次尝试读取文本。

    为什么不直接 errors="replace"？
      - replace 会把无法解码的字节变成 "�"，中文文档会变成一片乱码
      - 先尝试正确的编码，全都失败了再用 replace 兜底，能救回绝大多数文件

    latin-1 永远不会失败（它把任意字节映射成字符），所以它是天然的兜底项。
    但它排在最后 —— 如果前面的编码能成功，就不会走到它。
    """
    for encoding in _ENCODINGS:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue  # 换下一种编码
        except OSError as exc:
            # 文件不存在 / 没权限 —— 换编码也没用，直接报错
            raise DocumentLoadError(
                f"读取文件失败: {exc}",
                path=str(path),
            ) from exc

    # 所有编码都失败（几乎不可能，因为 latin-1 不会失败）
    logger.warning(f"{path.name} 编码识别失败，改用 replace 模式")
    return path.read_text(encoding="utf-8", errors="replace")


def load_text(path: Path, **_kwargs: Any) -> list[Document]:
    """加载纯文本文件。

    Args:
        path: 文件路径
        **_kwargs: 吞掉调用方多传的参数（如 lines_per_chunk）。
            下划线开头表示"我知道有这个参数但故意不用"，ruff 不会报未使用告警。

    Returns:
        长度为 0 或 1 的列表 —— 空文件返回 []，不产生垃圾 Document

    为什么整个文件只产生 1 个 Document？
      - Document 是"逻辑单位"，切分是 splitter 的活（单一职责）
      - 纯文本没有天然的页边界，整体交给 splitter 按语义切更合理
    """
    text = read_text_smart(path)

    # strip() 后为空 → 说明是空文件或只有空白字符
    if not text.strip():
        logger.warning(f"跳过空文件：{path.name}")
        return []

    meta = base_metadata(path)
    # 文本文件天然有行号，全部记下来，方便 §9 跳转
    meta[META_LINE_START] = 1
    meta[META_LINE_END] = len(text.splitlines())

    return [Document(page_content=text, metadata=meta)]
```

#### 3.7.3 `src/private_teacher/loaders/markdown_loader.py`

```python
"""
Markdown 加载器
===============

比纯文本多做两件事：
  1. 剥掉 YAML frontmatter（--- 包裹的元信息），避免它污染语义检索
  2. 把 frontmatter 里的 title 提取到 metadata（UI 显示更友好）
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from private_teacher.loaders.base import (
    META_LINE_END,
    META_LINE_START,
    base_metadata,
)
from private_teacher.loaders.text_loader import read_text_smart

# 匹配文件开头的 YAML frontmatter：
#   \A        字符串开头（比 ^ 严格，^ 在 MULTILINE 下会匹配每行开头）
#   ---\s*\n  三个短横线 + 换行
#   (.*?)     非贪婪捕获中间内容
#   \n---\s*\n  结束的三个短横线
#   re.S      让 . 也能匹配换行符
_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.S)

# 从 frontmatter 里抠 title: xxx
_TITLE_RE = re.compile(r"^title:\s*(.+)$", re.M)


def load_markdown(path: Path, **_kwargs: Any) -> list[Document]:
    """加载 Markdown 文件。

    Returns:
        0 或 1 个 Document
    """
    raw = read_text_smart(path)

    title: str | None = None
    body = raw
    # frontmatter 占了多少行 —— 剥掉之后行号要相应偏移，否则跳转会错位
    offset = 0

    match = _FRONTMATTER_RE.match(raw)
    if match:
        frontmatter = match.group(1)
        # match.end() 是 frontmatter 结束位置（字符下标）
        # 数一下前面有多少个换行，就是偏移了多少行
        offset = raw[: match.end()].count("\n")
        body = raw[match.end() :]

        title_match = _TITLE_RE.search(frontmatter)
        if title_match:
            # strip 掉可能的引号：title: "机器学习" → 机器学习
            title = title_match.group(1).strip().strip("\"'")

    if not body.strip():
        return []

    meta = base_metadata(path)
    # +1 因为行号从 1 开始
    meta[META_LINE_START] = offset + 1
    meta[META_LINE_END] = offset + len(body.splitlines())
    if title:
        meta["title"] = title

    return [Document(page_content=body, metadata=meta)]
```

#### 3.7.4 `src/private_teacher/loaders/code_loader.py`

```python
"""
代码加载器
==========

代码和散文不一样：**行号就是它的坐标系**。
用户看到检索结果时会问"这在哪个文件第几行"，所以行号必须精确到 1 行不差。

Phase 1 用最朴素的"每 N 行一块"策略。
它的缺点很明显 —— 会把一个函数拦腰切断。
Phase 2 会换成 tree-sitter 的 AST 切分（按函数/类边界切）。
先用简单方案跑通全链路，是刻意的工程取舍，不是偷懒。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from private_teacher.loaders.base import (
    META_LINE_END,
    META_LINE_START,
    base_metadata,
)
from private_teacher.loaders.text_loader import read_text_smart

# 扩展名 → 语言名。用于 metadata，UI 上可以做语法高亮
LANGUAGE_BY_EXT: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".jsx": "javascript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".scala": "scala",
    ".sh": "bash",
    ".sql": "sql",
    ".r": "r",
    ".m": "matlab",
    ".jl": "julia",
}

DEFAULT_LINES_PER_CHUNK = 100


def load_code(
    path: Path,
    lines_per_chunk: int = DEFAULT_LINES_PER_CHUNK,
    **_kwargs: Any,
) -> list[Document]:
    """按固定行数切分代码文件。

    Args:
        path: 代码文件
        lines_per_chunk: 每块多少行。100 行 ≈ 3000 字符 ≈ 750 token，
            对 embedding 模型来说是安全的长度

    Returns:
        每 lines_per_chunk 行一个 Document

    Example:
        250 行的文件、lines_per_chunk=100 → 3 个 Document：
            [1, 100] [101, 200] [201, 250]
    """
    if lines_per_chunk < 1:
        raise ValueError(f"lines_per_chunk 必须 >= 1，收到 {lines_per_chunk}")

    text = read_text_smart(path)
    # splitlines() 会去掉行尾换行符，且能正确处理 \n / \r\n / \r 三种换行
    lines = text.splitlines()

    if not lines:
        return []

    language = LANGUAGE_BY_EXT.get(path.suffix.lower(), "text")
    docs: list[Document] = []

    # range(0, 250, 100) → 0, 100, 200
    for start_idx in range(0, len(lines), lines_per_chunk):
        # 切片自动处理越界：lines[200:300] 在只有 250 行时返回 50 行
        chunk_lines = lines[start_idx : start_idx + lines_per_chunk]

        # 整块都是空白行 → 跳过，别往向量库塞垃圾
        if not any(line.strip() for line in chunk_lines):
            continue

        meta = base_metadata(path)
        meta["language"] = language
        # ⚠️ 下标从 0 开始，行号从 1 开始，所以要 +1
        meta[META_LINE_START] = start_idx + 1
        # line_end 是「包含」的最后一行号
        meta[META_LINE_END] = min(start_idx + lines_per_chunk, len(lines))

        docs.append(
            Document(
                page_content="\n".join(chunk_lines),
                metadata=meta,
            )
        )

    return docs
```

#### 3.7.5 `src/private_teacher/loaders/pdf_loader.py`

```python
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
        except Exception as exc:  # noqa: BLE001 - 第三方库异常类型不稳定
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
    except Exception as exc:  # noqa: BLE001
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
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"pdfplumber 也失败了：{exc}")

    if total_chars == 0:
        # 不抛异常 —— 扫描版 PDF 是合法输入，只是我们暂时处理不了。
        # 返回空列表让上层"跳过这个文件"，比中断整批上传体验好得多
        logger.warning(
            f"{path.name}: 没有提取到任何文字，可能是扫描版 PDF（需要 OCR）"
        )
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
```

#### 3.7.6 `src/private_teacher/loaders/pptx_loader.py`

```python
"""
PPTX 加载器
===========

一张幻灯片 = 一个 Document（page = 幻灯片序号）。

python-pptx 的形状（shape）模型要点：
  - shape.has_text_frame 为 True 才有 .text（文本框、标题）
  - 表格是 shape.has_table，文字藏在 table.rows[i].cells[j].text 里，
    **不会**出现在 shape.text 中 —— 这是最常见的"PPT 内容丢了"的原因
  - 演讲者备注在 slide.notes_slide 里，也是有价值的内容
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from loguru import logger

from private_teacher.loaders.base import base_metadata
from private_teacher.utils.exceptions import DocumentLoadError


def _shape_text(shape: Any) -> str:
    """从单个 shape 里抠出所有文字（含表格、组合形状）。"""
    parts: list[str] = []

    # ---------- 普通文本框 ----------
    if getattr(shape, "has_text_frame", False):
        text = shape.text_frame.text.strip()
        if text:
            parts.append(text)

    # ---------- 表格 ----------
    if getattr(shape, "has_table", False):
        for row in shape.table.rows:
            # 一行拼成 "单元格1 | 单元格2"，保留表格的横向语义
            cells = [cell.text.strip() for cell in row.cells]
            line = " | ".join(c for c in cells if c)
            if line:
                parts.append(line)

    # ---------- 组合形状（递归） ----------
    # GROUP = 6，组合形状里还嵌套着子形状，不递归就会漏内容
    if getattr(shape, "shape_type", None) == 6:
        for sub in shape.shapes:
            sub_text = _shape_text(sub)
            if sub_text:
                parts.append(sub_text)

    return "\n".join(parts)


def load_pptx(
    path: Path,
    include_notes: bool = True,
    **_kwargs: Any,
) -> list[Document]:
    """加载 PPTX。

    Args:
        path: pptx 文件
        include_notes: 是否把演讲者备注也算进内容。
            默认 True —— 老师的备注往往比幻灯片正文信息量更大

    Returns:
        每张有内容的幻灯片一个 Document

    Raises:
        DocumentLoadError: 文件损坏 / 不是合法 pptx（比如老的 .ppt 格式）
    """
    try:
        from pptx import Presentation

        prs = Presentation(str(path))
    except Exception as exc:  # noqa: BLE001
        raise DocumentLoadError(
            f"PPTX 解析失败（注意：python-pptx 不支持旧版 .ppt）: {exc}",
            path=str(path),
        ) from exc

    slides = list(prs.slides)
    total = len(slides)
    docs: list[Document] = []

    for slide_no, slide in enumerate(slides, start=1):
        parts = [_shape_text(shape) for shape in slide.shapes]

        # ---------- 演讲者备注 ----------
        if include_notes and slide.has_notes_slide:
            notes = slide.notes_slide.notes_text_frame.text.strip()
            if notes:
                # 加个前缀，让 LLM 知道这段是备注而非正文
                parts.append(f"[演讲者备注] {notes}")

        content = "\n".join(p for p in parts if p).strip()
        if not content:
            continue  # 纯图片页

        docs.append(
            Document(
                page_content=content,
                metadata=base_metadata(path, page=slide_no, total_pages=total),
            )
        )

    logger.debug(f"{path.name}: 提取 {len(docs)}/{total} 张幻灯片")
    return docs
```

#### 3.7.7 `src/private_teacher/loaders/registry.py`

```python
"""
Loader 注册表
=============

整个 loaders 包对外的**唯一入口**：

    from private_teacher.loaders import load
    docs = load(Path("xxx.pdf"))

调用方永远不需要知道 pdf 用什么库解析。
以后要加 .docx 支持，只需要写一个 load_docx 再往 LOADERS 里注册一行。
这就是"注册表模式"（registry pattern）——用数据（dict）代替 if/elif 分支。
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from private_teacher.loaders.code_loader import LANGUAGE_BY_EXT, load_code
from private_teacher.loaders.markdown_loader import load_markdown
from private_teacher.loaders.pdf_loader import load_pdf
from private_teacher.loaders.pptx_loader import load_pptx
from private_teacher.loaders.text_loader import load_text
from private_teacher.utils.exceptions import DocumentLoadError

# loader 的统一签名：(path, **kwargs) -> list[Document]
# 用 Callable[..., ...] 是因为各 loader 的关键字参数不同（lines_per_chunk 等）
LoaderFn = Callable[..., list[Document]]


def _build_registry() -> dict[str, LoaderFn]:
    """构造扩展名 → loader 的映射表。

    写成函数而不是直接写字面量，是为了能用循环把 20 多种代码扩展名
    一次性注册进去，避免手写 20 行重复代码（也就少 20 个打错字的机会）。
    """
    registry: dict[str, LoaderFn] = {
        # ---------- 文档 ----------
        ".pdf": load_pdf,
        ".pptx": load_pptx,
        # ---------- 文本 ----------
        ".txt": load_text,
        ".log": load_text,
        ".csv": load_text,
        ".tsv": load_text,
        # ---------- Markdown ----------
        ".md": load_markdown,
        ".markdown": load_markdown,
        # ---------- 结构化文本（当代码处理，保留行号） ----------
        ".json": load_code,
        ".yaml": load_code,
        ".yml": load_code,
        ".toml": load_code,
        ".ini": load_code,
        ".xml": load_code,
        ".html": load_code,
    }

    # 所有编程语言扩展名统一交给 load_code
    for ext in LANGUAGE_BY_EXT:
        registry[ext] = load_code

    return registry


LOADERS: dict[str, LoaderFn] = _build_registry()

# 对外暴露"我们支持哪些格式"，Streamlit 的 file_uploader 要用（type= 参数）
SUPPORTED_EXTENSIONS: tuple[str, ...] = tuple(sorted(LOADERS))


def is_supported(path: Path) -> bool:
    """这个文件能不能加载。用于目录扫描时过滤。"""
    return path.suffix.lower() in LOADERS


def get_loader(path: Path) -> LoaderFn:
    """按扩展名取 loader。

    Raises:
        DocumentLoadError: 扩展名不在注册表里
    """
    ext = path.suffix.lower()
    loader = LOADERS.get(ext)
    if loader is None:
        raise DocumentLoadError(
            f"不支持的文件格式: {ext or '(无扩展名)'}",
            path=str(path),
            supported=list(SUPPORTED_EXTENSIONS),
        )
    return loader


def load(path: Path, **kwargs: Any) -> list[Document]:
    """加载任意支持的文件 —— **整个 loaders 包的统一入口**。

    Args:
        path: 文件路径
        **kwargs: 透传给具体 loader，如 lines_per_chunk=50、include_notes=False。
            不认识的参数会被 loader 的 **_kwargs 安静吞掉，所以这里很宽松。

    Returns:
        Document 列表（可能为空 —— 空文件 / 扫描版 PDF）

    Raises:
        DocumentLoadError: 文件不存在 / 格式不支持 / 解析失败
    """
    # 先检查存在性，报错信息比让 open() 抛 FileNotFoundError 清楚
    if not path.exists():
        raise DocumentLoadError("文件不存在", path=str(path))
    if not path.is_file():
        raise DocumentLoadError("不是文件（是目录？）", path=str(path))

    loader = get_loader(path)
    return loader(path, **kwargs)
```

#### 3.7.8 `src/private_teacher/loaders/directory_loader.py`

```python
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
    dir_path: Path,
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
            logger.warning(
                f"跳过大文件 {item.name}"
                f"（{item.stat().st_size / 1024 / 1024:.1f}MB）"
            )
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
        except Exception as exc:  # noqa: BLE001
            # 未预料的异常也不能让整批挂掉，但要用 exception 打完整堆栈，
            # 因为这说明我们的错误处理有漏洞，需要修
            logger.exception(f"加载 {file.name} 时发生未预期错误: {exc}")
            failed.append(file.name)

    logger.info(
        f"目录加载完成：{len(files)} 个文件 → {len(all_docs)} 个 Document"
        + (f"，{len(failed)} 个失败" if failed else "")
    )
    return all_docs
```

#### 3.7.9 `src/private_teacher/loaders/__init__.py`

```python
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
```

#### 3.7.10 `scripts/make_sample_courseware.py`（生成测试样本）

```python
"""
生成测试用课件样本
==================

跑一次就好：
    uv run python scripts/make_sample_courseware.py

产物在 tests/fixtures/courseware/，会被 git 跟踪，
这样 CI 环境不需要重新生成，测试永远有稳定的输入。

为什么要"生成"而不是手工放几个文件？
  - 生成脚本本身就是文档：一眼看出样本长什么样
  - 内容可控 —— 测试里可以断言"第 3 页一定包含'决策树'"
"""

from __future__ import annotations

from pathlib import Path

# 输出目录：脚本文件的上级目录 / tests/fixtures/courseware
# parents[1] = scripts/ 的父目录 = 项目根目录
OUT_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "courseware"


def make_txt() -> None:
    """纯文本样本。"""
    content = """机器学习基础

机器学习是人工智能的一个分支，研究如何让计算机从数据中自动学习规律。

监督学习使用带标签的数据训练模型，常见任务包括分类和回归。
无监督学习处理无标签数据，典型算法有 K-means 聚类和主成分分析。
强化学习通过与环境交互获得奖励信号来优化策略。
"""
    (OUT_DIR / "sample.txt").write_text(content, encoding="utf-8")


def make_markdown() -> None:
    """带 YAML frontmatter 的 Markdown 样本。"""
    content = """---
title: 决策树算法
author: 测试
tags: [ml, tree]
---

# 决策树

决策树是一种树形结构的分类模型，每个内部节点表示一个特征判断。

## 信息增益

ID3 算法使用信息增益选择划分特征，C4.5 改用信息增益率。

## 剪枝

预剪枝在生成过程中提前停止；后剪枝先长成完整树再回退。
"""
    (OUT_DIR / "sample.md").write_text(content, encoding="utf-8")


def make_python() -> None:
    """Python 代码样本，刻意写超过 100 行以便测试多 chunk 切分。"""
    header = '"""K 近邻分类器的教学实现。"""\n\nimport math\n\n\n'
    body = "\n".join(
        f"def helper_{i}(x: float) -> float:\n"
        f'    """第 {i} 个辅助函数。"""\n'
        f"    return x * {i}\n"
        for i in range(1, 41)  # 40 个函数 × 3 行 ≈ 160 行
    )
    (OUT_DIR / "sample.py").write_text(header + body, encoding="utf-8")


def make_pdf() -> None:
    """PDF 样本。

    优先用 reportlab 生成；没装就跳过（PDF 测试会自动 skip）。
    不把 reportlab 加进主依赖 —— 它只在生成样本时用一次。
        uv add --dev reportlab
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
    except ImportError:
        print("⚠️  未安装 reportlab，跳过 PDF 样本（uv add --dev reportlab）")
        return

    pdf_path = OUT_DIR / "sample.pdf"
    c = canvas.Canvas(str(pdf_path), pagesize=A4)
    pages = [
        "Chapter 1 Introduction to Machine Learning",
        "Chapter 2 Supervised Learning and Regression",
        "Chapter 3 Decision Tree and Random Forest",
    ]
    for text in pages:
        # 用英文：默认字体不含中文字形，写中文会变成黑方块
        c.setFont("Helvetica", 16)
        c.drawString(72, 750, text)
        c.showPage()  # 结束当前页，开始下一页
    c.save()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    make_txt()
    make_markdown()
    make_python()
    make_pdf()

    print(f"✅ 样本已生成于 {OUT_DIR}")
    for f in sorted(OUT_DIR.iterdir()):
        print(f"   - {f.name}  ({f.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
```

#### 3.7.11 `tests/unit/test_loaders.py`（完整版）

```python
"""loaders 层单元测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from private_teacher.loaders import (
    SUPPORTED_EXTENSIONS,
    is_supported,
    iter_loadable_files,
    load,
    load_directory,
)
from private_teacher.loaders.base import sanitize_metadata
from private_teacher.utils.exceptions import DocumentLoadError

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "courseware"


# ============================================================
# base.py
# ============================================================
class TestBase:
    def test_sanitize_drops_none(self) -> None:
        """None 值必须被整个 key 丢掉（Chroma 不接受 None）。"""
        clean = sanitize_metadata({"a": 1, "b": None, "c": "x"})
        assert clean == {"a": 1, "c": "x"}
        assert "b" not in clean

    def test_sanitize_converts_path(self) -> None:
        clean = sanitize_metadata({"p": Path("/a/b")})
        assert isinstance(clean["p"], str)

    def test_sanitize_keeps_primitives(self) -> None:
        meta = {"s": "x", "i": 1, "f": 1.5, "b": True}
        assert sanitize_metadata(meta) == meta


# ============================================================
# text_loader
# ============================================================
class TestTextLoader:
    def test_loads_text_file(self, tmp_path: Path) -> None:
        f = tmp_path / "test.txt"
        f.write_text("hello world", encoding="utf-8")

        docs = load(f)

        assert len(docs) == 1
        assert "hello" in docs[0].page_content
        # resolve() 后是绝对路径，所以和 str(f.resolve()) 比
        assert docs[0].metadata["source_path"] == str(f.resolve())
        assert docs[0].metadata["source_name"] == "test.txt"
        assert docs[0].metadata["file_type"] == "txt"
        assert docs[0].metadata["page"] == 1

    def test_empty_file_returns_empty_list(self, tmp_path: Path) -> None:
        """空文件不应该产生 Document。"""
        f = tmp_path / "empty.txt"
        f.write_text("   \n\n  ", encoding="utf-8")
        assert load(f) == []

    def test_gbk_encoding_fallback(self, tmp_path: Path) -> None:
        """GBK 编码的中文文件也要能正确读出来。"""
        f = tmp_path / "gbk.txt"
        f.write_bytes("机器学习".encode("gb18030"))

        docs = load(f)

        assert len(docs) == 1
        assert "机器学习" in docs[0].page_content


# ============================================================
# markdown_loader
# ============================================================
class TestMarkdownLoader:
    def test_strips_frontmatter(self, tmp_path: Path) -> None:
        f = tmp_path / "a.md"
        f.write_text(
            "---\ntitle: 我的标题\n---\n\n# 正文\n内容",
            encoding="utf-8",
        )

        docs = load(f)

        assert "title:" not in docs[0].page_content  # frontmatter 被剥掉
        assert "# 正文" in docs[0].page_content
        assert docs[0].metadata["title"] == "我的标题"

    def test_no_frontmatter_still_works(self, tmp_path: Path) -> None:
        f = tmp_path / "b.md"
        f.write_text("# 直接正文", encoding="utf-8")

        docs = load(f)

        assert "# 直接正文" in docs[0].page_content
        assert docs[0].metadata["line_start"] == 1

    def test_frontmatter_shifts_line_numbers(self, tmp_path: Path) -> None:
        """剥掉 frontmatter 后行号要相应偏移，不能还从 1 开始。"""
        f = tmp_path / "c.md"
        f.write_text("---\ntitle: T\n---\n正文", encoding="utf-8")

        docs = load(f)

        assert docs[0].metadata["line_start"] > 1


# ============================================================
# code_loader
# ============================================================
class TestCodeLoader:
    def test_preserves_line_numbers(self, tmp_path: Path) -> None:
        f = tmp_path / "test.py"
        f.write_text(
            "\n".join(f"line {i}" for i in range(1, 251)),
            encoding="utf-8",
        )

        docs = load(f, lines_per_chunk=100)

        assert len(docs) == 3  # 100 + 100 + 50
        assert docs[0].metadata["line_start"] == 1
        assert docs[0].metadata["line_end"] == 100
        assert docs[1].metadata["line_start"] == 101
        assert docs[2].metadata["line_end"] == 250

    def test_language_detected(self, tmp_path: Path) -> None:
        f = tmp_path / "a.py"
        f.write_text("x = 1", encoding="utf-8")
        assert load(f)[0].metadata["language"] == "python"

    def test_blank_chunks_skipped(self, tmp_path: Path) -> None:
        """整块空白的 chunk 不入库。"""
        f = tmp_path / "a.py"
        # 前 5 行有内容，接着 200 行空白
        f.write_text("x = 1\n" * 5 + "\n" * 200, encoding="utf-8")

        docs = load(f, lines_per_chunk=100)

        assert len(docs) == 1  # 只有第一块有内容

    def test_invalid_chunk_size_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "a.py"
        f.write_text("x = 1", encoding="utf-8")
        with pytest.raises(ValueError):
            load(f, lines_per_chunk=0)


# ============================================================
# registry
# ============================================================
class TestRegistry:
    def test_unsupported_format_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "test.exe"
        f.write_text("binary", encoding="utf-8")
        with pytest.raises(DocumentLoadError, match="不支持"):
            load(f)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(DocumentLoadError, match="不存在"):
            load(tmp_path / "nope.txt")

    def test_directory_path_raises(self, tmp_path: Path) -> None:
        with pytest.raises(DocumentLoadError, match="不是文件"):
            load(tmp_path)

    def test_extension_case_insensitive(self, tmp_path: Path) -> None:
        """.TXT 和 .txt 都要能认。"""
        f = tmp_path / "A.TXT"
        f.write_text("hi", encoding="utf-8")
        assert len(load(f)) == 1

    def test_is_supported(self, tmp_path: Path) -> None:
        assert is_supported(Path("a.pdf"))
        assert is_supported(Path("a.py"))
        assert not is_supported(Path("a.exe"))

    def test_supported_extensions_not_empty(self) -> None:
        assert ".pdf" in SUPPORTED_EXTENSIONS
        assert ".md" in SUPPORTED_EXTENSIONS


# ============================================================
# directory_loader
# ============================================================
class TestDirectoryLoader:
    def test_directory_recursive(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("A", encoding="utf-8")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "b.txt").write_text("B", encoding="utf-8")

        docs = load_directory(tmp_path)

        assert len(docs) == 2

    def test_ignores_junk_dirs(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("A", encoding="utf-8")
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "x.py").write_text("x=1", encoding="utf-8")
        (tmp_path / ".git").mkdir()
        (tmp_path / ".git" / "y.txt").write_text("y", encoding="utf-8")

        assert len(load_directory(tmp_path)) == 1

    def test_skips_unsupported_silently(self, tmp_path: Path) -> None:
        """不支持的格式在目录扫描时应该被安静跳过，不抛异常。"""
        (tmp_path / "a.txt").write_text("A", encoding="utf-8")
        (tmp_path / "b.exe").write_text("bin", encoding="utf-8")

        assert len(load_directory(tmp_path)) == 1

    def test_skips_oversized(self, tmp_path: Path) -> None:
        (tmp_path / "big.txt").write_text("x" * 5000, encoding="utf-8")
        assert load_directory(tmp_path, max_file_size=1000) == []

    def test_iter_returns_sorted(self, tmp_path: Path) -> None:
        """扫描结果必须有序，否则 chunk_id 每次都不一样。"""
        for name in ("c.txt", "a.txt", "b.txt"):
            (tmp_path / name).write_text("x", encoding="utf-8")

        files = iter_loadable_files(tmp_path)

        assert [f.name for f in files] == ["a.txt", "b.txt", "c.txt"]

    def test_missing_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises(DocumentLoadError):
            load_directory(tmp_path / "nope")


# ============================================================
# fixtures 真实文件（没生成样本就 skip）
# ============================================================
@pytest.mark.skipif(
    not (FIXTURES / "sample.pdf").exists(),
    reason="先跑 uv run python scripts/make_sample_courseware.py",
)
class TestRealFixtures:
    def test_pdf_pages(self) -> None:
        docs = load(FIXTURES / "sample.pdf")
        assert len(docs) == 3
        assert docs[0].metadata["page"] == 1
        assert docs[0].metadata["total_pages"] == 3
        assert "Machine Learning" in docs[0].page_content

    def test_python_multi_chunk(self) -> None:
        docs = load(FIXTURES / "sample.py", lines_per_chunk=50)
        assert len(docs) >= 2
        # 相邻 chunk 的行号必须连续，不能有洞
        for prev, curr in zip(docs, docs[1:], strict=False):
            assert curr.metadata["line_start"] == prev.metadata["line_end"] + 1
```

---

## 4. 切分器（Splitters）

### 4.1 目标

把 `Document` 切分成**适合 embedding 的小块**（chunk），每块保持语义完整。

### 4.2 概念小课堂

- **为什么切？**：Embedding 模型有 token 上限（512/8192），长文本需要切
- **chunk size**：每个 chunk 多少字符（默认 500-1000）
- **chunk overlap**：相邻 chunk 重复多少字符（默认 10-20%），防止边界切断语义
- **Recursive splitting**：按"段落 → 句子 → 词"递归切，优先保持大块

### 4.3 文件内容指引

#### `src/private_teacher/rag/splitters.py`

**核心结构**：

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class ChunkerConfig:
    chunk_size: int = 800       # 字符数
    chunk_overlap: int = 100     # 字符数

def split_documents(
    docs: list[Document],
    config: ChunkerConfig = ChunkerConfig(),
) -> list[Document]:
    """用 recursive splitter 切所有 Document。"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", " ", ""],  # 中英文都考虑
    )
    chunks = splitter.split_documents(docs)
    # 给每个 chunk 加 chunk_id
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"{chunk.metadata.get('source_path', '')}:{chunk.metadata.get('page', '')}:{i}"
    return chunks
```

**关键设计**：
- `chunk_id` = `source_path:page:序号` → 唯一标识
- 保留原始 metadata → 切分不丢信息

**思考题**：
- chunk_size 选 800 是为什么？→ 中文 1 字 ≈ 1 字符，800 字 ≈ 400 token，刚好
- overlap 为什么 100？→ 12.5% 是经验值，防止句子被切两半

### 4.4 测试代码

**tests/unit/test_splitters.py**

```python
class TestSplitDocuments:
    def test_short_doc_not_split(self):
        doc = Document(page_content="短文本", metadata={"page": 1})
        chunks = split_documents([doc], ChunkerConfig(chunk_size=800))
        assert len(chunks) == 1

    def test_long_doc_split_with_overlap(self):
        long_text = "句子。" * 500  # 1500 字符
        doc = Document(page_content=long_text, metadata={"page": 1})
        chunks = split_documents([doc], ChunkerConfig(chunk_size=800, chunk_overlap=100))
        assert len(chunks) >= 2

    def test_chunk_id_unique(self):
        doc = Document(page_content="x" * 2000, metadata={"page": 1})
        chunks = split_documents([doc])
        ids = [c.metadata["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids))  # 全部唯一

    def test_preserves_metadata(self):
        doc = Document(
            page_content="x" * 2000,
            metadata={"source_path": "/a.pdf", "page": 3, "line_start": 1}
        )
        chunks = split_documents([doc])
        for c in chunks:
            assert c.metadata["source_path"] == "/a.pdf"
            assert c.metadata["page"] == 3
```

### 4.5 验证

```bash
uv run pytest tests/unit/test_splitters.py -v
```

### 4.6 踩坑

| 现象 | 原因 | 解决 |
| --- | --- | --- |
| chunk 切断代码函数 | 按字符切不管语义 | Phase 2 改 AST-aware（tree-sitter） |
| 中文句子里被切断 | 默认 `separators` 是英文 | 加 `"。" "！" "？"` |
| overlap 太大导致重复 | overlap > chunk_size / 2 | 限制 overlap ≤ chunk_size / 2 |

---

### 4.7 完整代码实现

> **一个重要决定**：本项目**不引入 `langchain-text-splitters`**，而是自己实现递归切分器。
>
> 理由：
> 1. `langchain 1.x` 没有把 text-splitters 作为默认依赖（`uv pip list` 里根本没有它），要额外装；
> 2. 递归切分算法只有 60 行，自己写一遍你才真正明白 "chunk_size / overlap / separators" 三者如何相互作用；
> 3. 中文场景需要定制分隔符与"字符数 vs token 数"的取舍，自研更可控。
>
> 如果你更想用官方实现：`uv add langchain-text-splitters`，然后把 `split_documents`
> 内部换成 `RecursiveCharacterTextSplitter`，**外部接口保持不变**即可 —— 这正是把切分逻辑
> 包一层函数的价值。

#### 4.7.1 `src/private_teacher/rag/splitters.py`

```python
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
    min_chunk_size: int = 20

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
                f"chunk_overlap({self.chunk_overlap}) 必须小于 "
                f"chunk_size({self.chunk_size})"
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
            while buffer and (
                total > overlap or (total + len(atom) > chunk_size and total > 0)
            ):
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
    atoms = _atomize(text, config.separators, config.chunk_size)
    chunks = _merge_atoms(atoms, config.chunk_size, config.chunk_overlap)

    # 清洗：去掉首尾空白 + 丢弃太短的碎屑
    return [
        c.strip() for c in chunks if len(c.strip()) >= config.min_chunk_size
    ]


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
```

#### 4.7.2 `tests/unit/test_splitters.py`（完整版）

```python
"""splitters 单元测试。"""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from private_teacher.rag.splitters import (
    ChunkerConfig,
    estimate_tokens,
    split_documents,
    split_text,
)


# ============================================================
# 配置校验
# ============================================================
class TestChunkerConfig:
    def test_defaults(self) -> None:
        cfg = ChunkerConfig()
        assert cfg.chunk_size == 800
        assert cfg.chunk_overlap == 100

    def test_overlap_must_be_smaller(self) -> None:
        """overlap >= chunk_size 会导致合并死循环，必须提前拦住。"""
        with pytest.raises(ValueError, match="chunk_overlap"):
            ChunkerConfig(chunk_size=100, chunk_overlap=100)

    def test_invalid_size(self) -> None:
        with pytest.raises(ValueError, match="chunk_size"):
            ChunkerConfig(chunk_size=0)

    def test_is_frozen(self) -> None:
        """frozen dataclass 不允许修改字段。"""
        cfg = ChunkerConfig()
        with pytest.raises(Exception):  # FrozenInstanceError
            cfg.chunk_size = 999  # type: ignore[misc]


# ============================================================
# split_text
# ============================================================
class TestSplitText:
    def test_short_text_single_chunk(self) -> None:
        assert split_text("这是一段很短的文本内容，不需要切分。") == [
            "这是一段很短的文本内容，不需要切分。"
        ]

    def test_all_chunks_within_size(self) -> None:
        """核心不变量：任何 chunk 都不能超过 chunk_size。"""
        text = "机器学习是人工智能的分支。" * 200  # 2600 字符
        chunks = split_text(text, ChunkerConfig(chunk_size=300, chunk_overlap=50))

        assert len(chunks) > 1
        for c in chunks:
            assert len(c) <= 300, f"chunk 超长：{len(c)}"

    def test_no_content_lost(self) -> None:
        """切分不能丢内容：所有 chunk 拼起来应覆盖原文的每个字符。"""
        text = "".join(f"第{i}句话。" for i in range(1, 101))
        chunks = split_text(text, ChunkerConfig(chunk_size=200, chunk_overlap=0))

        # overlap=0 时直接拼接应等于原文（去掉切分时 strip 的空白）
        assert "".join(chunks) == text

    def test_overlap_creates_repetition(self) -> None:
        """overlap > 0 时，相邻 chunk 应该有重叠内容。"""
        text = "".join(f"句子{i}。" for i in range(1, 201))
        chunks = split_text(text, ChunkerConfig(chunk_size=200, chunk_overlap=60))

        assert len(chunks) >= 2
        # 拼起来的总长度应该大于原文（因为有重复部分）
        assert sum(len(c) for c in chunks) > len(text)

    def test_prefers_paragraph_boundary(self) -> None:
        """有段落分隔时，应优先在段落处切，而不是把段落腰斩。"""
        para = "内容" * 100  # 200 字符
        text = f"{para}\n\n{para}\n\n{para}"
        chunks = split_text(text, ChunkerConfig(chunk_size=250, chunk_overlap=0))

        # 每块应该正好是一个完整段落（250 装不下两个 200 的段落）
        assert len(chunks) == 3

    def test_chinese_sentence_separator(self) -> None:
        """没有换行的纯中文长句串，也要能按句号切开。"""
        text = "这是一个很长的句子用来测试中文切分效果。" * 50
        chunks = split_text(text, ChunkerConfig(chunk_size=100, chunk_overlap=0))

        for c in chunks:
            assert len(c) <= 100

    def test_no_separator_hard_cut(self) -> None:
        """一个分隔符都没有的超长串，靠字符硬切保底。"""
        text = "x" * 1000
        chunks = split_text(text, ChunkerConfig(chunk_size=100, chunk_overlap=0))

        assert len(chunks) == 10
        assert all(len(c) == 100 for c in chunks)

    def test_tiny_fragments_dropped(self) -> None:
        """低于 min_chunk_size 的碎屑要被丢掉。"""
        chunks = split_text("短", ChunkerConfig(min_chunk_size=20))
        assert chunks == []

    def test_empty_input(self) -> None:
        assert split_text("") == []


# ============================================================
# split_documents
# ============================================================
class TestSplitDocuments:
    def _doc(self, content: str, **meta) -> Document:
        base = {"source_path": "/tmp/a.txt", "page": 1}
        base.update(meta)
        return Document(page_content=content, metadata=base)

    def test_short_doc_not_split(self) -> None:
        docs = [self._doc("这是一段足够长但不超过上限的测试文本内容。")]
        assert len(split_documents(docs)) == 1

    def test_long_doc_split(self) -> None:
        docs = [self._doc("句子内容。" * 500)]
        chunks = split_documents(
            docs, ChunkerConfig(chunk_size=800, chunk_overlap=100)
        )
        assert len(chunks) >= 3

    def test_chunk_id_unique(self) -> None:
        docs = [self._doc("测试内容。" * 400)]
        chunks = split_documents(docs)
        ids = [c.metadata["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_id_stable_across_runs(self) -> None:
        """⭐ 最重要的测试：同样输入必须产出同样的 chunk_id。

        如果这个测试挂了，说明用了 hash() 或全局计数器，
        增量索引会因为 id 漂移而完全失效。
        """
        docs = [self._doc("稳定性测试内容。" * 300)]
        ids_1 = [c.metadata["chunk_id"] for c in split_documents(docs)]
        ids_2 = [c.metadata["chunk_id"] for c in split_documents(docs)]
        assert ids_1 == ids_2

    def test_chunk_id_isolated_per_page(self) -> None:
        """不同页的编号相互独立 —— 新增一页不该影响其他页的 id。"""
        p1 = self._doc("第一页内容。" * 300, page=1)
        p2 = self._doc("第二页内容。" * 300, page=2)

        only_p1 = [c.metadata["chunk_id"] for c in split_documents([p1])]
        both = [
            c.metadata["chunk_id"]
            for c in split_documents([p1, p2])
            if c.metadata["page"] == 1
        ]
        assert only_p1 == both

    def test_preserves_metadata(self) -> None:
        docs = [
            self._doc(
                "内容。" * 400,
                source_path="/a.pdf",
                page=3,
                line_start=1,
                file_type="pdf",
            )
        ]
        for c in split_documents(docs):
            assert c.metadata["source_path"] == "/a.pdf"
            assert c.metadata["page"] == 3
            assert c.metadata["file_type"] == "pdf"

    def test_metadata_not_shared(self) -> None:
        """⚠️ 每个 chunk 必须有独立的 metadata dict（浅拷贝陷阱）。"""
        docs = [self._doc("内容。" * 400)]
        chunks = split_documents(docs)

        assert len(chunks) >= 2
        # 改一个不能影响另一个
        chunks[0].metadata["page"] = 999
        assert chunks[1].metadata["page"] == 1

    def test_empty_docs(self) -> None:
        assert split_documents([]) == []


class TestEstimateTokens:
    def test_chinese(self) -> None:
        # 10 个汉字 ≈ 10 token
        assert estimate_tokens("机器学习深度学习算法") == 10

    def test_english(self) -> None:
        # 40 个 ASCII 字符 ≈ 10 token
        assert estimate_tokens("a" * 40) == 10
```

#### 4.7.3 `src/private_teacher/rag/__init__.py`（先建好，后面几节继续补）

```python
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
```

---

## 5. Embedding

### 5.1 目标

把每个 chunk 转成**向量**（一串数字），用于相似度搜索。

### 5.2 概念小课堂

- **Embedding 模型**：把文本映射到 N 维向量的模型（OpenAI 是 1536 维，本地 bge-m3 是 1024 维）
- **相似度**：两个向量越近 = 两段文本语义越像
- **三种调用方式**：OpenAI API / HuggingFace 本地 / Ollama

### 5.3 文件内容指引

#### `src/private_teacher/rag/embeddings.py`

**先在 `LLMSettings` 加字段**（在 `config.py`）：

```python
class LLMSettings(BaseSettings):
    # ... 已有字段 ...
    embedding_provider: Literal["openai", "huggingface", "ollama"] = "openai"
    embedding_model: str = "text-embedding-3-small"
    embedding_base_url: str | None = None  # 用 OpenAI 兼容协议
    embedding_dimension: int = 1536
```

**在 `.env.example` 加**：

```bash
PRIVATE_TEACHER_LLM_EMBEDDING_PROVIDER=openai
PRIVATE_TEACHER_LLM_EMBEDDING_MODEL=text-embedding-3-small
PRIVATE_TEACHER_LLM_EMBEDDING_DIMENSION=1536
```

**Embedder 工厂**：

```python
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
# 可选：HuggingFace 走 sentence-transformers

def get_embedder(settings: LLMSettings):
    match settings.embedding_provider:
        case "openai":
            return OpenAIEmbeddings(
                model=settings.embedding_model,
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                openai_api_base=settings.embedding_base_url,
            )
        case "ollama":
            return OllamaEmbeddings(
                model=settings.embedding_model,
                base_url=settings.ollama_base_url,
            )
```

### 5.4 测试代码

**tests/unit/test_embeddings.py**

```python
class TestEmbedderFactory:
    def test_openai_requires_key(self, monkeypatch):
        from private_teacher.config import LLMSettings
        settings = LLMSettings(embedding_provider="openai")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ConfigurationError):
            get_embedder(settings)
```

> 大部分 embedding 测试需要真 API，用 `@pytest.mark.requires_llm` 标记。

### 5.5 验证

```bash
uv run pytest tests/unit/test_embeddings.py -v
```

### 5.6 完整代码实现

#### 5.6.1 `src/private_teacher/config.py` 需要新增的字段

```python
# 在 LLMSettings 类里，追加以下字段
# （放在 "----- Ollma 配置 -----" 之后、model_config 之前）

    # ----- Embedding 配置 -----
    # 注意：embedding 的 provider 和 chat 的 provider 是**独立**的。
    # 常见组合：用 Claude 做对话（质量高），用本地 bge 做 embedding（省钱且快）。
    embedding_provider: Literal["openai", "ollama", "huggingface", "hash"] = Field(
        default="openai",
        description="Embedding 提供方：openai / ollama / huggingface / hash（离线测试用）",
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="Embedding 模型名。openai: text-embedding-3-small；"
        "ollama: nomic-embed-text；huggingface: BAAI/bge-small-zh-v1.5",
    )
    embedding_base_url: str | None = Field(
        default=None,
        description="Embedding 服务地址（走中转 API 时填）。留空用官方地址",
    )
    embedding_dimension: int = Field(
        default=1536,
        description="向量维度。⚠️ 必须与模型实际输出一致，否则 Chroma 写入报错。"
        "text-embedding-3-small=1536, bge-small-zh=512, nomic-embed-text=768",
    )
    embedding_batch_size: int = Field(
        default=64,
        description="每批送多少条文本去向量化。太大容易超 API 的单请求上限，"
        "太小则请求次数多、速度慢",
    )
```

**`.env.example` 追加**：

```bash
# ---------- Embedding ----------
# 省钱方案：用 Ollama 本地 embedding（先 ollama pull nomic-embed-text）
#   PRIVATE_TEACHER_LLM_EMBEDDING_PROVIDER=ollama
#   PRIVATE_TEACHER_LLM_EMBEDDING_MODEL=nomic-embed-text
#   PRIVATE_TEACHER_LLM_EMBEDDING_DIMENSION=768
PRIVATE_TEACHER_LLM_EMBEDDING_PROVIDER=openai
PRIVATE_TEACHER_LLM_EMBEDDING_MODEL=text-embedding-3-small
PRIVATE_TEACHER_LLM_EMBEDDING_DIMENSION=1536
PRIVATE_TEACHER_LLM_EMBEDDING_BATCH_SIZE=64
# 用中转 API 时填，例如 https://your-proxy.com/v1
# PRIVATE_TEACHER_LLM_EMBEDDING_BASE_URL=
```

#### 5.6.2 `src/private_teacher/rag/embeddings.py`

```python
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
from typing import Protocol, runtime_checkable

from loguru import logger

from private_teacher.config import LLMSettings
from private_teacher.utils.exceptions import ConfigurationError, RAGError


# ============================================================
# 接口定义
# ============================================================
@runtime_checkable
class Embedder(Protocol):
    """Embedding 模型的最小接口。

    用 Protocol（结构化类型/鸭子类型）而不是抽象基类 ABC：
      - langchain 的 OpenAIEmbeddings 不可能继承我们的基类
      - Protocol 只要求"长得像"就行，不要求继承关系
      - runtime_checkable 让 isinstance(x, Embedder) 也能用（只检查方法名）
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
        # bigram：zip(cjk, cjk[1:]) 把 [a,b,c] 配成 [(a,b),(b,c)]
        tokens.extend(a + b for a, b in zip(cjk, cjk[1:], strict=False))

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
        logger.info(f"加载本地 embedding 模型：{model_name}（首次会下载）")
        self._model = SentenceTransformer(model_name)

        # bge 系列要求给查询加指令前缀才能发挥最佳效果
        self._query_prefix = (
            "为这个句子生成表示以用于检索相关文章：" if "bge" in model_name.lower() else ""
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # normalize_embeddings=True 让输出已经是单位向量，
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
            logger.warning(
                "使用 HashEmbedder —— 它不理解语义，仅供测试/离线 demo 使用"
            )
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
        except Exception as exc:  # noqa: BLE001
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
```

#### 5.6.3 `tests/conftest.py` 追加 embedder fixture

```python
# ============================================================
# Phase 1：Embedding fixture
# ============================================================
@pytest.fixture
def fake_embedder():
    """确定性的假 embedder（HashEmbedder，256 维）。

    用 256 维而不是 1536 维：
      - 测试里只关心"相似的文本向量更接近"，维度不影响这个性质
      - 维度小 6 倍，测试快 6 倍
    """
    from private_teacher.rag.embeddings import HashEmbedder

    return HashEmbedder(dimension=256)


@pytest.fixture
def hash_llm_settings():
    """把 embedding provider 强制设成 hash 的 LLMSettings。

    有了它，Indexer / Retriever / KBService 的测试都能完全离线跑，
    不需要任何 API key。
    """
    from private_teacher.config import LLMSettings

    return LLMSettings(
        embedding_provider="hash",
        embedding_dimension=256,
    )
```

#### 5.6.4 `tests/unit/test_embeddings.py`（完整版）

```python
"""embeddings 单元测试。

注意：所有测试都不需要网络和 API key —— 真实 provider 的测试用
@pytest.mark.requires_llm 标记，默认跳过。
"""

from __future__ import annotations

import math

import pytest

from private_teacher.config import LLMSettings
from private_teacher.rag.embeddings import (
    Embedder,
    HashEmbedder,
    clear_embedder_cache,
    embed_in_batches,
    get_cached_embedder,
    get_embedder,
)
from private_teacher.utils.exceptions import ConfigurationError, RAGError


def _cosine(a: list[float], b: list[float]) -> float:
    """余弦相似度。两个向量都归一化过时，它等价于点积。"""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


# ============================================================
# HashEmbedder
# ============================================================
class TestHashEmbedder:
    def test_dimension(self) -> None:
        e = HashEmbedder(dimension=128)
        assert len(e.embed_query("测试")) == 128

    def test_deterministic(self) -> None:
        """⭐ 同样输入必须永远同样输出。"""
        e = HashEmbedder(dimension=64)
        assert e.embed_query("机器学习") == e.embed_query("机器学习")

    def test_normalized(self) -> None:
        """输出应是单位向量（L2 范数为 1）。"""
        vec = HashEmbedder(dimension=64).embed_query("深度学习很有趣")
        norm = math.sqrt(sum(v * v for v in vec))
        assert norm == pytest.approx(1.0)

    def test_empty_text_no_nan(self) -> None:
        """空文本不能产生零向量（会让余弦相似度变 NaN）。"""
        vec = HashEmbedder(dimension=64).embed_query("")
        assert math.sqrt(sum(v * v for v in vec)) == pytest.approx(1.0)

    def test_similar_texts_closer(self) -> None:
        """⭐ 关键能力：相关文本的相似度应高于无关文本。

        这个测试保证 HashEmbedder 能支撑起后面 retriever 的测试断言。
        """
        e = HashEmbedder(dimension=512)
        query = e.embed_query("机器学习")
        related = e.embed_query("机器学习是人工智能的一个分支")
        unrelated = e.embed_query("今天天气不错适合出门散步")

        assert _cosine(query, related) > _cosine(query, unrelated)

    def test_identical_text_similarity_is_one(self) -> None:
        e = HashEmbedder(dimension=256)
        v = e.embed_query("完全相同的文本")
        assert _cosine(v, v) == pytest.approx(1.0)

    def test_embed_documents_batch(self) -> None:
        e = HashEmbedder(dimension=64)
        vecs = e.embed_documents(["a", "b", "c"])
        assert len(vecs) == 3
        assert all(len(v) == 64 for v in vecs)

    def test_satisfies_protocol(self) -> None:
        """结构化类型检查：HashEmbedder 应满足 Embedder 协议。"""
        assert isinstance(HashEmbedder(), Embedder)


# ============================================================
# 工厂
# ============================================================
class TestEmbedderFactory:
    def test_hash_provider(self) -> None:
        settings = LLMSettings(embedding_provider="hash", embedding_dimension=128)
        embedder = get_embedder(settings)
        assert isinstance(embedder, HashEmbedder)
        assert len(embedder.embed_query("x")) == 128

    def test_openai_requires_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """没有 OPENAI_API_KEY 时必须立刻报错，而不是等到调 API 才失败。"""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        settings = LLMSettings(embedding_provider="openai")

        with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
            get_embedder(settings)

    def test_openai_with_key_builds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """有 key 就能构造成功（不发起真实请求）。"""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake")
        settings = LLMSettings(embedding_provider="openai")

        embedder = get_embedder(settings)

        assert hasattr(embedder, "embed_query")

    def test_unknown_provider(self) -> None:
        """构造一个非法 provider（绕过 pydantic 校验来测兜底分支）。"""
        settings = LLMSettings(embedding_provider="hash")
        object.__setattr__(settings, "embedding_provider", "nope")

        with pytest.raises(ConfigurationError, match="未知"):
            get_embedder(settings)


class TestEmbedderCache:
    def test_same_config_same_instance(self) -> None:
        clear_embedder_cache()
        settings = LLMSettings(embedding_provider="hash", embedding_dimension=64)

        # is 比较的是对象身份，不是值 —— 确认真的复用了同一个实例
        assert get_cached_embedder(settings) is get_cached_embedder(settings)

    def test_different_config_different_instance(self) -> None:
        clear_embedder_cache()
        a = get_cached_embedder(
            LLMSettings(embedding_provider="hash", embedding_dimension=64)
        )
        b = get_cached_embedder(
            LLMSettings(embedding_provider="hash", embedding_dimension=128)
        )
        assert a is not b


# ============================================================
# 批处理
# ============================================================
class TestEmbedInBatches:
    def test_returns_all_vectors(self, fake_embedder) -> None:
        texts = [f"文本{i}" for i in range(10)]
        vecs = embed_in_batches(fake_embedder, texts, batch_size=3)
        assert len(vecs) == 10

    def test_empty_input(self, fake_embedder) -> None:
        assert embed_in_batches(fake_embedder, []) == []

    def test_progress_callback(self, fake_embedder) -> None:
        seen: list[tuple[int, int]] = []
        embed_in_batches(
            fake_embedder,
            [f"t{i}" for i in range(10)],
            batch_size=4,
            on_progress=lambda done, total: seen.append((done, total)),
        )
        # 10 条 / 每批 4 条 = 3 批
        assert seen == [(4, 10), (8, 10), (10, 10)]

    def test_wraps_error_as_ragerror(self) -> None:
        """底层异常必须被包装成 RAGError，且带上批次信息。"""

        class BrokenEmbedder:
            def embed_documents(self, texts):
                raise RuntimeError("模拟 API 挂了")

            def embed_query(self, text):
                return [0.0]

        with pytest.raises(RAGError, match="向量化失败"):
            embed_in_batches(BrokenEmbedder(), ["a", "b"])


# ============================================================
# 真实 provider（默认跳过）
# ============================================================
@pytest.mark.requires_llm
class TestRealProviders:
    def test_openai_embedding(self) -> None:
        """需要真实 key：uv run pytest -m requires_llm"""
        settings = LLMSettings(embedding_provider="openai")
        vec = get_embedder(settings).embed_query("hello")
        assert len(vec) == settings.embedding_dimension
```

> **怎么跑需要真实 API 的测试**：`uv run pytest -m requires_llm`
> 默认的 `make test` 不带 `-m`，所以这些测试会被收集但不会被跳过 ——
> 如果你希望默认跳过，在 `conftest.py` 加一个 `pytest_collection_modifyitems` 钩子，
> 检测到没有对应环境变量就 `item.add_marker(pytest.mark.skip(...))`。

---

## 6. 向量库（Chroma）

### 6.1 目标

存 chunks 的向量 + metadata，支持**相似度搜索**。

### 6.2 概念小课堂

- **Chroma**：轻量级本地向量库，零运维，文件存储
- **Collection**：一个 collection = 一组相关向量（我们用 `course_{course_id}`）
- **持久化**：写到磁盘，重启后还在
- **metadata filter**：搜索时按 metadata 过滤（如只搜某门课）

### 6.3 文件内容指引

#### `src/private_teacher/rag/vector_store.py`

**核心结构**：

```python
import chromadb
from langchain_chroma import Chroma
from pathlib import Path
from private_teacher.config import PathSettings

class VectorStoreManager:
    """每门课一个 Chroma collection。"""

    def __init__(self, course_id: str, data_dir: Path):
        self.course_id = course_id
        self.persist_dir = data_dir / "knowledge_bases" / course_id / "chroma"
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection_name = f"course_{course_id}"

    def get_store(self, embedder) -> Chroma:
        return Chroma(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=embedder,
        )

    def add_documents(self, embedder, chunks: list[Document]) -> list[str]:
        """加 chunks，返回 chunk_id 列表。"""
        store = self.get_store(embedder)
        return store.add_documents(chunks)

    def similarity_search(self, embedder, query: str, k: int = 5):
        store = self.get_store(embedder)
        return store.similarity_search(query, k=k)

    def delete_collection(self):
        self.client.delete_collection(self.collection_name)
```

### 6.4 测试代码

**tests/unit/test_vector_store.py**

```python
class TestVectorStoreManager:
    def test_create_and_search(self, tmp_path, fake_embedder):
        mgr = VectorStoreManager("c1", tmp_path)
        chunks = [
            Document(page_content="Python 是一门编程语言", metadata={"page": 1}),
            Document(page_content="Java 也用于后端", metadata={"page": 1}),
        ]
        ids = mgr.add_documents(fake_embedder, chunks)
        assert len(ids) == 2

        results = mgr.similarity_search(fake_embedder, "编程语言", k=1)
        assert len(results) == 1
        assert "Python" in results[0].page_content
```

**fake_embedder**（conftest.py）：
```python
class FakeEmbedder:
    """假 embedder：把文本转成固定维度的随机向量。"""
    def embed_documents(self, texts):
        return [[hash(t) % 100 / 100.0] * 1536 for t in texts]
    def embed_query(self, text):
        return [hash(text) % 100 / 100.0] * 1536
```

### 6.5 验证

```bash
uv run pytest tests/unit/test_vector_store.py -v
```

### 6.6 完整代码实现

> 这里用 **chromadb 原生 API**（理由见 §0.2 的补充说明）。
> 你会看到向量库真正的四个动作：`upsert` / `query` / `delete` / `count`。

#### 6.6.1 `src/private_teacher/rag/vector_store.py`

```python
"""
向量库封装（Chroma）
====================

【一个 collection = 一门课】
collection 名字用 "course_{course_id}"，每门课的向量物理隔离在自己的目录里。
好处：删课程 = 删一个目录；一门课的索引损坏不会影响其他课。

【为什么用 upsert 而不是 add】
add：id 已存在会抛异常
upsert：id 已存在则覆盖
我们的 chunk_id 是内容稳定的（见 §4），所以重复索引同一个文件时，
upsert 会原地更新而不是产生重复向量 —— 这是增量索引能正确工作的基础。

【距离与相似度】
Chroma 默认用 L2（欧氏距离），我们显式改成 cosine（余弦距离）：
  - 文本 embedding 的行业惯例就是余弦相似度
  - 余弦距离范围 [0, 2]，相似度 = 1 - 距离，范围 [-1, 1]，直观好解释
  - ⚠️ space 是**建 collection 时**定死的，之后改不了。改了就得重建库。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from loguru import logger

from private_teacher.loaders.base import META_CHUNK_ID, META_SOURCE_PATH, sanitize_metadata
from private_teacher.rag.embeddings import Embedder, embed_in_batches
from private_teacher.utils.exceptions import RAGError


class VectorStoreManager:
    """管理某一门课的 Chroma collection。

    典型用法：
        mgr = VectorStoreManager("a1b2c3d4", Path("data"))
        mgr.add_documents(embedder, chunks)
        hits = mgr.similarity_search(embedder, "机器学习", k=5)
    """

    def __init__(self, course_id: str, data_dir: Path) -> None:
        """
        Args:
            course_id: 课程 id
            data_dir: 数据根目录（PathSettings.data_dir）
        """
        self.course_id = course_id
        self.persist_dir = data_dir / "knowledge_bases" / course_id / "chroma"
        # PersistentClient 要求目录存在
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.collection_name = f"course_{course_id}"
        # 延迟初始化：只有真正用到时才创建 client，
        # 因为 chromadb 首次 import + 建 client 有几百毫秒开销
        self._client: Any = None

    # ============================================================
    # 底层 client / collection
    # ============================================================
    @property
    def client(self) -> Any:
        """惰性创建 Chroma 持久化客户端。"""
        if self._client is None:
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            self._client = chromadb.PersistentClient(
                path=str(self.persist_dir),
                settings=ChromaSettings(
                    # 关掉匿名遥测 —— 本项目是 local-first，不该往外发任何数据
                    anonymized_telemetry=False,
                    # 允许 reset()，测试里要用
                    allow_reset=True,
                ),
            )
        return self._client

    def get_collection(self) -> Any:
        """取（或创建）collection。

        get_or_create 是幂等的：存在就返回，不存在就建。
        比 try get / except create 干净。
        """
        return self.client.get_or_create_collection(
            name=self.collection_name,
            # ⚠️ hnsw:space 只在**创建时**生效。
            #    如果 collection 已存在且是 l2，这里传 cosine 也不会改变它。
            #    换距离度量的唯一办法是删掉重建。
            metadata={"hnsw:space": "cosine"},
        )

    # ============================================================
    # 写入
    # ============================================================
    def add_documents(
        self,
        embedder: Embedder,
        chunks: list[Document],
        batch_size: int = 64,
        on_progress: Any = None,
    ) -> list[str]:
        """把 chunk 写入向量库（upsert 语义）。

        Args:
            embedder: 向量化器
            chunks: split_documents 的产物（metadata 里必须有 chunk_id）
            batch_size: 向量化批大小
            on_progress: 进度回调 (done, total)

        Returns:
            实际写入的 id 列表

        Raises:
            RAGError: 缺少 chunk_id / 向量化失败 / 写入失败
        """
        if not chunks:
            return []

        # ---------- 1. 取出三份平行数组 ----------
        ids: list[str] = []
        texts: list[str] = []
        metadatas: list[dict[str, Any]] = []

        for chunk in chunks:
            chunk_id = chunk.metadata.get(META_CHUNK_ID)
            if not chunk_id:
                # 没有 chunk_id 说明没走 split_documents，是调用方的 bug。
                # 与其静默生成随机 id（后患无穷），不如立刻报错
                raise RAGError(
                    "chunk 缺少 chunk_id，请先经过 split_documents 处理",
                    source=chunk.metadata.get(META_SOURCE_PATH, "?"),
                )

            ids.append(str(chunk_id))
            texts.append(chunk.page_content)
            # sanitize：把 None / Path 等 Chroma 不认的值清洗掉（见 §3.7.1）
            metadatas.append(sanitize_metadata(chunk.metadata))

        # ---------- 2. 去重 ----------
        # 同一批里出现重复 id，Chroma 会报错。
        # 正常不该发生，但如果调用方把同一个文件加载了两次就会 —— 这里兜一下底
        seen: set[str] = set()
        keep: list[int] = []
        for i, cid in enumerate(ids):
            if cid not in seen:
                seen.add(cid)
                keep.append(i)
        if len(keep) != len(ids):
            logger.warning(f"发现 {len(ids) - len(keep)} 个重复 chunk_id，已去重")
            ids = [ids[i] for i in keep]
            texts = [texts[i] for i in keep]
            metadatas = [metadatas[i] for i in keep]

        # ---------- 3. 向量化 ----------
        vectors = embed_in_batches(embedder, texts, batch_size, on_progress)

        # ---------- 4. 写入 ----------
        collection = self.get_collection()
        try:
            # upsert：id 已存在则覆盖，不存在则插入
            collection.upsert(
                ids=ids,
                documents=texts,
                metadatas=metadatas,
                embeddings=vectors,
            )
        except Exception as exc:  # noqa: BLE001
            raise RAGError(
                f"写入向量库失败: {exc}",
                course_id=self.course_id,
                count=len(ids),
            ) from exc

        logger.info(f"向量库写入 {len(ids)} 条（course={self.course_id}）")
        return ids

    # ============================================================
    # 检索
    # ============================================================
    def query(
        self,
        embedder: Embedder,
        query_text: str,
        k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[tuple[Document, float]]:
        """相似度检索，返回 (Document, 相似度) 列表。

        Args:
            embedder: 向量化器（用它的 embed_query，不是 embed_documents！）
            query_text: 查询语句
            k: 返回条数
            where: metadata 过滤条件，例如 {"file_type": "pdf"}。
                多条件要用 Chroma 的语法：
                    {"$and": [{"file_type": "pdf"}, {"page": {"$gte": 10}}]}

        Returns:
            [(Document, similarity), ...]，按相似度从高到低。
            similarity = 1 - cosine_distance，范围约 [-1, 1]，越大越像。

        为什么返回相似度而不只是 Document？
          - UI 上要显示"匹配度 87%"
          - Phase 6 的 rerank / 阈值过滤需要分数
        """
        collection = self.get_collection()

        # 空库直接返回，否则 Chroma 在某些版本会报错
        if collection.count() == 0:
            logger.warning(f"知识库为空（course={self.course_id}）")
            return []

        query_vector = embedder.embed_query(query_text)

        try:
            raw = collection.query(
                query_embeddings=[query_vector],
                # n_results 超过库里实际条数时 Chroma 会自动截断，不会报错
                n_results=k,
                where=where,
                # include 决定返回哪些字段。
                # 不写 "embeddings" —— 向量很大，传回来纯属浪费内存
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:  # noqa: BLE001
            raise RAGError(
                f"向量检索失败: {exc}",
                course_id=self.course_id,
                query=query_text[:50],
            ) from exc

        # ---------- 解析返回结构 ----------
        # Chroma 支持一次查多个 query，所以每个字段都是"二维"的：
        #   raw["documents"] = [[第1个query的结果...]]
        # 我们只查了一个，取 [0]
        docs_list = raw.get("documents") or [[]]
        metas_list = raw.get("metadatas") or [[]]
        dists_list = raw.get("distances") or [[]]

        results: list[tuple[Document, float]] = []
        for text, meta, dist in zip(
            docs_list[0], metas_list[0], dists_list[0], strict=False
        ):
            # 余弦距离 → 相似度
            similarity = 1.0 - float(dist)
            results.append(
                (
                    Document(page_content=text, metadata=dict(meta or {})),
                    similarity,
                )
            )

        return results

    def similarity_search(
        self,
        embedder: Embedder,
        query_text: str,
        k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[Document]:
        """只要 Document、不要分数的简化版（分数会塞进 metadata["score"]）。"""
        results = self.query(embedder, query_text, k, where)
        docs: list[Document] = []
        for doc, score in results:
            # round 到 4 位：分数是给人看的，不需要 17 位小数
            doc.metadata["score"] = round(score, 4)
            docs.append(doc)
        return docs

    # ============================================================
    # 删除与统计
    # ============================================================
    def count(self) -> int:
        """库里有多少条向量。UI 显示"知识库规模"用。"""
        try:
            return int(self.get_collection().count())
        except Exception:  # noqa: BLE001
            # collection 还没建 / 目录被手删了 —— 都当作 0
            return 0

    def delete_by_source(self, source_path: str | Path) -> int:
        """删除某个源文件产生的所有向量。

        场景：用户删了一个课件，它的向量必须同步清掉，
        否则检索还会返回"已经不存在的文件"的内容。

        Returns:
            删除前该文件的向量条数（Chroma 的 delete 不返回删除数量，
            所以先查后删）
        """
        collection = self.get_collection()
        where = {META_SOURCE_PATH: str(Path(source_path).resolve())}

        # get 只查不返回向量，加 limit 也行；这里只要数量
        existing = collection.get(where=where, include=[])
        n = len(existing.get("ids", []))

        if n:
            collection.delete(where=where)
            logger.info(f"删除 {n} 条向量（source={Path(source_path).name}）")

        return n

    def delete_collection(self) -> None:
        """删除整个 collection（全量重建前调用）。

        ⚠️ collection 不存在时，Chroma 会抛异常。
           这里吞掉它 —— "删一个不存在的东西"应当是幂等成功，而不是失败。
        """
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"删除 collection: {self.collection_name}")
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"collection 不存在或已删除：{exc}")

    def reset(self) -> None:
        """删除并立刻重建一个空 collection。"""
        self.delete_collection()
        self.get_collection()

    # ============================================================
    # 调试辅助
    # ============================================================
    def peek(self, n: int = 5) -> list[Document]:
        """随便看几条，确认库里到底存了什么。

        排查"检索不到"的问题时，第一件事就是 peek 一下 ——
        很多时候会发现库根本是空的，或者存的是乱码。
        """
        collection = self.get_collection()
        raw = collection.peek(limit=n)

        docs: list[Document] = []
        for text, meta in zip(
            raw.get("documents") or [],
            raw.get("metadatas") or [],
            strict=False,
        ):
            docs.append(Document(page_content=text, metadata=dict(meta or {})))
        return docs

    def list_sources(self) -> list[str]:
        """列出库里所有不同的源文件路径。

        用途：校验"数据库里登记的文档"和"向量库里实际有的文档"是否一致。
        """
        collection = self.get_collection()
        # include=["metadatas"] 只取元数据，不取正文和向量，省内存
        raw = collection.get(include=["metadatas"])

        sources = {
            str(m.get(META_SOURCE_PATH, ""))
            for m in (raw.get("metadatas") or [])
            if m
        }
        return sorted(s for s in sources if s)
```

#### 6.6.2 `tests/unit/test_vector_store.py`（完整版）

```python
"""vector_store 单元测试。

这些测试会真的建一个 Chroma 库（在 tmp_path 里），跑完自动删除。
不 mock Chroma 的原因：Chroma 的 API 细节（返回结构、距离定义）
正是最容易出错的地方，mock 掉就等于什么都没测。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.documents import Document

from private_teacher.rag.splitters import split_documents
from private_teacher.rag.vector_store import VectorStoreManager
from private_teacher.utils.exceptions import RAGError


def make_chunks(pairs: list[tuple[str, str]]) -> list[Document]:
    """辅助函数：从 [(源文件, 内容)] 造出带 chunk_id 的 chunk。

    走真正的 split_documents，保证 chunk_id 的生成逻辑和生产环境一致。
    """
    docs = [
        Document(
            page_content=content,
            metadata={
                "source_path": src,
                "source_name": Path(src).name,
                "file_type": Path(src).suffix.lstrip("."),
                "page": 1,
                "total_pages": 1,
            },
        )
        for src, content in pairs
    ]
    return split_documents(docs)


@pytest.fixture
def store(tmp_path: Path) -> VectorStoreManager:
    return VectorStoreManager("test_course", tmp_path)


# ============================================================
# 基础读写
# ============================================================
class TestAddAndSearch:
    def test_add_returns_ids(self, store, fake_embedder) -> None:
        chunks = make_chunks(
            [
                ("/a.txt", "Python 是一门解释型编程语言，语法简洁易读。"),
                ("/b.txt", "Java 是一门静态类型语言，广泛用于企业后端开发。"),
            ]
        )
        ids = store.add_documents(fake_embedder, chunks)

        assert len(ids) == 2
        assert store.count() == 2

    def test_search_finds_relevant(self, store, fake_embedder) -> None:
        """⭐ 核心能力：查"编程语言"应该把相关文本排在前面。"""
        chunks = make_chunks(
            [
                ("/a.txt", "Python 是一门解释型编程语言，语法简洁易读。"),
                ("/b.txt", "今天天气晴朗，适合出门散步和野餐。"),
            ]
        )
        store.add_documents(fake_embedder, chunks)

        results = store.similarity_search(fake_embedder, "编程语言", k=1)

        assert len(results) == 1
        assert "Python" in results[0].page_content

    def test_search_returns_scores(self, store, fake_embedder) -> None:
        store.add_documents(fake_embedder, make_chunks([("/a.txt", "机器学习基础")]))

        results = store.query(fake_embedder, "机器学习", k=1)

        assert len(results) == 1
        doc, score = results[0]
        # 用 cosine 空间，完全相同的文本相似度应接近 1
        assert 0.0 < score <= 1.001

    def test_results_sorted_by_score(self, store, fake_embedder) -> None:
        """返回结果必须按相似度降序。"""
        store.add_documents(
            fake_embedder,
            make_chunks(
                [
                    ("/a.txt", "机器学习是人工智能的重要分支领域。"),
                    ("/b.txt", "机器学习模型需要大量训练数据支撑。"),
                    ("/c.txt", "红烧肉的做法是先焯水再炒糖色。"),
                ]
            ),
        )

        results = store.query(fake_embedder, "机器学习", k=3)
        scores = [s for _, s in results]

        assert scores == sorted(scores, reverse=True)

    def test_empty_store_returns_empty(self, store, fake_embedder) -> None:
        """空库检索不能崩，应该返回空列表。"""
        assert store.similarity_search(fake_embedder, "任意查询") == []

    def test_add_empty_list(self, store, fake_embedder) -> None:
        assert store.add_documents(fake_embedder, []) == []

    def test_k_larger_than_count(self, store, fake_embedder) -> None:
        """k 大于库里总数时应返回全部，而不是报错。"""
        store.add_documents(fake_embedder, make_chunks([("/a.txt", "只有一条数据")]))
        assert len(store.similarity_search(fake_embedder, "数据", k=100)) == 1


# ============================================================
# upsert 幂等性
# ============================================================
class TestUpsert:
    def test_reindex_does_not_duplicate(self, store, fake_embedder) -> None:
        """⭐ 同样的 chunk 写两次，数量不应翻倍（upsert 语义）。"""
        chunks = make_chunks([("/a.txt", "重复索引测试内容，这段文字会被写两次。")])

        store.add_documents(fake_embedder, chunks)
        n1 = store.count()
        store.add_documents(fake_embedder, chunks)
        n2 = store.count()

        assert n1 == n2

    def test_duplicate_ids_in_same_batch(self, store, fake_embedder) -> None:
        """同一批里有重复 id 时要自动去重，不能让 Chroma 报错。"""
        chunks = make_chunks([("/a.txt", "内容甲乙丙丁戊己庚辛壬癸")])
        doubled = chunks + chunks  # 故意重复

        ids = store.add_documents(fake_embedder, doubled)

        assert len(ids) == len(chunks)

    def test_missing_chunk_id_raises(self, store, fake_embedder) -> None:
        """没经过 splitter 的 Document 不允许直接写入。"""
        bad = [Document(page_content="没有 chunk_id", metadata={"page": 1})]

        with pytest.raises(RAGError, match="chunk_id"):
            store.add_documents(fake_embedder, bad)


# ============================================================
# 元数据过滤
# ============================================================
class TestMetadataFilter:
    def test_filter_by_file_type(self, store, fake_embedder) -> None:
        store.add_documents(
            fake_embedder,
            make_chunks(
                [
                    ("/a.pdf", "机器学习课件第一章内容介绍。"),
                    ("/b.py", "机器学习代码示例 def train(): pass"),
                ]
            ),
        )

        results = store.similarity_search(
            fake_embedder, "机器学习", k=10, where={"file_type": "pdf"}
        )

        assert len(results) == 1
        assert results[0].metadata["file_type"] == "pdf"

    def test_metadata_preserved(self, store, fake_embedder) -> None:
        """写进去的 metadata 必须能原样取回来（溯源功能依赖它）。"""
        store.add_documents(
            fake_embedder, make_chunks([("/data/course/a.pdf", "第一页的内容文本")])
        )

        doc = store.similarity_search(fake_embedder, "内容", k=1)[0]

        assert doc.metadata["source_path"] == "/data/course/a.pdf"
        assert doc.metadata["source_name"] == "a.pdf"
        assert doc.metadata["page"] == 1
        assert "chunk_id" in doc.metadata


# ============================================================
# 删除
# ============================================================
class TestDelete:
    def test_delete_by_source(self, store, fake_embedder) -> None:
        store.add_documents(
            fake_embedder,
            make_chunks(
                [("/a.txt", "文件甲的内容部分"), ("/b.txt", "文件乙的内容部分")]
            ),
        )
        assert store.count() == 2

        # ⚠️ delete_by_source 内部会 resolve() 路径，
        #    所以这里传的路径要和写入时 metadata 里的一致
        deleted = store.delete_by_source(Path("/a.txt"))

        assert deleted == 1
        assert store.count() == 1

    def test_delete_collection(self, store, fake_embedder) -> None:
        store.add_documents(fake_embedder, make_chunks([("/a.txt", "一些内容文本")]))
        store.delete_collection()
        assert store.count() == 0

    def test_delete_missing_collection_is_safe(self, store) -> None:
        """删一个不存在的 collection 不应该抛异常（幂等）。"""
        store.delete_collection()
        store.delete_collection()  # 再删一次

    def test_reset(self, store, fake_embedder) -> None:
        store.add_documents(fake_embedder, make_chunks([("/a.txt", "内容内容内容")]))
        store.reset()
        assert store.count() == 0
        # reset 后应该能立刻继续写
        store.add_documents(fake_embedder, make_chunks([("/b.txt", "新的内容文本")]))
        assert store.count() == 1


# ============================================================
# 持久化 & 辅助
# ============================================================
class TestPersistence:
    def test_survives_new_manager(self, tmp_path: Path, fake_embedder) -> None:
        """⭐ 写完之后新建一个 manager，数据必须还在（持久化生效）。"""
        m1 = VectorStoreManager("c1", tmp_path)
        m1.add_documents(fake_embedder, make_chunks([("/a.txt", "持久化测试内容")]))

        m2 = VectorStoreManager("c1", tmp_path)  # 模拟重启进程

        assert m2.count() == 1

    def test_courses_isolated(self, tmp_path: Path, fake_embedder) -> None:
        """不同课程的向量互不可见。"""
        a = VectorStoreManager("course_a", tmp_path)
        b = VectorStoreManager("course_b", tmp_path)

        a.add_documents(fake_embedder, make_chunks([("/a.txt", "甲课程的内容")]))

        assert a.count() == 1
        assert b.count() == 0


class TestHelpers:
    def test_peek(self, store, fake_embedder) -> None:
        store.add_documents(
            fake_embedder,
            make_chunks([("/a.txt", "内容一二三"), ("/b.txt", "内容四五六")]),
        )
        assert len(store.peek(n=10)) == 2

    def test_list_sources(self, store, fake_embedder) -> None:
        store.add_documents(
            fake_embedder,
            make_chunks([("/a.txt", "内容甲甲甲"), ("/b.txt", "内容乙乙乙")]),
        )
        assert store.list_sources() == ["/a.txt", "/b.txt"]

    def test_count_on_fresh_store(self, tmp_path: Path) -> None:
        assert VectorStoreManager("brand_new", tmp_path).count() == 0
```

> **踩坑补充**
>
> | 现象 | 原因 | 解决 |
> | --- | --- | --- |
> | `Expected metadata value to be str/int/float/bool` | metadata 里混进了 None 或 list | 写入前过 `sanitize_metadata` |
> | 改了 `hnsw:space` 但距离没变 | space 只在建 collection 时生效 | `delete_collection()` 后重建 |
> | 相似度是负数 | 用了 l2 空间，`1 - 距离` 会变负 | 确认 collection 是 cosine 空间 |
> | 重复索引后条数翻倍 | 用了 `add` 而不是 `upsert`，或 chunk_id 不稳定 | 检查 §4 的 chunk_id 稳定性测试 |
> | 多条件 `where` 报错 | Chroma 需要显式 `$and` | `{"$and": [{...}, {...}]}` |

---

## 7. 索引器（Indexer）

### 7.1 目标

把"文档 → chunks → 向量 → Chroma"完整串起来，**支持增量构建**。

### 7.2 概念小课堂

- **全量构建**：每次都重新建整库（慢但稳）
- **增量构建**：只处理新增/修改的文件（快但要追踪状态）
- **Manifest**：记录每个文档的 sha256 + indexed_at，用来判断要不要重建

### 7.3 文件内容指引

#### `src/private_teacher/rag/indexer.py`

```python
from private_teacher.config import LLMSettings, PathSettings
from private_teacher.storage.db import Database
from private_teacher.storage.document_repo import list_by_course, get_by_sha256, mark_indexed
from private_teacher.loaders import load
from private_teacher.rag.splitters import split_documents
from private_teacher.rag.embeddings import get_embedder
from private_teacher.rag.vector_store import VectorStoreManager

class Indexer:
    def __init__(self, course_id: str, llm_settings: LLMSettings, path_settings: PathSettings):
        self.course_id = course_id
        self.llm = llm_settings
        self.paths = path_settings
        self.embedder = get_embedder(llm_settings)
        self.store_mgr = VectorStoreManager(course_id, path_settings.data_dir)

    def build_full(self, conn) -> dict:
        """全量构建：处理所有文档。"""
        docs = list_by_course(conn, self.course_id)
        all_chunks = []
        for doc in docs:
            file_docs = load(Path(doc.path))
            chunks = split_documents(file_docs)
            all_chunks.extend(chunks)

        # 清空再写
        self.store_mgr.delete_collection()
        ids = self.store_mgr.add_documents(self.embedder, all_chunks)

        for doc in docs:
            mark_indexed(conn, doc.id)

        return {
            "documents": len(docs),
            "chunks": len(all_chunks),
            "ids": ids,
        }

    def build_incremental(self, conn) -> dict:
        """增量构建：只处理新增/修改的文档。"""
        # 找出 indexed_at IS NULL 的文档（未索引）+ sha256 变化的
        new_chunks = []
        processed_docs = []
        for doc in list_by_course(conn, self.course_id):
            # 简单策略：indexed_at 为空 或 manifest 不匹配就重建
            if doc.indexed_at is None:
                file_docs = load(Path(doc.path))
                new_chunks.extend(split_documents(file_docs))
                processed_docs.append(doc)

        if new_chunks:
            self.store_mgr.add_documents(self.embedder, new_chunks)
            for doc in processed_docs:
                mark_indexed(conn, doc.id)

        return {
            "documents": len(processed_docs),
            "chunks": len(new_chunks),
        }
```

**关键设计**：
- 增量构建只处理 `indexed_at IS NULL` 的文档
- 全量构建前 `delete_collection()` 保证一致性

### 7.4 测试代码

**tests/unit/test_indexer.py**

```python
class TestIndexer:
    def test_build_full_creates_chunks(self, tmp_path, tmp_db, fake_embedder):
        # 准备：上传一个文档到 tmp_db
        course = create_course(tmp_db, "Python")
        f = tmp_path / "doc.txt"
        f.write_text("Python 是一种编程语言")
        create_doc(tmp_db, course.id, f, "main", sha256_of(f), f.stat().st_size)

        # 构建
        settings = LLMSettings(...)
        indexer = Indexer(course.id, settings, PathSettings(data_dir=tmp_path))
        result = indexer.build_full(tmp_db)
        assert result["chunks"] >= 1

    def test_incremental_only_processes_new(self, tmp_path, tmp_db):
        # 第一次构建
        ...
        # 再上传一个文档
        ...
        # 增量构建
        result = indexer.build_incremental(tmp_db)
        assert result["documents"] == 1  # 只处理新的
```

### 7.5 验证

```bash
uv run pytest tests/unit/test_indexer.py -v
```

### 7.6 完整代码实现

> Indexer 是 Phase 1 的**编排中枢**：它把 loader / splitter / embedder / vector_store
> 四个模块串成一条流水线，并负责"哪些文件需要重新处理"这个状态判断。

#### 7.6.1 `src/private_teacher/rag/manifest.py`（索引清单）

```python
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
```

#### 7.6.2 `src/private_teacher/rag/indexer.py`

```python
"""
索引构建器
==========

一条完整的流水线：

    documents 表
        ↓  load()            加载 → Document（带页码/行号）
        ↓  split_documents() 切分 → chunk（带 chunk_id）
        ↓  embed + upsert    向量化并写入 Chroma
        ↓  mark_indexed()    回写数据库状态
        ↓  manifest.save()   记录哈希，供下次增量判断

两种模式：
  build_full        —— 清空重来。用于"我怀疑索引坏了"
  build_incremental —— 只处理新增/修改/删除的文件。日常用这个
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from sqlite3 import Connection
from typing import Any

from loguru import logger

from private_teacher.config import LLMSettings, PathSettings
from private_teacher.loaders import load
from private_teacher.rag.embeddings import Embedder, get_cached_embedder
from private_teacher.rag.manifest import IndexManifest, ManifestEntry
from private_teacher.rag.splitters import ChunkerConfig, split_documents
from private_teacher.rag.vector_store import VectorStoreManager
from private_teacher.storage import document_repo
from private_teacher.storage.models import Document as DocRecord, now_utc, to_iso
from private_teacher.utils.exceptions import DocumentLoadError


@dataclass(slots=True)
class IndexResult:
    """一次索引操作的结果报告。

    为什么要返回这么细的统计？
      - UI 上要显示"新增 3 个文件 / 42 个片段"
      - 出问题时，errors 列表能直接告诉用户是哪个文件失败了，
        而不是让他去翻日志
    """

    documents_indexed: int = 0  # 成功建索引的文件数
    documents_skipped: int = 0  # 没变化、跳过的文件数
    documents_removed: int = 0  # 清理掉的已删除文件数
    chunks_added: int = 0  # 新增的 chunk 数
    errors: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return not self.errors

    def summary(self) -> str:
        """给 UI 用的一句话摘要。"""
        parts = [
            f"索引 {self.documents_indexed} 个文件",
            f"新增 {self.chunks_added} 个片段",
        ]
        if self.documents_skipped:
            parts.append(f"跳过 {self.documents_skipped} 个未变更文件")
        if self.documents_removed:
            parts.append(f"清理 {self.documents_removed} 个已删除文件")
        if self.errors:
            parts.append(f"⚠️ {len(self.errors)} 个失败")
        return "，".join(parts) + f"（耗时 {self.duration_seconds:.1f}s）"


class Indexer:
    """某一门课的知识库构建器。"""

    def __init__(
        self,
        course_id: str,
        llm_settings: LLMSettings,
        path_settings: PathSettings,
        chunker_config: ChunkerConfig | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        """
        Args:
            course_id: 课程 id
            llm_settings: 用来构造 embedder
            path_settings: 数据目录
            chunker_config: 切分参数；不传用默认
            embedder: 直接注入 embedder（测试用）。
                传了就不再从 settings 构造 —— 这叫"依赖注入"，
                让测试可以塞一个假的进来，不需要 mock 全局函数。
        """
        self.course_id = course_id
        self.llm = llm_settings
        self.paths = path_settings
        self.chunker_config = chunker_config or ChunkerConfig()

        # get_cached_embedder：本地模型只加载一次
        self._embedder = embedder or get_cached_embedder(llm_settings)
        self.store = VectorStoreManager(course_id, path_settings.data_dir)

    # ============================================================
    # 单个文件的处理
    # ============================================================
    def _index_one(
        self,
        doc: DocRecord,
        on_progress: Any = None,
    ) -> int:
        """加载 → 切分 → 写向量库。返回写入的 chunk 数。

        Raises:
            DocumentLoadError: 文件读取/解析失败
            RAGError: 向量化或写入失败
        """
        # ---------- 加载 ----------
        raw_docs = load(doc.path)
        if not raw_docs:
            # 空文件 / 扫描版 PDF —— 不算错误，但要如实告知
            logger.warning(f"{doc.path.name} 没有可提取的文本，跳过")
            return 0

        # ---------- 切分 ----------
        chunks = split_documents(raw_docs, self.chunker_config)
        if not chunks:
            return 0

        # 给每个 chunk 补上课程/文档/分类信息。
        # 为什么在这里补而不是在 loader 里？
        #   → loader 不该知道"课程"这种业务概念，它只管解析文件（分层原则）
        for chunk in chunks:
            chunk.metadata["course_id"] = self.course_id
            chunk.metadata["document_id"] = doc.id
            chunk.metadata["category"] = doc.category

        # ---------- 写入 ----------
        # 先删这个文件的旧向量：文件改小了的话，
        # 旧的多余 chunk（比如原来有 10 块现在只有 6 块）不会被 upsert 覆盖到，
        # 不删就会变成"幽灵内容"永远留在库里
        self.store.delete_by_source(doc.path)

        ids = self.store.add_documents(
            self._embedder,
            chunks,
            batch_size=self.llm.embedding_batch_size,
            on_progress=on_progress,
        )
        return len(ids)

    # ============================================================
    # 增量构建
    # ============================================================
    def build_incremental(
        self,
        conn: Connection,
        on_file: Any = None,
    ) -> IndexResult:
        """只处理新增 / 修改 / 删除的文件。

        Args:
            conn: 数据库连接
            on_file: 可选回调 (当前序号, 总数, 文件名)，给进度条用

        Returns:
            IndexResult 统计报告
        """
        started = time.perf_counter()
        result = IndexResult()

        manifest = IndexManifest.load(self.course_id, self.paths.data_dir)
        docs = document_repo.list_by_course(conn, self.course_id)

        # ---------- 1. 清理已删除的文件 ----------
        current_paths = {str(d.path.resolve()) for d in docs}
        for stale in manifest.stale_sources(current_paths):
            self.store.delete_by_source(stale)
            manifest.remove(stale)
            result.documents_removed += 1
            logger.info(f"清理已删除文件的向量：{Path(stale).name}")

        # ---------- 2. 处理新增/修改 ----------
        total = len(docs)
        for idx, doc in enumerate(docs, start=1):
            source_key = str(doc.path.resolve())

            if on_file is not None:
                on_file(idx, total, doc.path.name)

            # 物理文件被用户手动删了
            if not doc.path.exists():
                result.errors.append(f"{doc.path.name}: 文件不存在")
                continue

            # 内容没变 → 跳过（增量的意义所在）
            if not manifest.needs_index(source_key, doc.sha256):
                result.documents_skipped += 1
                continue

            try:
                n_chunks = self._index_one(doc)
            except DocumentLoadError as exc:
                # 单个文件失败不中断整批 —— 用户可能上传了 50 个文件，
                # 不能因为第 7 个是加密 PDF 就让前 6 个的工作白费
                result.errors.append(f"{doc.path.name}: {exc}")
                logger.warning(f"索引失败 {doc.path.name}: {exc}")
                continue
            except Exception as exc:  # noqa: BLE001
                result.errors.append(f"{doc.path.name}: {exc}")
                logger.exception(f"索引 {doc.path.name} 时发生未预期错误")
                continue

            # ---------- 成功：更新三处状态 ----------
            document_repo.mark_indexed(conn, doc.id)  # ① 数据库
            manifest.upsert(  # ② 清单
                ManifestEntry(
                    source_path=source_key,
                    sha256=doc.sha256,
                    chunk_count=n_chunks,
                    indexed_at=to_iso(now_utc()) or "",
                    document_id=doc.id,
                )
            )
            result.documents_indexed += 1
            result.chunks_added += n_chunks

        # ③ 落盘
        manifest.save(self.paths.data_dir)

        result.duration_seconds = time.perf_counter() - started
        logger.info(f"增量索引完成：{result.summary()}")
        return result

    # ============================================================
    # 全量构建
    # ============================================================
    def build_full(
        self,
        conn: Connection,
        on_file: Any = None,
    ) -> IndexResult:
        """推倒重来：清空向量库和清单，然后全部重新索引。

        什么时候用？
          - 换了 embedding 模型（向量维度/语义空间都变了，旧向量全部作废）
          - 改了 chunk_size（chunk 边界变了，chunk_id 也会变）
          - 怀疑索引状态不一致

        实现上就是"先清空，再走一遍增量" —— 清空之后所有文件都变成"新增"，
        增量逻辑自然会处理全部文件。**不要为全量单独写一套流程**，
        两套代码迟早会出现行为不一致的 bug。
        """
        logger.info(f"开始全量重建知识库（course={self.course_id}）")

        # ① 删向量库（连 collection 一起删，确保 hnsw 配置也重置）
        self.store.reset()

        # ② 清空清单
        manifest = IndexManifest(course_id=self.course_id)
        manifest.save(self.paths.data_dir)

        # ③ 把数据库里的 indexed_at 全部清空
        document_repo.mark_all_unindexed(conn, self.course_id)

        # ④ 走增量流程（此时所有文件都是"新增"）
        return self.build_incremental(conn, on_file=on_file)

    # ============================================================
    # 单文件维护
    # ============================================================
    def remove_document(self, conn: Connection, document: DocRecord) -> int:
        """从知识库里移除一个文档的所有向量（用户删除课件时调用）。

        Returns:
            删除的向量条数
        """
        source_key = str(document.path.resolve())

        n = self.store.delete_by_source(source_key)

        manifest = IndexManifest.load(self.course_id, self.paths.data_dir)
        manifest.remove(source_key)
        manifest.save(self.paths.data_dir)

        return n

    # ============================================================
    # 状态查询
    # ============================================================
    def stats(self, conn: Connection) -> dict[str, Any]:
        """知识库现状。UI 顶部的指标卡片用。"""
        manifest = IndexManifest.load(self.course_id, self.paths.data_dir)
        docs = document_repo.list_by_course(conn, self.course_id)

        return {
            "documents_total": len(docs),
            "documents_indexed": len(manifest.entries),
            "documents_pending": sum(
                1
                for d in docs
                if manifest.needs_index(str(d.path.resolve()), d.sha256)
            ),
            "chunks_in_manifest": manifest.total_chunks,
            "vectors_in_store": self.store.count(),
        }
```

#### 7.6.3 `tests/unit/test_indexer.py`（完整版）

```python
"""indexer 单元测试（全部离线，用 HashEmbedder）。"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from private_teacher.config import PathSettings
from private_teacher.rag.indexer import Indexer
from private_teacher.rag.manifest import IndexManifest, ManifestEntry
from private_teacher.storage import course_repo, document_repo
from private_teacher.utils.hashing import compute_sha256


# ============================================================
# 测试辅助
# ============================================================
@pytest.fixture
def paths(tmp_path: Path) -> PathSettings:
    return PathSettings(data_dir=tmp_path / "data")


@pytest.fixture
def course(tmp_db: sqlite3.Connection):
    return course_repo.create(tmp_db, name="测试课程")


def add_file(
    conn: sqlite3.Connection,
    course_id: str,
    tmp_path: Path,
    name: str,
    content: str,
    category: str = "main",
):
    """在磁盘上造一个文件并登记到数据库。返回 Document 记录。"""
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")

    return document_repo.create(
        conn,
        course_id,
        path,
        category,
        compute_sha256(path),
        path.stat().st_size,
    )


@pytest.fixture
def indexer(course, hash_llm_settings, paths, fake_embedder) -> Indexer:
    return Indexer(
        course.id,
        hash_llm_settings,
        paths,
        embedder=fake_embedder,  # 依赖注入，完全离线
    )


# ============================================================
# Manifest
# ============================================================
class TestManifest:
    def test_load_missing_returns_empty(self, tmp_path: Path) -> None:
        m = IndexManifest.load("c1", tmp_path)
        assert m.entries == {}

    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        m = IndexManifest(course_id="c1")
        m.upsert(
            ManifestEntry(
                source_path="/a.txt",
                sha256="abc",
                chunk_count=3,
                indexed_at="2026-01-01T00:00:00+00:00",
            )
        )
        m.save(tmp_path)

        loaded = IndexManifest.load("c1", tmp_path)

        assert loaded.entries["/a.txt"].sha256 == "abc"
        assert loaded.total_chunks == 3

    def test_corrupted_file_returns_empty(self, tmp_path: Path) -> None:
        """损坏的 manifest 不该让程序崩溃，应降级为空清单。"""
        path = IndexManifest.path_for("c1", tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{ 这不是合法 JSON", encoding="utf-8")

        assert IndexManifest.load("c1", tmp_path).entries == {}

    def test_version_mismatch_returns_empty(self, tmp_path: Path) -> None:
        path = IndexManifest.path_for("c1", tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"version": 999, "entries": {}}', encoding="utf-8")

        assert IndexManifest.load("c1", tmp_path).entries == {}

    def test_needs_index_logic(self) -> None:
        m = IndexManifest(course_id="c1")

        assert m.needs_index("/a.txt", "sha1") is True  # 全新

        m.upsert(
            ManifestEntry("/a.txt", "sha1", 1, "2026-01-01T00:00:00+00:00")
        )
        assert m.needs_index("/a.txt", "sha1") is False  # 没变
        assert m.needs_index("/a.txt", "sha2") is True  # 变了

    def test_stale_sources(self) -> None:
        m = IndexManifest(course_id="c1")
        m.upsert(ManifestEntry("/a.txt", "s", 1, "t"))
        m.upsert(ManifestEntry("/b.txt", "s", 1, "t"))

        assert m.stale_sources({"/a.txt"}) == ["/b.txt"]


# ============================================================
# 全量构建
# ============================================================
class TestBuildFull:
    def test_creates_chunks(self, tmp_db, course, tmp_path, indexer) -> None:
        add_file(
            tmp_db,
            course.id,
            tmp_path,
            "doc.txt",
            "Python 是一种广泛使用的解释型编程语言，强调代码可读性。",
        )

        result = indexer.build_full(tmp_db)

        assert result.ok
        assert result.documents_indexed == 1
        assert result.chunks_added >= 1
        assert indexer.store.count() == result.chunks_added

    def test_marks_documents_indexed(self, tmp_db, course, tmp_path, indexer) -> None:
        doc = add_file(tmp_db, course.id, tmp_path, "a.txt", "一些课程内容文本")

        indexer.build_full(tmp_db)

        assert document_repo.get_by_id(tmp_db, doc.id).is_indexed is True
        assert document_repo.list_unindexed(tmp_db, course.id) == []

    def test_writes_manifest(self, tmp_db, course, tmp_path, indexer, paths) -> None:
        add_file(tmp_db, course.id, tmp_path, "a.txt", "课程内容文本内容")

        indexer.build_full(tmp_db)

        manifest = IndexManifest.load(course.id, paths.data_dir)
        assert len(manifest.entries) == 1
        assert manifest.total_chunks >= 1

    def test_rebuild_does_not_duplicate(
        self, tmp_db, course, tmp_path, indexer
    ) -> None:
        """⭐ 全量重建两次，向量数不应翻倍。"""
        add_file(tmp_db, course.id, tmp_path, "a.txt", "重建测试内容文本")

        indexer.build_full(tmp_db)
        n1 = indexer.store.count()
        indexer.build_full(tmp_db)
        n2 = indexer.store.count()

        assert n1 == n2

    def test_empty_course(self, tmp_db, course, indexer) -> None:
        """一个文件都没有时也要正常返回，不能崩。"""
        result = indexer.build_full(tmp_db)
        assert result.ok
        assert result.documents_indexed == 0


# ============================================================
# 增量构建
# ============================================================
class TestBuildIncremental:
    def test_only_processes_new(self, tmp_db, course, tmp_path, indexer) -> None:
        """⭐ 增量的核心：已索引的文件要被跳过。"""
        add_file(tmp_db, course.id, tmp_path, "a.txt", "第一个文件的内容")
        indexer.build_incremental(tmp_db)

        add_file(tmp_db, course.id, tmp_path, "b.txt", "第二个文件的内容")
        result = indexer.build_incremental(tmp_db)

        assert result.documents_indexed == 1  # 只处理了新的
        assert result.documents_skipped == 1  # 老的被跳过

    def test_second_run_skips_everything(
        self, tmp_db, course, tmp_path, indexer
    ) -> None:
        add_file(tmp_db, course.id, tmp_path, "a.txt", "内容没有任何变化")

        indexer.build_incremental(tmp_db)
        result = indexer.build_incremental(tmp_db)

        assert result.documents_indexed == 0
        assert result.documents_skipped == 1
        assert result.chunks_added == 0

    def test_modified_file_reindexed(self, tmp_db, course, tmp_path, indexer) -> None:
        """文件内容改了（sha256 变了）→ 必须重新索引。"""
        doc = add_file(tmp_db, course.id, tmp_path, "a.txt", "原始内容文本")
        indexer.build_incremental(tmp_db)

        # 模拟用户改了文件：改内容 + 更新数据库里的 sha256
        doc.path.write_text("修改后的全新内容文本", encoding="utf-8")
        tmp_db.execute(
            "UPDATE documents SET sha256 = ? WHERE id = ?",
            (compute_sha256(doc.path), doc.id),
        )
        tmp_db.commit()

        result = indexer.build_incremental(tmp_db)

        assert result.documents_indexed == 1
        assert result.documents_skipped == 0

    def test_deleted_file_cleaned_up(
        self, tmp_db, course, tmp_path, indexer
    ) -> None:
        """⭐ 数据库里删掉的文档，它的向量残留必须被清理。"""
        doc = add_file(tmp_db, course.id, tmp_path, "a.txt", "将要被删除的内容")
        add_file(tmp_db, course.id, tmp_path, "b.txt", "会保留下来的内容")
        indexer.build_incremental(tmp_db)
        assert indexer.store.count() >= 2

        document_repo.delete(tmp_db, doc.id)
        result = indexer.build_incremental(tmp_db)

        assert result.documents_removed == 1
        assert "/a.txt" not in " ".join(indexer.store.list_sources())

    def test_shrinking_file_removes_old_chunks(
        self, tmp_db, course, tmp_path, indexer
    ) -> None:
        """⚠️ 文件变短后，多余的旧 chunk 必须消失（不能留幽灵内容）。"""
        doc = add_file(
            tmp_db, course.id, tmp_path, "a.txt", "长内容段落。" * 500
        )
        indexer.build_incremental(tmp_db)
        n_before = indexer.store.count()
        assert n_before >= 3

        doc.path.write_text("变得很短的内容。", encoding="utf-8")
        tmp_db.execute(
            "UPDATE documents SET sha256 = ? WHERE id = ?",
            (compute_sha256(doc.path), doc.id),
        )
        tmp_db.commit()
        indexer.build_incremental(tmp_db)

        assert indexer.store.count() < n_before

    def test_missing_physical_file_reported(
        self, tmp_db, course, tmp_path, indexer
    ) -> None:
        """物理文件被手删 → 记进 errors，但不中断整批。"""
        doc = add_file(tmp_db, course.id, tmp_path, "a.txt", "会被删掉的文件")
        add_file(tmp_db, course.id, tmp_path, "b.txt", "正常存在的文件内容")
        doc.path.unlink()

        result = indexer.build_incremental(tmp_db)

        assert not result.ok
        assert len(result.errors) == 1
        assert "a.txt" in result.errors[0]
        assert result.documents_indexed == 1  # b.txt 仍然成功

    def test_unsupported_file_reported(
        self, tmp_db, course, tmp_path, indexer
    ) -> None:
        """不支持的格式记入 errors，不抛异常。"""
        add_file(tmp_db, course.id, tmp_path, "a.exe", "二进制内容")

        result = indexer.build_incremental(tmp_db)

        assert not result.ok
        assert result.documents_indexed == 0

    def test_progress_callback(self, tmp_db, course, tmp_path, indexer) -> None:
        add_file(tmp_db, course.id, tmp_path, "a.txt", "内容甲甲甲甲甲")
        add_file(tmp_db, course.id, tmp_path, "b.txt", "内容乙乙乙乙乙")

        seen: list[tuple[int, int, str]] = []
        indexer.build_incremental(
            tmp_db, on_file=lambda i, n, name: seen.append((i, n, name))
        )

        assert len(seen) == 2
        assert seen[0][1] == 2  # 总数正确


# ============================================================
# 元数据注入 & 统计
# ============================================================
class TestMetadataAndStats:
    def test_chunks_carry_course_metadata(
        self, tmp_db, course, tmp_path, indexer, fake_embedder
    ) -> None:
        """chunk 里必须带上 course_id / document_id / category。"""
        doc = add_file(
            tmp_db, course.id, tmp_path, "a.txt", "课程内容文本", "auxiliary"
        )
        indexer.build_full(tmp_db)

        hit = indexer.store.similarity_search(fake_embedder, "课程", k=1)[0]

        assert hit.metadata["course_id"] == course.id
        assert hit.metadata["document_id"] == doc.id
        assert hit.metadata["category"] == "auxiliary"

    def test_stats(self, tmp_db, course, tmp_path, indexer) -> None:
        add_file(tmp_db, course.id, tmp_path, "a.txt", "已索引的内容文本")
        indexer.build_full(tmp_db)
        add_file(tmp_db, course.id, tmp_path, "b.txt", "还没索引的内容")

        stats = indexer.stats(tmp_db)

        assert stats["documents_total"] == 2
        assert stats["documents_indexed"] == 1
        assert stats["documents_pending"] == 1
        assert stats["vectors_in_store"] >= 1

    def test_remove_document(self, tmp_db, course, tmp_path, indexer, paths) -> None:
        doc = add_file(tmp_db, course.id, tmp_path, "a.txt", "要被移除的内容文本")
        indexer.build_full(tmp_db)

        removed = indexer.remove_document(tmp_db, doc)

        assert removed >= 1
        assert indexer.store.count() == 0
        assert IndexManifest.load(course.id, paths.data_dir).entries == {}


class TestIndexResult:
    def test_summary_readable(self) -> None:
        from private_teacher.rag.indexer import IndexResult

        r = IndexResult(
            documents_indexed=3, chunks_added=42, documents_skipped=1,
            duration_seconds=1.234,
        )
        s = r.summary()

        assert "3 个文件" in s
        assert "42 个片段" in s
        assert r.ok is True
```

---

## 8. 检索器（Retriever）

### 8.1 目标

根据用户 query 返回**最相关的 top-k chunks**。

### 8.2 概念小课堂

- **纯向量检索**：用 embedding 相似度（语义强，但漏关键词）
- **BM25 关键词检索**：传统 TF-IDF 算法（精确匹配强，但不懂语义）
- **混合检索（Hybrid）**：两者结合，效果最好

### 8.3 文件内容指引

#### `src/private_teacher/rag/retriever.py`

**先实现纯向量版（hybrid 留 Phase 6）**：

```python
class Retriever:
    def __init__(self, course_id: str, llm_settings: LLMSettings, paths: PathSettings):
        self.course_id = course_id
        self.embedder = get_embedder(llm_settings)
        self.store_mgr = VectorStoreManager(course_id, paths.data_dir)

    def retrieve(self, query: str, k: int = 5) -> list[Document]:
        store = self.store_mgr.get_store(self.embedder)
        return store.similarity_search(query, k=k)
```

**Phase 6 再加 BM25**（用 `rank_bm25`）：

```python
class HybridRetriever:
    def __init__(self, ...):
        self.vector_retriever = Retriever(...)
        self.bm25 = self._build_bm25_index()
    
    def retrieve(self, query, k=5):
        # 向量 + BM25 各自取 top-k
        vec_results = self.vector_retriever.retrieve(query, k=k)
        bm25_results = self.bm25.search(query, k=k)
        # 用 reciprocal rank fusion 合并
        return self._fuse(vec_results, bm25_results)
```

### 8.4 测试代码

**tests/unit/test_retriever.py**

```python
class TestRetriever:
    def test_retrieve_returns_relevant(self, tmp_path, fake_embedder):
        # 准备知识库
        mgr = VectorStoreManager("c1", tmp_path)
        chunks = [
            Document(page_content="机器学习是AI的子领域", metadata={"page": 1}),
            Document(page_content="Python 用于数据科学", metadata={"page": 1}),
        ]
        mgr.add_documents(fake_embedder, chunks)

        # 检索
        retriever = Retriever("c1", ...)
        results = retriever.retrieve("机器学习", k=1)
        assert "机器学习" in results[0].page_content
```

### 8.5 验证

```bash
uv run pytest tests/unit/test_retriever.py -v
```

### 8.6 完整代码实现

#### 8.6.1 `src/private_teacher/rag/retriever.py`

```python
"""
检索器
======

职责：把用户的一句话，变成 top-k 个最相关的课件片段。

【为什么不直接调 VectorStoreManager.similarity_search？】
因为真实检索需要在"向量搜索"之外做这些事：
  1. 过滤：只搜主干课件 / 只搜 PDF
  2. 超采样 + 去重：一个文件可能霸占全部 5 个名额，
     用户其实更想看到 3 个不同来源的结果
  3. 分数阈值：相似度 0.1 的结果与其给出来误导 LLM，不如不给
  4. 上下文拼装：把 top-k 拼成一段带来源标注的文本，直接喂给 LLM

这些都是"检索策略"，不该塞进向量库封装层里。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from loguru import logger

from private_teacher.config import LLMSettings, PathSettings
from private_teacher.rag.embeddings import Embedder, get_cached_embedder
from private_teacher.rag.vector_store import VectorStoreManager


@dataclass(slots=True)
class RetrievalResult:
    """一条检索结果。"""

    document: Document
    score: float  # 相似度，越大越相关
    rank: int  # 名次，从 1 开始

    # ---------- 便捷属性（UI 直接用，不用到处翻 metadata） ----------
    @property
    def content(self) -> str:
        return self.document.page_content

    @property
    def source_name(self) -> str:
        return str(self.document.metadata.get("source_name", "未知文件"))

    @property
    def source_path(self) -> Path:
        return Path(str(self.document.metadata.get("source_path", "")))

    @property
    def page(self) -> int:
        return int(self.document.metadata.get("page", 1))

    @property
    def chunk_id(self) -> str:
        return str(self.document.metadata.get("chunk_id", ""))

    def citation(self) -> str:
        """生成人类可读的引用标记，如 "机器学习.pdf 第 12 页"。"""
        parts = [self.source_name]

        # 有行号的（代码/文本）优先显示行号，比页码更精确
        line_start = self.document.metadata.get("line_start")
        line_end = self.document.metadata.get("line_end")
        if line_start:
            parts.append(f"第 {line_start}-{line_end} 行")
        elif int(self.document.metadata.get("total_pages", 1)) > 1:
            parts.append(f"第 {self.page} 页")

        return " ".join(parts)


class Retriever:
    """某门课的向量检索器。"""

    def __init__(
        self,
        course_id: str,
        llm_settings: LLMSettings,
        paths: PathSettings,
        embedder: Embedder | None = None,
    ) -> None:
        self.course_id = course_id
        self.llm = llm_settings
        self._embedder = embedder or get_cached_embedder(llm_settings)
        self.store = VectorStoreManager(course_id, paths.data_dir)

    # ============================================================
    # 过滤条件构造
    # ============================================================
    @staticmethod
    def _build_where(
        category: str | None,
        file_types: list[str] | None,
    ) -> dict[str, Any] | None:
        """把易用的参数翻译成 Chroma 的 where 语法。

        Chroma 的规则（很容易踩坑）：
          - 单条件：{"category": "main"}
          - 多条件：必须显式 {"$and": [{...}, {...}]}，
            **不能**写成 {"category": "main", "file_type": "pdf"}（会报错）
          - 集合匹配：{"file_type": {"$in": ["pdf", "md"]}}
        """
        conditions: list[dict[str, Any]] = []

        if category:
            conditions.append({"category": category})

        if file_types:
            # 统一小写、去掉可能带的点：".PDF" → "pdf"
            normalized = [t.lower().lstrip(".") for t in file_types]
            conditions.append({"file_type": {"$in": normalized}})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    # ============================================================
    # 核心检索
    # ============================================================
    def retrieve_with_scores(
        self,
        query: str,
        k: int = 5,
        category: str | None = None,
        file_types: list[str] | None = None,
        min_score: float | None = None,
        max_per_source: int | None = None,
    ) -> list[RetrievalResult]:
        """检索并返回带分数的结果。

        Args:
            query: 查询语句
            k: 最终返回条数
            category: "main" / "auxiliary"，None = 不限
            file_types: 如 ["pdf", "md"]，None = 不限
            min_score: 相似度下限，低于它的结果直接丢弃。
                建议值 0.2~0.3（cosine）。设太高会经常一条都搜不到
            max_per_source: 同一个源文件最多返回几条。
                设成 2 可以避免"一个文件霸占全部结果"

        Returns:
            按相关度降序的 RetrievalResult 列表
        """
        if not query.strip():
            return []

        where = self._build_where(category, file_types)

        # ---------- 超采样 ----------
        # 因为后面要按 min_score 和 max_per_source 过滤掉一部分，
        # 所以先多取一些，保证过滤完还能凑够 k 条。
        # 3 倍是经验值，上限 50 防止 k 很大时把整个库都拉出来
        fetch_k = min(k * 3, 50) if (min_score or max_per_source) else k

        raw = self.store.query(self._embedder, query, k=fetch_k, where=where)

        results: list[RetrievalResult] = []
        per_source: dict[str, int] = {}

        for doc, score in raw:
            # ---------- 分数过滤 ----------
            if min_score is not None and score < min_score:
                # 结果已按分数降序，一旦低于阈值，后面的只会更低 → 直接停
                break

            # ---------- 来源配额 ----------
            if max_per_source is not None:
                src = str(doc.metadata.get("source_path", ""))
                if per_source.get(src, 0) >= max_per_source:
                    continue
                per_source[src] = per_source.get(src, 0) + 1

            # 分数也放进 metadata 一份，方便直接传给只认 Document 的下游
            doc.metadata["score"] = round(score, 4)

            results.append(
                RetrievalResult(document=doc, score=score, rank=len(results) + 1)
            )

            if len(results) >= k:
                break

        logger.debug(f"检索 '{query[:30]}' → {len(results)} 条结果")
        return results

    def retrieve(self, query: str, k: int = 5, **kwargs: Any) -> list[Document]:
        """只要 Document 的简化版（分数在 metadata["score"] 里）。"""
        return [r.document for r in self.retrieve_with_scores(query, k, **kwargs)]

    # ============================================================
    # 给 LLM 用的上下文拼装
    # ============================================================
    def build_context(
        self,
        query: str,
        k: int = 5,
        max_chars: int = 4000,
        **kwargs: Any,
    ) -> tuple[str, list[RetrievalResult]]:
        """检索并拼成可直接塞进 prompt 的上下文字符串。

        Args:
            max_chars: 上下文长度上限。超过就截断 ——
                盲目把 20 个 chunk 全塞进 prompt 会爆 token 且稀释重点

        Returns:
            (上下文文本, 用到的结果列表)

        输出格式：
            [1] 机器学习.pdf 第 3 页
            决策树是一种树形结构的分类模型……

            [2] notes.md
            ……

        为什么要编号 [1] [2]？
          → 让 LLM 能在回答里写"根据 [1]"，我们就能把它映射回真实来源，
            这是"可溯源问答"的基础（Phase 2 会重度依赖）。
        """
        results = self.retrieve_with_scores(query, k, **kwargs)
        if not results:
            return "", []

        blocks: list[str] = []
        used: list[RetrievalResult] = []
        total = 0

        for i, r in enumerate(results, start=1):
            block = f"[{i}] {r.citation()}\n{r.content}"

            # 加上这一块会超长 → 停止（保证 prompt 不失控）
            if total + len(block) > max_chars and used:
                break

            blocks.append(block)
            used.append(r)
            total += len(block)

        return "\n\n".join(blocks), used

    # ============================================================
    # 状态
    # ============================================================
    def is_ready(self) -> bool:
        """知识库里有没有数据。UI 上用它决定是否禁用搜索框。"""
        return self.store.count() > 0
```

#### 8.6.2 混合检索（选做，Phase 6 正式启用）

> `rank-bm25` 已经在依赖里了，这里给出完整实现。
> **建议 Phase 1 先只用向量检索**，把全链路跑通再回来加这一层。

```python
# 追加到 src/private_teacher/rag/retriever.py 末尾

class HybridRetriever:
    """向量检索 + BM25 关键词检索，用 RRF 融合。

    【为什么要混合？】
      向量检索懂语义但对精确词不敏感：
        搜 "ResNet50"，可能返回一堆讲"深度网络"的段落，却漏掉真正提到 ResNet50 的那句
      BM25 精确匹配强但不懂同义：
        搜 "如何评估模型"，匹配不到只写了"模型评价指标"的段落
      两者的失败模式正好互补，所以合起来效果最好。

    【RRF（Reciprocal Rank Fusion，倒数排名融合）】
        score(doc) = Σ  1 / (k + rank_i(doc))
                    各路召回
      k 通常取 60。它的妙处是**只看排名不看分数** ——
      向量的余弦相似度（0~1）和 BM25 的分数（0~几十）量纲完全不同，
      直接加权求和需要繁琐的归一化，而 RRF 天然免疫这个问题。
    """

    RRF_K = 60

    def __init__(
        self,
        course_id: str,
        llm_settings: LLMSettings,
        paths: PathSettings,
        embedder: Embedder | None = None,
    ) -> None:
        self.vector = Retriever(course_id, llm_settings, paths, embedder)
        self.store = self.vector.store
        # BM25 索引是懒加载的：它需要把全库文本读进内存，
        # 只有真的用到混合检索时才付这个代价
        self._bm25: Any = None
        self._bm25_docs: list[Document] = []

    # ---------- BM25 索引 ----------
    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """分词。

        中文没有空格，这里用最朴素的"单字 + bigram"方案，
        效果比整句当一个 token 好得多。
        要更好的效果可以引入 jieba：`uv add jieba`，然后
            return list(jieba.cut_for_search(text))
        """
        import re

        words = re.findall(r"[a-zA-Z0-9_]+", text.lower())
        cjk = [c for c in text if "一" <= c <= "鿿"]
        bigrams = [a + b for a, b in zip(cjk, cjk[1:], strict=False)]
        return words + cjk + bigrams

    def _ensure_bm25(self) -> None:
        """构建 BM25 索引（懒加载 + 缓存）。"""
        if self._bm25 is not None:
            return

        from rank_bm25 import BM25Okapi

        # 把整个 collection 拉出来。
        # ⚠️ 库很大时这会很占内存 —— 十万级 chunk 就该换 Elasticsearch 了
        collection = self.store.get_collection()
        raw = collection.get(include=["documents", "metadatas"])

        self._bm25_docs = [
            Document(page_content=text, metadata=dict(meta or {}))
            for text, meta in zip(
                raw.get("documents") or [],
                raw.get("metadatas") or [],
                strict=False,
            )
        ]

        if not self._bm25_docs:
            self._bm25 = None
            return

        corpus = [self._tokenize(d.page_content) for d in self._bm25_docs]
        self._bm25 = BM25Okapi(corpus)

    def invalidate(self) -> None:
        """索引更新后调用，强制下次重建 BM25。"""
        self._bm25 = None
        self._bm25_docs = []

    # ---------- 检索 ----------
    def _bm25_search(self, query: str, k: int) -> list[Document]:
        self._ensure_bm25()
        if self._bm25 is None:
            return []

        scores = self._bm25.get_scores(self._tokenize(query))

        # 取分数最高的 k 个下标：
        # enumerate 配对成 (下标, 分数)，按分数降序排，取前 k
        ranked = sorted(enumerate(scores), key=lambda pair: pair[1], reverse=True)
        return [self._bm25_docs[i] for i, s in ranked[:k] if s > 0]

    def retrieve(self, query: str, k: int = 5, **kwargs: Any) -> list[Document]:
        """混合检索。"""
        # 两路各取 2k 条，融合后再截断到 k
        vec_docs = self.vector.retrieve(query, k=k * 2, **kwargs)
        bm25_docs = self._bm25_search(query, k=k * 2)

        # ---------- RRF 融合 ----------
        fused: dict[str, float] = {}
        by_id: dict[str, Document] = {}

        for rank, doc in enumerate(vec_docs, start=1):
            cid = str(doc.metadata.get("chunk_id", id(doc)))
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (self.RRF_K + rank)
            by_id[cid] = doc

        for rank, doc in enumerate(bm25_docs, start=1):
            cid = str(doc.metadata.get("chunk_id", id(doc)))
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (self.RRF_K + rank)
            by_id.setdefault(cid, doc)

        ordered = sorted(fused.items(), key=lambda pair: pair[1], reverse=True)

        results: list[Document] = []
        for cid, score in ordered[:k]:
            doc = by_id[cid]
            doc.metadata["rrf_score"] = round(score, 6)
            results.append(doc)

        return results
```

#### 8.6.3 `tests/unit/test_retriever.py`（完整版）

```python
"""retriever 单元测试（离线）。"""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.documents import Document

from private_teacher.config import PathSettings
from private_teacher.rag.retriever import HybridRetriever, RetrievalResult, Retriever
from private_teacher.rag.splitters import split_documents
from private_teacher.rag.vector_store import VectorStoreManager

COURSE_ID = "c1"


@pytest.fixture
def paths(tmp_path: Path) -> PathSettings:
    return PathSettings(data_dir=tmp_path / "data")


@pytest.fixture
def seeded(paths, fake_embedder) -> VectorStoreManager:
    """预先灌了 4 条数据的向量库。"""
    store = VectorStoreManager(COURSE_ID, paths.data_dir)

    raw = [
        ("/course/ml.pdf", "pdf", "main", 1, "机器学习是人工智能的核心分支领域。"),
        ("/course/ml.pdf", "pdf", "main", 2, "监督学习需要带标签的训练数据集合。"),
        ("/course/code.py", "py", "auxiliary", 1, "def train_model(): 训练模型的代码"),
        ("/course/food.md", "md", "auxiliary", 1, "红烧肉需要先焯水再炒糖色调味。"),
    ]

    docs = [
        Document(
            page_content=content,
            metadata={
                "source_path": src,
                "source_name": Path(src).name,
                "file_type": ftype,
                "category": cat,
                "page": page,
                "total_pages": 2,
                "course_id": COURSE_ID,
            },
        )
        for src, ftype, cat, page, content in raw
    ]
    store.add_documents(fake_embedder, split_documents(docs))
    return store


@pytest.fixture
def retriever(hash_llm_settings, paths, fake_embedder, seeded) -> Retriever:
    return Retriever(COURSE_ID, hash_llm_settings, paths, embedder=fake_embedder)


# ============================================================
# 基本检索
# ============================================================
class TestRetrieve:
    def test_returns_relevant(self, retriever) -> None:
        """⭐ 查"机器学习"，最相关的应排第一。"""
        results = retriever.retrieve("机器学习", k=1)

        assert len(results) == 1
        assert "机器学习" in results[0].page_content

    def test_ranked_by_score(self, retriever) -> None:
        results = retriever.retrieve_with_scores("机器学习", k=4)
        scores = [r.score for r in results]

        assert scores == sorted(scores, reverse=True)
        assert [r.rank for r in results] == [1, 2, 3, 4]

    def test_score_in_metadata(self, retriever) -> None:
        doc = retriever.retrieve("机器学习", k=1)[0]
        assert "score" in doc.metadata

    def test_empty_query(self, retriever) -> None:
        assert retriever.retrieve("") == []
        assert retriever.retrieve("   ") == []

    def test_empty_store(self, hash_llm_settings, paths, fake_embedder) -> None:
        empty = Retriever("nobody", hash_llm_settings, paths, embedder=fake_embedder)
        assert empty.retrieve("任意查询") == []
        assert empty.is_ready() is False

    def test_is_ready(self, retriever) -> None:
        assert retriever.is_ready() is True


# ============================================================
# 过滤
# ============================================================
class TestFilters:
    def test_where_single_condition(self) -> None:
        assert Retriever._build_where("main", None) == {"category": "main"}

    def test_where_multiple_uses_and(self) -> None:
        """⚠️ 多条件必须包在 $and 里，否则 Chroma 报错。"""
        where = Retriever._build_where("main", ["pdf"])
        assert "$and" in where

    def test_where_none(self) -> None:
        assert Retriever._build_where(None, None) is None

    def test_file_type_normalized(self) -> None:
        where = Retriever._build_where(None, [".PDF", "MD"])
        assert where == {"file_type": {"$in": ["pdf", "md"]}}

    def test_filter_by_category(self, retriever) -> None:
        results = retriever.retrieve("学习", k=10, category="main")
        assert all(d.metadata["category"] == "main" for d in results)
        assert len(results) >= 1

    def test_filter_by_file_type(self, retriever) -> None:
        results = retriever.retrieve("训练", k=10, file_types=["py"])
        assert all(d.metadata["file_type"] == "py" for d in results)

    def test_min_score_filters(self, retriever) -> None:
        """阈值设到极高时应该一条都不返回。"""
        assert retriever.retrieve("机器学习", k=5, min_score=0.99) == []

    def test_max_per_source(self, retriever) -> None:
        """同一个 PDF 有 2 页，限制每源 1 条时只能出 1 条。"""
        results = retriever.retrieve_with_scores(
            "学习", k=5, max_per_source=1
        )
        sources = [r.source_path for r in results]
        assert len(sources) == len(set(sources))


# ============================================================
# RetrievalResult
# ============================================================
class TestRetrievalResult:
    def _make(self, **meta) -> RetrievalResult:
        base = {"source_name": "a.pdf", "source_path": "/a.pdf", "page": 3}
        base.update(meta)
        return RetrievalResult(
            document=Document(page_content="内容", metadata=base),
            score=0.8,
            rank=1,
        )

    def test_citation_with_page(self) -> None:
        r = self._make(total_pages=10)
        assert r.citation() == "a.pdf 第 3 页"

    def test_citation_with_lines(self) -> None:
        """有行号时优先显示行号（比页码精确）。"""
        r = self._make(line_start=10, line_end=25)
        assert r.citation() == "a.pdf 第 10-25 行"

    def test_citation_single_page_file(self) -> None:
        """单页文件不显示"第 1 页"（是噪音）。"""
        r = self._make(total_pages=1)
        assert r.citation() == "a.pdf"

    def test_properties(self) -> None:
        r = self._make()
        assert r.source_name == "a.pdf"
        assert r.page == 3
        assert r.content == "内容"


# ============================================================
# 上下文拼装
# ============================================================
class TestBuildContext:
    def test_numbered_blocks(self, retriever) -> None:
        context, used = retriever.build_context("机器学习", k=2)

        assert "[1]" in context
        assert len(used) <= 2

    def test_respects_max_chars(self, retriever) -> None:
        context, used = retriever.build_context("学习", k=4, max_chars=60)

        assert len(used) >= 1
        # 允许第一块超限（至少要给一块），但不能塞满 4 块
        assert len(used) < 4

    def test_empty_when_no_hits(self, hash_llm_settings, paths, fake_embedder) -> None:
        empty = Retriever("nobody", hash_llm_settings, paths, embedder=fake_embedder)
        context, used = empty.build_context("查询")
        assert context == ""
        assert used == []


# ============================================================
# 混合检索
# ============================================================
class TestHybridRetriever:
    def test_finds_exact_keyword(
        self, hash_llm_settings, paths, fake_embedder, seeded
    ) -> None:
        hybrid = HybridRetriever(
            COURSE_ID, hash_llm_settings, paths, embedder=fake_embedder
        )
        results = hybrid.retrieve("train_model", k=3)

        assert len(results) >= 1
        assert any("train_model" in d.page_content for d in results)

    def test_rrf_score_attached(
        self, hash_llm_settings, paths, fake_embedder, seeded
    ) -> None:
        hybrid = HybridRetriever(
            COURSE_ID, hash_llm_settings, paths, embedder=fake_embedder
        )
        results = hybrid.retrieve("机器学习", k=2)
        assert all("rrf_score" in d.metadata for d in results)

    def test_no_duplicates(
        self, hash_llm_settings, paths, fake_embedder, seeded
    ) -> None:
        """两路都召回的文档只能出现一次。"""
        hybrid = HybridRetriever(
            COURSE_ID, hash_llm_settings, paths, embedder=fake_embedder
        )
        results = hybrid.retrieve("机器学习", k=5)
        ids = [d.metadata["chunk_id"] for d in results]
        assert len(ids) == len(set(ids))

    def test_empty_store(self, hash_llm_settings, paths, fake_embedder) -> None:
        hybrid = HybridRetriever(
            "nobody", hash_llm_settings, paths, embedder=fake_embedder
        )
        assert hybrid.retrieve("查询") == []
```

---

## 9. 源文件解析（Source Resolver）

### 9.1 目标

根据 chunk_id 找到**原始文件** + **行号/页码**，用于"跳转到源文件"功能。

### 9.2 文件内容指引

#### `src/private_teacher/rag/source_resolver.py`

```python
from pathlib import Path
from dataclasses import dataclass

@dataclass
class SourceLocation:
    file_path: Path
    page: int | None
    line_start: int | None
    line_end: int | None
    snippet: str  # 该 chunk 的内容预览

class SourceResolver:
    def __init__(self, course_id: str, paths: PathSettings):
        self.course_id = course_id
        self.data_dir = paths.data_dir

    def resolve(self, chunk: Document) -> SourceLocation:
        meta = chunk.metadata
        return SourceLocation(
            file_path=Path(meta["source_path"]),
            page=meta.get("page"),
            line_start=meta.get("line_start"),
            line_end=meta.get("line_end"),
            snippet=chunk.page_content[:200] + "...",
        )

    def read_snippet(self, location: SourceLocation, context_lines: int = 3) -> str:
        """读源文件对应行/页 + 上下 N 行。"""
        if location.line_start:
            lines = location.file_path.read_text().splitlines()
            start = max(0, location.line_start - 1 - context_lines)
            end = min(len(lines), location.line_end + context_lines)
            return "\n".join(lines[start:end])
        return location.snippet
```

### 9.3 测试代码

**tests/unit/test_source_resolver.py**

```python
def test_resolve_from_chunk(tmp_path):
    f = tmp_path / "test.py"
    f.write_text("line1\nline2\nline3\nline4\nline5")
    chunk = Document(
        page_content="line3\nline4",
        metadata={"source_path": str(f), "page": 1, "line_start": 3, "line_end": 4}
    )
    resolver = SourceResolver("c1", PathSettings(data_dir=tmp_path))
    loc = resolver.resolve(chunk)
    assert loc.line_start == 3
    assert loc.line_end == 4
    snippet = resolver.read_snippet(loc, context_lines=1)
    assert "line2" in snippet  # 上下文
    assert "line5" in snippet
```

### 9.4 完整代码实现

> 溯源是这个项目区别于"普通聊天机器人"的关键能力：
> **每一句回答都能点回原始课件的具体位置。**

#### 9.4.1 `src/private_teacher/rag/source_resolver.py`

```python
"""
源文件溯源
==========

输入：一个检索到的 chunk
输出：它来自哪个文件、第几页、第几行，以及能不能真的打开

【三种精度】
  行级（代码/txt/md）：line_start ~ line_end，最精确，可以高亮具体几行
  页级（PDF/PPTX）  ：page，能跳到某一页
  文件级            ：只能定位到文件（兜底）

【必须处理的现实问题】
  文件可能已经被用户移动、改名、删除。
  这时不能抛异常让整个页面崩掉 —— 要优雅降级成"文件已移除"提示，
  并且仍然把 chunk 的内容展示出来（内容存在向量库里，不依赖源文件）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document
from loguru import logger

from private_teacher.config import PathSettings


@dataclass(slots=True)
class SourceLocation:
    """一个 chunk 在源文件中的位置。"""

    file_path: Path
    source_name: str
    file_type: str
    page: int | None
    total_pages: int | None
    line_start: int | None
    line_end: int | None
    snippet: str  # chunk 内容预览
    exists: bool  # 源文件当前是否还在磁盘上

    # ---------- 展示辅助 ----------
    @property
    def has_lines(self) -> bool:
        return self.line_start is not None

    @property
    def has_pages(self) -> bool:
        return self.page is not None and (self.total_pages or 1) > 1

    def label(self) -> str:
        """一行式位置描述，UI 的标题栏用。

        例："机器学习.pdf · 第 12 页" / "train.py · 第 30-55 行"
        """
        parts = [self.source_name]

        if self.has_lines:
            if self.line_end and self.line_end != self.line_start:
                parts.append(f"第 {self.line_start}-{self.line_end} 行")
            else:
                parts.append(f"第 {self.line_start} 行")
        elif self.has_pages:
            parts.append(f"第 {self.page}/{self.total_pages} 页")

        if not self.exists:
            parts.append("⚠️ 文件已移除")

        return " · ".join(parts)

    def uri(self) -> str:
        """生成可点击的定位链接。

        - PDF：file:///path/to/a.pdf#page=12
          （Chrome / Edge / Preview 都支持 #page= 锚点直接跳页）
        - 其他：file:///path/to/a.py
          （编辑器链接如 vscode://file/path:line 依赖用户装了 VSCode，
            不作为默认行为，留给 UI 层自己决定）
        """
        if not self.exists:
            return ""

        # as_uri() 会正确处理空格、中文、Windows 盘符等，
        # 手写 "file://" + str(path) 在这些情况下都会出错
        base = self.file_path.as_uri()

        if self.file_type == "pdf" and self.page:
            return f"{base}#page={self.page}"
        return base


class SourceResolver:
    """把 chunk 映射回源文件位置。"""

    def __init__(self, course_id: str, paths: PathSettings) -> None:
        self.course_id = course_id
        self.paths = paths

    # ============================================================
    # 解析
    # ============================================================
    def resolve(self, chunk: Document, snippet_chars: int = 200) -> SourceLocation:
        """从 chunk 的 metadata 还原位置信息。

        Args:
            chunk: 检索结果
            snippet_chars: 内容预览截断长度

        Returns:
            SourceLocation（即使文件已丢失也会返回，用 exists 标记）
        """
        meta = chunk.metadata

        raw_path = str(meta.get("source_path", ""))
        file_path = Path(raw_path) if raw_path else Path()

        # ---------- 容错查找 ----------
        # 文件可能因为项目目录整体搬家而找不到，
        # 试着在当前课件目录下按文件名再找一次
        exists = bool(raw_path) and file_path.exists()
        if not exists and raw_path:
            recovered = self._try_recover(file_path)
            if recovered is not None:
                file_path = recovered
                exists = True

        content = chunk.page_content
        snippet = (
            content
            if len(content) <= snippet_chars
            else content[:snippet_chars].rstrip() + "..."
        )

        return SourceLocation(
            file_path=file_path,
            source_name=str(meta.get("source_name") or file_path.name or "未知文件"),
            file_type=str(meta.get("file_type", "")),
            # _as_int：metadata 从 Chroma 取回来可能是 str，统一转 int
            page=_as_int(meta.get("page")),
            total_pages=_as_int(meta.get("total_pages")),
            line_start=_as_int(meta.get("line_start")),
            line_end=_as_int(meta.get("line_end")),
            snippet=snippet,
            exists=exists,
        )

    def resolve_many(self, chunks: list[Document]) -> list[SourceLocation]:
        """批量解析。"""
        return [self.resolve(c) for c in chunks]

    def _try_recover(self, missing: Path) -> Path | None:
        """按文件名在本课程的课件目录里重新找一遍。

        场景：用户把整个项目文件夹从 D:\\ 挪到了 E:\\，
        绝对路径全失效，但文件其实都还在。
        """
        course_dir = self.paths.course_dir(self.course_id)
        if not course_dir.exists() or not missing.name:
            return None

        # next(iterator, None)：取第一个匹配，没有就返回 None
        found = next(course_dir.rglob(missing.name), None)
        if found is not None:
            logger.debug(f"源文件路径已失效，按文件名找回：{found}")
        return found

    # ============================================================
    # 读取源文件片段
    # ============================================================
    def read_snippet(
        self,
        location: SourceLocation,
        context_lines: int = 3,
    ) -> str:
        """读源文件对应位置的内容（含上下文）。

        Args:
            location: resolve() 的结果
            context_lines: 上下各多读几行

        Returns:
            源文件真实内容；文件不可读时退回 location.snippet

        为什么要"重新读源文件"，而不是直接用 chunk 内容？
          - chunk 是切分后的片段，边界可能在句子中间
          - 从源文件读能给出完整上下文，用户体验好得多
          - 还能验证"向量库里的内容和源文件是否还一致"
        """
        # 没有行号信息（PDF/PPTX）→ 只能用 chunk 内容
        if not location.exists or not location.has_lines:
            return location.snippet

        try:
            from private_teacher.loaders.text_loader import read_text_smart

            lines = read_text_smart(location.file_path).splitlines()
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"读取源文件失败 {location.file_path}: {exc}")
            return location.snippet

        # ---------- 行号 → 下标 ----------
        # 行号从 1 开始，列表下标从 0 开始，所以要 -1
        start = max(0, (location.line_start or 1) - 1 - context_lines)
        end = min(len(lines), (location.line_end or location.line_start or 1) + context_lines)

        return "\n".join(lines[start:end])

    def read_numbered_snippet(
        self,
        location: SourceLocation,
        context_lines: int = 3,
    ) -> str:
        """带行号的片段，命中行用 ▶ 标出。

        输出示例：
             28 | def train():
             29 |     model = build()
          ▶  30 |     model.fit(X, y)
          ▶  31 |     return model
             32 |
        """
        if not location.exists or not location.has_lines:
            return location.snippet

        try:
            from private_teacher.loaders.text_loader import read_text_smart

            lines = read_text_smart(location.file_path).splitlines()
        except Exception:  # noqa: BLE001
            return location.snippet

        start = max(0, (location.line_start or 1) - 1 - context_lines)
        end = min(
            len(lines),
            (location.line_end or location.line_start or 1) + context_lines,
        )

        out: list[str] = []
        for idx in range(start, end):
            line_no = idx + 1
            hit = (
                location.line_start is not None
                and location.line_start <= line_no <= (location.line_end or location.line_start)
            )
            marker = "▶" if hit else " "
            # >4 是右对齐到 4 位宽，让行号列整齐
            out.append(f"{marker} {line_no:>4} | {lines[idx]}")

        return "\n".join(out)

    # ============================================================
    # 分组
    # ============================================================
    def group_by_file(
        self,
        chunks: list[Document],
    ) -> dict[str, list[SourceLocation]]:
        """按源文件分组。

        用途：检索结果按文件折叠展示，
        "机器学习.pdf（3 处匹配）" 比平铺 3 条更容易读。
        """
        grouped: dict[str, list[SourceLocation]] = {}
        for loc in self.resolve_many(chunks):
            grouped.setdefault(loc.source_name, []).append(loc)

        # 每组内部按页码/行号排序，符合阅读顺序
        for locations in grouped.values():
            locations.sort(key=lambda loc: (loc.page or 0, loc.line_start or 0))

        return grouped


def _as_int(value: object) -> int | None:
    """安全地把 metadata 里的值转成 int。

    为什么需要这个？
      Chroma 存取过程中，数字类型可能变成 str（尤其是从旧版本库读出来时）。
      直接 int(value) 遇到 None 或 "" 会抛异常，
      而 metadata 缺字段是完全正常的情况（PDF 就没有行号）。
    """
    if value is None or value == "":
        return None
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
```

#### 9.4.2 `tests/unit/test_source_resolver.py`（完整版）

```python
"""source_resolver 单元测试。"""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.documents import Document

from private_teacher.config import PathSettings
from private_teacher.rag.source_resolver import SourceResolver, _as_int


@pytest.fixture
def paths(tmp_path: Path) -> PathSettings:
    return PathSettings(data_dir=tmp_path / "data")


@pytest.fixture
def resolver(paths) -> SourceResolver:
    return SourceResolver("c1", paths)


def make_chunk(content: str = "内容", **meta) -> Document:
    base = {
        "source_path": "/nowhere/a.txt",
        "source_name": "a.txt",
        "file_type": "txt",
        "page": 1,
        "total_pages": 1,
    }
    base.update(meta)
    return Document(page_content=content, metadata=base)


# ============================================================
# 解析
# ============================================================
class TestResolve:
    def test_resolve_lines(self, tmp_path, resolver) -> None:
        f = tmp_path / "test.py"
        f.write_text("line1\nline2\nline3\nline4\nline5", encoding="utf-8")

        loc = resolver.resolve(
            make_chunk(
                "line3\nline4",
                source_path=str(f),
                source_name="test.py",
                file_type="py",
                line_start=3,
                line_end=4,
            )
        )

        assert loc.line_start == 3
        assert loc.line_end == 4
        assert loc.exists is True
        assert loc.has_lines is True

    def test_resolve_page(self, resolver) -> None:
        loc = resolver.resolve(
            make_chunk(file_type="pdf", page=12, total_pages=50)
        )
        assert loc.page == 12
        assert loc.has_pages is True
        assert loc.has_lines is False

    def test_missing_file_marked(self, resolver) -> None:
        """文件不存在时不能抛异常，要用 exists=False 标记。"""
        loc = resolver.resolve(make_chunk(source_path="/definitely/not/here.txt"))
        assert loc.exists is False
        assert "文件已移除" in loc.label()

    def test_snippet_truncated(self, resolver) -> None:
        loc = resolver.resolve(make_chunk("x" * 500), snippet_chars=100)
        assert loc.snippet.endswith("...")
        assert len(loc.snippet) <= 103

    def test_snippet_not_truncated_when_short(self, resolver) -> None:
        loc = resolver.resolve(make_chunk("短内容"))
        assert loc.snippet == "短内容"

    def test_string_numbers_converted(self, resolver) -> None:
        """metadata 里存成字符串的数字也要能正确解析。"""
        loc = resolver.resolve(make_chunk(page="7", line_start="3"))
        assert loc.page == 7
        assert loc.line_start == 3

    def test_recover_moved_file(self, tmp_path, paths, resolver) -> None:
        """⭐ 路径失效但文件还在课件目录里 → 应该按文件名找回。"""
        course_dir = paths.course_dir("c1") / "main"
        course_dir.mkdir(parents=True)
        real = course_dir / "moved.txt"
        real.write_text("内容还在", encoding="utf-8")

        loc = resolver.resolve(
            make_chunk(source_path="/old/path/moved.txt", source_name="moved.txt")
        )

        assert loc.exists is True
        assert loc.file_path == real

    def test_resolve_many(self, resolver) -> None:
        chunks = [make_chunk(), make_chunk(), make_chunk()]
        assert len(resolver.resolve_many(chunks)) == 3


# ============================================================
# 标签与链接
# ============================================================
class TestLabelAndUri:
    def test_label_with_lines(self, tmp_path, resolver) -> None:
        f = tmp_path / "a.py"
        f.write_text("x", encoding="utf-8")

        loc = resolver.resolve(
            make_chunk(
                source_path=str(f), source_name="a.py", line_start=10, line_end=25
            )
        )
        assert loc.label() == "a.py · 第 10-25 行"

    def test_label_single_line(self, tmp_path, resolver) -> None:
        f = tmp_path / "a.py"
        f.write_text("x", encoding="utf-8")

        loc = resolver.resolve(
            make_chunk(
                source_path=str(f), source_name="a.py", line_start=7, line_end=7
            )
        )
        assert loc.label() == "a.py · 第 7 行"

    def test_label_with_page(self, resolver) -> None:
        loc = resolver.resolve(
            make_chunk(source_name="ml.pdf", file_type="pdf", page=3, total_pages=20)
        )
        assert "第 3/20 页" in loc.label()

    def test_uri_pdf_has_page_anchor(self, tmp_path, resolver) -> None:
        """PDF 链接要带 #page= 锚点，浏览器才能直接跳页。"""
        f = tmp_path / "a.pdf"
        f.write_bytes(b"%PDF-1.4")

        loc = resolver.resolve(
            make_chunk(
                source_path=str(f),
                source_name="a.pdf",
                file_type="pdf",
                page=5,
                total_pages=10,
            )
        )
        assert loc.uri().endswith("#page=5")
        assert loc.uri().startswith("file://")

    def test_uri_empty_when_missing(self, resolver) -> None:
        loc = resolver.resolve(make_chunk(source_path="/gone.txt"))
        assert loc.uri() == ""


# ============================================================
# 读取源文件
# ============================================================
class TestReadSnippet:
    def test_reads_with_context(self, tmp_path, resolver) -> None:
        f = tmp_path / "test.py"
        f.write_text("line1\nline2\nline3\nline4\nline5", encoding="utf-8")

        loc = resolver.resolve(
            make_chunk(
                "line3\nline4",
                source_path=str(f),
                file_type="py",
                line_start=3,
                line_end=4,
            )
        )
        snippet = resolver.read_snippet(loc, context_lines=1)

        assert "line2" in snippet  # 上文
        assert "line3" in snippet
        assert "line5" in snippet  # 下文
        assert "line1" not in snippet  # 超出上下文范围

    def test_context_clamped_at_boundaries(self, tmp_path, resolver) -> None:
        """上下文超出文件首尾时不能报错（切片越界）。"""
        f = tmp_path / "a.txt"
        f.write_text("only line", encoding="utf-8")

        loc = resolver.resolve(
            make_chunk(source_path=str(f), line_start=1, line_end=1)
        )
        assert resolver.read_snippet(loc, context_lines=100) == "only line"

    def test_falls_back_for_pdf(self, resolver) -> None:
        """PDF 没有行号，直接返回 chunk 预览。"""
        loc = resolver.resolve(
            make_chunk("PDF 的内容", file_type="pdf", page=3, total_pages=10)
        )
        assert resolver.read_snippet(loc) == "PDF 的内容"

    def test_falls_back_when_missing(self, resolver) -> None:
        loc = resolver.resolve(
            make_chunk("原始内容", source_path="/gone.txt", line_start=1, line_end=2)
        )
        assert resolver.read_snippet(loc) == "原始内容"

    def test_numbered_snippet_marks_hits(self, tmp_path, resolver) -> None:
        f = tmp_path / "a.py"
        f.write_text("\n".join(f"line{i}" for i in range(1, 11)), encoding="utf-8")

        loc = resolver.resolve(
            make_chunk(source_path=str(f), line_start=5, line_end=6)
        )
        out = resolver.read_numbered_snippet(loc, context_lines=1)

        lines = out.splitlines()
        assert lines[0].startswith("  ")  # 第 4 行：上下文
        assert any(line.startswith("▶") for line in lines)
        assert "   5 |" in out


# ============================================================
# 分组 & 工具
# ============================================================
class TestGrouping:
    def test_group_by_file(self, resolver) -> None:
        chunks = [
            make_chunk(source_name="a.pdf", page=3),
            make_chunk(source_name="a.pdf", page=1),
            make_chunk(source_name="b.md", page=1),
        ]
        grouped = resolver.group_by_file(chunks)

        assert set(grouped) == {"a.pdf", "b.md"}
        assert len(grouped["a.pdf"]) == 2
        # 组内按页码排序
        assert [loc.page for loc in grouped["a.pdf"]] == [1, 3]


class TestAsInt:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (5, 5),
            ("5", 5),
            (None, None),
            ("", None),
            ("abc", None),
            (3.7, 3),
        ],
    )
    def test_conversion(self, value, expected) -> None:
        assert _as_int(value) == expected
```

---

## 10. 业务编排（Services）

### 10.1 目标

把"上传 / 删除 / 重建索引 / 检索"封装成 service，UI 层只调 service。

### 10.2 文件内容指引

#### `src/private_teacher/services/course_service.py`

```python
class CourseService:
    """课程管理：上传 / 删除 / 重建索引。"""

    def __init__(self, paths: PathSettings):
        self.paths = paths

    def create_course(self, conn, name: str, description: str = "") -> Course:
        course = create(conn, name, description)
        # 创建物理目录
        (self.paths.course_dir(course.id) / "main").mkdir(parents=True, exist_ok=True)
        (self.paths.course_dir(course.id) / "auxiliary").mkdir(parents=True, exist_ok=True)
        return course

    def upload_document(
        self, conn, course_id: str, source_path: Path, category: str
    ) -> Document:
        """上传文档：复制到 data/courseware/{course_id}/{category}/"""
        # 1. 计算 sha256（去重）
        sha = compute_sha256(source_path)

        # 2. 检查重复
        if get_by_sha256(conn, course_id, sha):
            raise FileExistsError(f"文件已存在（sha256={sha[:8]}...）")

        # 3. 复制文件
        dest = self.paths.course_dir(course_id) / category / source_path.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, dest)

        # 4. 写库
        return create_doc(
            conn, course_id, dest, category, sha, dest.stat().st_size
        )

    def delete_course(self, conn, course_id: str):
        """删除课程：删文件 + 删库 + 删向量库。"""
        delete(conn, course_id)
        shutil.rmtree(self.paths.course_dir(course_id), ignore_errors=True)
        VectorStoreManager(course_id, self.paths.data_dir).delete_collection()
```

#### `src/private_teacher/services/kb_service.py`

```python
class KBService:
    """知识库服务：构建 / 检索。"""

    def __init__(self, llm: LLMSettings, paths: PathSettings):
        self.llm = llm
        self.paths = paths

    def build_kb(self, conn, course_id: str, incremental: bool = True) -> dict:
        indexer = Indexer(course_id, self.llm, self.paths)
        if incremental:
            return indexer.build_incremental(conn)
        return indexer.build_full(conn)

    def search(self, course_id: str, query: str, k: int = 5):
        retriever = Retriever(course_id, self.llm, self.paths)
        return retriever.retrieve(query, k=k)

    def resolve_sources(self, course_id: str, chunks: list[Document]):
        resolver = SourceResolver(course_id, self.paths)
        return [resolver.resolve(c) for c in chunks]
```

### 10.3 测试代码

**tests/unit/test_services.py**

```python
class TestCourseService:
    def test_upload_dedup_by_sha256(self, tmp_path, tmp_db):
        f = tmp_path / "a.txt"
        f.write_text("hello")
        service = CourseService(PathSettings(data_dir=tmp_path))
        course = service.create_course(tmp_db, "Test")

        service.upload_document(tmp_db, course.id, f, "main")
        with pytest.raises(FileExistsError):
            service.upload_document(tmp_db, course.id, f, "main")

    def test_delete_course_removes_everything(self, tmp_path, tmp_db):
        ...
```

### 10.4 完整代码实现

> Service 层是"**唯一知道全貌**"的一层：它同时操作数据库、文件系统和向量库，
> 并保证三者状态一致。UI 层只调 service，绝不直接碰 repo / store。

#### 10.4.1 `src/private_teacher/utils/exceptions.py` 追加

```python
# 追加到文件末尾

class DuplicateDocumentError(PrivateTeacherError, FileExistsError):
    """重复上传同一个文件（sha256 已存在）。

    ⚠️ 注意这里的**多重继承**：
      同时继承 PrivateTeacherError 和内置的 FileExistsError。

    为什么这么做？
      - 继承 PrivateTeacherError：上层可以用 `except PrivateTeacherError` 统一兜住
      - 继承 FileExistsError：语义天然吻合，且调用方用
        `except FileExistsError` 也能捕获（对 UI 代码更直观）
      两个基类都直接/间接继承自 Exception，MRO 能正常解析，不会冲突。
    """

    pass


class ServiceError(PrivateTeacherError):
    """业务编排层的错误（参数非法、状态冲突等）。"""

    pass
```

#### 10.4.2 `src/private_teacher/services/course_service.py`

```python
"""
课程服务
========

管理"课程 + 课件文件"的完整生命周期。

【核心职责：保持三处状态一致】
    ① SQLite 记录（documents 表）
    ② 物理文件（data/courseware/{course_id}/{category}/）
    ③ 向量库（data/knowledge_bases/{course_id}/）

任何一处漏掉，就会出现"检索到了但打不开"或者"删了还能搜到"这类灵异现象。
"""

from __future__ import annotations

import shutil
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from private_teacher.config import PathSettings
from private_teacher.loaders import is_supported
from private_teacher.rag.vector_store import VectorStoreManager
from private_teacher.storage import course_repo, document_repo
from private_teacher.storage.models import Course, Document
from private_teacher.utils.exceptions import (
    DocumentLoadError,
    DuplicateDocumentError,
    ServiceError,
)
from private_teacher.utils.hashing import compute_sha256, compute_sha256_bytes


@dataclass(slots=True)
class CourseStats:
    """课程概览，UI 卡片用。"""

    course: Course
    total_documents: int
    main_documents: int
    auxiliary_documents: int
    indexed_documents: int
    total_size: int  # 字节

    @property
    def total_size_mb(self) -> float:
        return self.total_size / 1024 / 1024

    @property
    def index_progress(self) -> float:
        """已索引比例 0.0~1.0（进度条用）。"""
        if self.total_documents == 0:
            return 0.0
        return self.indexed_documents / self.total_documents


class CourseService:
    """课程与课件管理。"""

    def __init__(self, paths: PathSettings) -> None:
        self.paths = paths

    # ============================================================
    # 课程 CRUD
    # ============================================================
    def create_course(
        self,
        conn: sqlite3.Connection,
        name: str,
        description: str = "",
    ) -> Course:
        """创建课程，并建好物理目录。

        Raises:
            ServiceError: 课程名为空
        """
        name = name.strip()
        if not name:
            raise ServiceError("课程名不能为空")

        course = course_repo.create(conn, name, description)

        # 立刻建目录：晚建不如早建，
        # 否则上传时才发现建不了（权限/磁盘满）会让用户丢文件
        for category in ("main", "auxiliary"):
            (self.paths.course_dir(course.id) / category).mkdir(
                parents=True, exist_ok=True
            )

        logger.info(f"课程已创建：{name}（{course.id}）")
        return course

    def list_courses(self, conn: sqlite3.Connection) -> list[Course]:
        return course_repo.list_all(conn)

    def get_course(self, conn: sqlite3.Connection, course_id: str) -> Course:
        """取课程，不存在就抛异常（UI 上是"页面不该出现"的状态）。"""
        course = course_repo.get_by_id(conn, course_id)
        if course is None:
            raise ServiceError("课程不存在", course_id=course_id)
        return course

    def update_course(
        self,
        conn: sqlite3.Connection,
        course_id: str,
        name: str | None = None,
        description: str | None = None,
    ) -> Course:
        updated = course_repo.update(conn, course_id, name, description)
        if updated is None:
            raise ServiceError("课程不存在", course_id=course_id)
        return updated

    def delete_course(self, conn: sqlite3.Connection, course_id: str) -> None:
        """彻底删除课程：数据库 + 文件 + 向量库。

        ⚠️ 删除顺序很讲究：**先删外部资源，最后删数据库记录**。
           如果先删数据库，中途崩溃就再也找不到该删哪些文件了（孤儿数据）。
           反过来，先删文件后崩溃，数据库记录还在，用户可以重试删除。
           这个原则叫"先清理难恢复的引用，最后清理索引"。
        """
        # ① 向量库
        VectorStoreManager(course_id, self.paths.data_dir).delete_collection()

        # ② 知识库目录（chroma + manifest）
        kb_dir = self.paths.data_dir / "knowledge_bases" / course_id
        shutil.rmtree(kb_dir, ignore_errors=True)

        # ③ 课件文件
        shutil.rmtree(self.paths.course_dir(course_id), ignore_errors=True)

        # ④ 数据库（documents 靠 CASCADE 自动删）
        course_repo.delete(conn, course_id)

        logger.info(f"课程已彻底删除：{course_id}")

    # ============================================================
    # 文档上传
    # ============================================================
    def upload_document(
        self,
        conn: sqlite3.Connection,
        course_id: str,
        source_path: Path,
        category: str = "main",
    ) -> Document:
        """把一个已存在于磁盘的文件导入课程。

        流程：校验 → 算哈希 → 查重 → 复制 → 登记

        Args:
            source_path: 源文件（会被复制，不是移动）
            category: "main" 或 "auxiliary"

        Returns:
            登记好的 Document

        Raises:
            DocumentLoadError: 文件不存在 / 格式不支持
            DuplicateDocumentError: 内容重复（同时也是 FileExistsError）
            ServiceError: 课程不存在 / category 非法
        """
        # ---------- 1. 前置校验 ----------
        self.get_course(conn, course_id)  # 课程必须存在

        if category not in ("main", "auxiliary"):
            raise ServiceError(f"category 必须是 main/auxiliary，收到 {category!r}")

        if not source_path.exists() or not source_path.is_file():
            raise DocumentLoadError("文件不存在", path=str(source_path))

        if not is_supported(source_path):
            raise DocumentLoadError(
                f"不支持的格式: {source_path.suffix}",
                path=str(source_path),
            )

        # ---------- 2. 内容去重 ----------
        # 先算哈希再复制：重复文件根本不该占磁盘
        sha = compute_sha256(source_path)
        existing = document_repo.get_by_sha256(conn, course_id, sha)
        if existing is not None:
            raise DuplicateDocumentError(
                f"文件内容重复（已存在：{existing.path.name}）",
                sha256=sha[:12],
                existing_path=str(existing.path),
            )

        # ---------- 3. 复制到课件目录 ----------
        dest_dir = self.paths.course_dir(course_id) / category
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = self._unique_dest(dest_dir, source_path.name)

        # copy2 会连同修改时间/权限一起复制（copy 只复制内容）
        shutil.copy2(source_path, dest)

        # ---------- 4. 登记 ----------
        try:
            doc = document_repo.create(
                conn, course_id, dest, category, sha, dest.stat().st_size
            )
        except Exception:
            # 写库失败要把刚复制的文件删掉，否则留下无主的孤儿文件
            dest.unlink(missing_ok=True)
            raise

        course_repo.touch(conn, course_id)  # 把课程顶到列表最前
        logger.info(f"上传成功：{dest.name} → {course_id}/{category}")
        return doc

    def upload_bytes(
        self,
        conn: sqlite3.Connection,
        course_id: str,
        filename: str,
        data: bytes,
        category: str = "main",
    ) -> Document:
        """直接从内存字节流上传（Streamlit file_uploader 用）。

        比"先写临时文件再调 upload_document"好在哪？
          - 重复文件根本不落盘（先用字节算哈希查重）
          - 少一次磁盘往返
        """
        self.get_course(conn, course_id)

        if category not in ("main", "auxiliary"):
            raise ServiceError(f"category 必须是 main/auxiliary，收到 {category!r}")

        if not is_supported(Path(filename)):
            raise DocumentLoadError(f"不支持的格式: {Path(filename).suffix}")

        sha = compute_sha256_bytes(data)
        existing = document_repo.get_by_sha256(conn, course_id, sha)
        if existing is not None:
            raise DuplicateDocumentError(
                f"文件内容重复（已存在：{existing.path.name}）",
                sha256=sha[:12],
            )

        dest_dir = self.paths.course_dir(course_id) / category
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = self._unique_dest(dest_dir, filename)
        dest.write_bytes(data)

        try:
            doc = document_repo.create(
                conn, course_id, dest, category, sha, len(data)
            )
        except Exception:
            dest.unlink(missing_ok=True)
            raise

        course_repo.touch(conn, course_id)
        return doc

    @staticmethod
    def _unique_dest(dest_dir: Path, filename: str) -> Path:
        """避免同名文件互相覆盖：a.pdf → a_1.pdf → a_2.pdf

        为什么会同名但不重复？
          内容不同、文件名相同（比如两个学期的"讲义.pdf"）。
          sha256 查重放行了，但直接复制会覆盖掉前一个 —— 必须改名。
        """
        dest = dest_dir / filename
        if not dest.exists():
            return dest

        stem, suffix = Path(filename).stem, Path(filename).suffix
        for i in range(1, 1000):
            candidate = dest_dir / f"{stem}_{i}{suffix}"
            if not candidate.exists():
                return candidate

        raise ServiceError(f"同名文件过多，无法生成唯一文件名: {filename}")

    # ============================================================
    # 文档删除与查询
    # ============================================================
    def delete_document(
        self,
        conn: sqlite3.Connection,
        document_id: str,
    ) -> None:
        """删除一个课件：向量 → 文件 → 数据库。"""
        doc = document_repo.get_by_id(conn, document_id)
        if doc is None:
            raise ServiceError("文档不存在", document_id=document_id)

        # ① 向量
        VectorStoreManager(doc.course_id, self.paths.data_dir).delete_by_source(
            doc.path
        )

        # ② 从 manifest 里移除（否则增量索引会以为它还在）
        from private_teacher.rag.manifest import IndexManifest

        manifest = IndexManifest.load(doc.course_id, self.paths.data_dir)
        manifest.remove(str(doc.path.resolve()))
        manifest.save(self.paths.data_dir)

        # ③ 物理文件
        doc.path.unlink(missing_ok=True)

        # ④ 数据库
        document_repo.delete(conn, document_id)

        logger.info(f"文档已删除：{doc.path.name}")

    def list_documents(
        self,
        conn: sqlite3.Connection,
        course_id: str,
        category: str | None = None,
    ) -> list[Document]:
        return document_repo.list_by_course(conn, course_id, category)

    def get_stats(self, conn: sqlite3.Connection, course_id: str) -> CourseStats:
        """课程概览统计。"""
        course = self.get_course(conn, course_id)
        docs = document_repo.list_by_course(conn, course_id)

        return CourseStats(
            course=course,
            total_documents=len(docs),
            main_documents=sum(1 for d in docs if d.category == "main"),
            auxiliary_documents=sum(1 for d in docs if d.category == "auxiliary"),
            indexed_documents=sum(1 for d in docs if d.is_indexed),
            total_size=sum(d.size for d in docs),
        )

    def import_directory(
        self,
        conn: sqlite3.Connection,
        course_id: str,
        dir_path: Path,
        category: str = "main",
    ) -> tuple[list[Document], list[str]]:
        """批量导入一个目录。

        Returns:
            (成功导入的文档列表, 跳过原因列表)

        为什么返回"跳过原因"而不是抛异常？
          批量操作里"部分失败"是常态。用户需要的是一份报告，
          而不是"第 3 个文件出错所以全部回滚"。
        """
        from private_teacher.loaders import iter_loadable_files

        imported: list[Document] = []
        skipped: list[str] = []

        for file in iter_loadable_files(dir_path):
            try:
                imported.append(
                    self.upload_document(conn, course_id, file, category)
                )
            except DuplicateDocumentError:
                skipped.append(f"{file.name}: 内容重复")
            except (DocumentLoadError, ServiceError) as exc:
                skipped.append(f"{file.name}: {exc}")

        logger.info(f"目录导入完成：成功 {len(imported)}，跳过 {len(skipped)}")
        return imported, skipped
```

#### 10.4.3 `src/private_teacher/services/kb_service.py`

```python
"""
知识库服务
==========

对外提供三个动作：**建库 / 检索 / 溯源**。

它是 Phase 2 Agent 层唯一需要认识的入口 ——
Agent 只要会调 `kb.search_with_sources(...)`，就拥有了完整的 RAG 能力。
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Any

from loguru import logger

from private_teacher.config import LLMSettings, PathSettings
from private_teacher.rag.embeddings import Embedder, get_cached_embedder
from private_teacher.rag.indexer import Indexer, IndexResult
from private_teacher.rag.retriever import RetrievalResult, Retriever
from private_teacher.rag.source_resolver import SourceLocation, SourceResolver
from private_teacher.rag.splitters import ChunkerConfig


@dataclass(slots=True)
class SearchHit:
    """检索结果 + 溯源信息，UI 直接渲染这个对象。

    为什么要把 RetrievalResult 和 SourceLocation 打包在一起？
      → UI 每显示一条结果，都同时需要"内容 + 分数"和"来自哪里"。
        让 UI 自己去调两个方法再配对，既啰嗦又容易出错。
    """

    result: RetrievalResult
    location: SourceLocation

    @property
    def content(self) -> str:
        return self.result.content

    @property
    def score(self) -> float:
        return self.result.score

    @property
    def label(self) -> str:
        return self.location.label()


class KBService:
    """知识库服务。"""

    def __init__(
        self,
        llm: LLMSettings,
        paths: PathSettings,
        chunker_config: ChunkerConfig | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self.llm = llm
        self.paths = paths
        self.chunker_config = chunker_config or ChunkerConfig()
        # 允许注入 embedder（测试用）；否则用带缓存的工厂
        self._embedder = embedder or get_cached_embedder(llm)

    # ---------- 内部工厂 ----------
    # 为什么每次都新建 Indexer/Retriever，而不缓存成实例属性？
    #   → 它们很轻（真正重的 embedder 已经共享了），
    #     而且 KBService 可能同时服务多门课，缓存反而要处理 course_id 切换
    def _indexer(self, course_id: str) -> Indexer:
        return Indexer(
            course_id, self.llm, self.paths, self.chunker_config, self._embedder
        )

    def _retriever(self, course_id: str) -> Retriever:
        return Retriever(course_id, self.llm, self.paths, self._embedder)

    def _resolver(self, course_id: str) -> SourceResolver:
        return SourceResolver(course_id, self.paths)

    # ============================================================
    # 构建
    # ============================================================
    def build_kb(
        self,
        conn: sqlite3.Connection,
        course_id: str,
        incremental: bool = True,
        on_file: Any = None,
    ) -> IndexResult:
        """构建知识库。

        Args:
            incremental: True=增量（默认，快），False=全量重建
            on_file: 进度回调 (序号, 总数, 文件名)
        """
        indexer = self._indexer(course_id)

        if incremental:
            return indexer.build_incremental(conn, on_file=on_file)
        return indexer.build_full(conn, on_file=on_file)

    def stats(self, conn: sqlite3.Connection, course_id: str) -> dict[str, Any]:
        return self._indexer(course_id).stats(conn)

    def is_ready(self, course_id: str) -> bool:
        """知识库能不能用于检索。"""
        return self._retriever(course_id).is_ready()

    # ============================================================
    # 检索
    # ============================================================
    def search(
        self,
        course_id: str,
        query: str,
        k: int = 5,
        **kwargs: Any,
    ) -> list[RetrievalResult]:
        """检索（不含溯源）。"""
        return self._retriever(course_id).retrieve_with_scores(query, k, **kwargs)

    def search_with_sources(
        self,
        course_id: str,
        query: str,
        k: int = 5,
        **kwargs: Any,
    ) -> list[SearchHit]:
        """⭐ 检索 + 溯源，UI 和 Agent 的主入口。"""
        results = self.search(course_id, query, k, **kwargs)
        resolver = self._resolver(course_id)

        return [
            SearchHit(result=r, location=resolver.resolve(r.document))
            for r in results
        ]

    def resolve_sources(
        self,
        course_id: str,
        chunks: list[Any],
    ) -> list[SourceLocation]:
        """给一批 chunk 做溯源（兼容旧接口）。"""
        return self._resolver(course_id).resolve_many(chunks)

    # ============================================================
    # 给 LLM 用
    # ============================================================
    def build_context(
        self,
        course_id: str,
        query: str,
        k: int = 5,
        max_chars: int = 4000,
        **kwargs: Any,
    ) -> tuple[str, list[RetrievalResult]]:
        """检索并拼成带编号引用的上下文（Phase 2 Agent 会大量使用）。"""
        return self._retriever(course_id).build_context(
            query, k, max_chars, **kwargs
        )

    # ============================================================
    # 维护
    # ============================================================
    def clear_kb(self, conn: sqlite3.Connection, course_id: str) -> None:
        """清空知识库（保留课件文件本身）。"""
        indexer = self._indexer(course_id)
        indexer.store.reset()

        from private_teacher.rag.manifest import IndexManifest
        from private_teacher.storage import document_repo

        IndexManifest(course_id=course_id).save(self.paths.data_dir)
        document_repo.mark_all_unindexed(conn, course_id)

        logger.info(f"知识库已清空：{course_id}")
```

#### 10.4.4 `src/private_teacher/services/__init__.py`

```python
"""业务编排层。

分层原则回顾（从下到上）：
    storage / loaders / rag   基础能力，互不知道对方的业务含义
    services                  编排它们，保证跨模块状态一致
    ui                        只调 services，不碰下面任何一层

判断代码该放哪一层的简单方法：
  - 只操作数据库          → repo
  - 只处理文件/向量        → loaders / rag
  - 需要同时协调两个以上   → services
"""

from private_teacher.services.course_service import CourseService, CourseStats
from private_teacher.services.kb_service import KBService, SearchHit

__all__ = ["CourseService", "CourseStats", "KBService", "SearchHit"]
```

#### 10.4.5 `tests/unit/test_services.py`（完整版）

```python
"""services 层单元测试（全部离线）。"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from private_teacher.config import PathSettings
from private_teacher.services import CourseService, KBService
from private_teacher.storage import course_repo, document_repo
from private_teacher.utils.exceptions import (
    DocumentLoadError,
    DuplicateDocumentError,
    ServiceError,
)


@pytest.fixture
def paths(tmp_path: Path) -> PathSettings:
    return PathSettings(data_dir=tmp_path / "data")


@pytest.fixture
def service(paths) -> CourseService:
    return CourseService(paths)


@pytest.fixture
def kb(hash_llm_settings, paths, fake_embedder) -> KBService:
    return KBService(hash_llm_settings, paths, embedder=fake_embedder)


@pytest.fixture
def src_file(tmp_path: Path) -> Path:
    """一个待上传的源文件。"""
    f = tmp_path / "src" / "lecture.txt"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(
        "机器学习是人工智能的核心分支，研究如何从数据中学习规律。",
        encoding="utf-8",
    )
    return f


# ============================================================
# 课程 CRUD
# ============================================================
class TestCourseCrud:
    def test_create_makes_dirs(self, tmp_db, service, paths) -> None:
        course = service.create_course(tmp_db, "Python 入门")

        assert (paths.course_dir(course.id) / "main").is_dir()
        assert (paths.course_dir(course.id) / "auxiliary").is_dir()

    def test_empty_name_rejected(self, tmp_db, service) -> None:
        with pytest.raises(ServiceError, match="不能为空"):
            service.create_course(tmp_db, "   ")

    def test_get_missing_raises(self, tmp_db, service) -> None:
        with pytest.raises(ServiceError, match="不存在"):
            service.get_course(tmp_db, "nope")

    def test_update(self, tmp_db, service) -> None:
        course = service.create_course(tmp_db, "旧名字")
        updated = service.update_course(tmp_db, course.id, name="新名字")
        assert updated.name == "新名字"


# ============================================================
# 上传
# ============================================================
class TestUpload:
    def test_copies_into_course_dir(self, tmp_db, service, src_file, paths) -> None:
        course = service.create_course(tmp_db, "Test")

        doc = service.upload_document(tmp_db, course.id, src_file, "main")

        assert doc.path.exists()
        assert doc.path.parent == paths.course_dir(course.id) / "main"
        assert src_file.exists()  # 是复制不是移动，源文件还在

    def test_dedup_by_sha256(self, tmp_db, service, src_file) -> None:
        """⭐ 同内容文件不能重复上传。"""
        course = service.create_course(tmp_db, "Test")
        service.upload_document(tmp_db, course.id, src_file, "main")

        with pytest.raises(DuplicateDocumentError):
            service.upload_document(tmp_db, course.id, src_file, "main")

    def test_duplicate_is_also_fileexistserror(
        self, tmp_db, service, src_file
    ) -> None:
        """多重继承生效：用内置异常也能捕获。"""
        course = service.create_course(tmp_db, "Test")
        service.upload_document(tmp_db, course.id, src_file, "main")

        with pytest.raises(FileExistsError):
            service.upload_document(tmp_db, course.id, src_file, "main")

    def test_same_name_different_content_renamed(
        self, tmp_db, service, tmp_path
    ) -> None:
        """⭐ 同名但内容不同 → 自动改名，不能互相覆盖。"""
        course = service.create_course(tmp_db, "Test")

        a = tmp_path / "d1" / "note.txt"
        b = tmp_path / "d2" / "note.txt"
        a.parent.mkdir(parents=True)
        b.parent.mkdir(parents=True)
        a.write_text("第一份内容", encoding="utf-8")
        b.write_text("第二份完全不同的内容", encoding="utf-8")

        doc_a = service.upload_document(tmp_db, course.id, a)
        doc_b = service.upload_document(tmp_db, course.id, b)

        assert doc_a.path != doc_b.path
        assert doc_a.path.exists() and doc_b.path.exists()
        assert doc_b.path.name == "note_1.txt"

    def test_unsupported_format_rejected(self, tmp_db, service, tmp_path) -> None:
        course = service.create_course(tmp_db, "Test")
        bad = tmp_path / "a.exe"
        bad.write_text("x", encoding="utf-8")

        with pytest.raises(DocumentLoadError, match="不支持"):
            service.upload_document(tmp_db, course.id, bad)

    def test_missing_source_rejected(self, tmp_db, service, tmp_path) -> None:
        course = service.create_course(tmp_db, "Test")
        with pytest.raises(DocumentLoadError):
            service.upload_document(tmp_db, course.id, tmp_path / "nope.txt")

    def test_invalid_category(self, tmp_db, service, src_file) -> None:
        course = service.create_course(tmp_db, "Test")
        with pytest.raises(ServiceError, match="category"):
            service.upload_document(tmp_db, course.id, src_file, "bad")

    def test_upload_bytes(self, tmp_db, service) -> None:
        course = service.create_course(tmp_db, "Test")

        doc = service.upload_bytes(
            tmp_db, course.id, "note.md", "# 标题\n内容".encode(), "auxiliary"
        )

        assert doc.path.exists()
        assert doc.category == "auxiliary"

    def test_upload_bytes_dedup(self, tmp_db, service) -> None:
        course = service.create_course(tmp_db, "Test")
        data = b"same content"
        service.upload_bytes(tmp_db, course.id, "a.txt", data)

        with pytest.raises(DuplicateDocumentError):
            service.upload_bytes(tmp_db, course.id, "b.txt", data)

    def test_import_directory(self, tmp_db, service, tmp_path) -> None:
        course = service.create_course(tmp_db, "Test")
        src = tmp_path / "batch"
        src.mkdir()
        (src / "a.txt").write_text("内容甲", encoding="utf-8")
        (src / "b.md").write_text("内容乙", encoding="utf-8")
        (src / "c.exe").write_text("忽略", encoding="utf-8")

        imported, skipped = service.import_directory(tmp_db, course.id, src)

        assert len(imported) == 2  # .exe 在扫描阶段就被过滤了
        assert skipped == []


# ============================================================
# 删除
# ============================================================
class TestDelete:
    def test_delete_document_removes_everything(
        self, tmp_db, service, kb, src_file
    ) -> None:
        course = service.create_course(tmp_db, "Test")
        doc = service.upload_document(tmp_db, course.id, src_file)
        kb.build_kb(tmp_db, course.id)
        assert kb.is_ready(course.id)

        service.delete_document(tmp_db, doc.id)

        assert not doc.path.exists()  # 文件没了
        assert document_repo.get_by_id(tmp_db, doc.id) is None  # 记录没了
        assert kb.search(course.id, "机器学习") == []  # 向量也没了

    def test_delete_course_removes_everything(
        self, tmp_db, service, kb, src_file, paths
    ) -> None:
        """⭐ 删课程后：文件、数据库、向量库三处都要干净。"""
        course = service.create_course(tmp_db, "Test")
        service.upload_document(tmp_db, course.id, src_file)
        kb.build_kb(tmp_db, course.id)

        service.delete_course(tmp_db, course.id)

        assert course_repo.get_by_id(tmp_db, course.id) is None
        assert document_repo.list_by_course(tmp_db, course.id) == []
        assert not paths.course_dir(course.id).exists()
        assert not (paths.data_dir / "knowledge_bases" / course.id).exists()

    def test_delete_missing_document(self, tmp_db, service) -> None:
        with pytest.raises(ServiceError, match="不存在"):
            service.delete_document(tmp_db, "nope")


# ============================================================
# 统计
# ============================================================
class TestStats:
    def test_counts(self, tmp_db, service, kb, tmp_path) -> None:
        course = service.create_course(tmp_db, "Test")
        service.upload_bytes(tmp_db, course.id, "a.txt", b"main content", "main")
        service.upload_bytes(
            tmp_db, course.id, "b.txt", b"aux content", "auxiliary"
        )

        stats = service.get_stats(tmp_db, course.id)

        assert stats.total_documents == 2
        assert stats.main_documents == 1
        assert stats.auxiliary_documents == 1
        assert stats.indexed_documents == 0
        assert stats.index_progress == 0.0

        kb.build_kb(tmp_db, course.id)
        assert service.get_stats(tmp_db, course.id).index_progress == 1.0


# ============================================================
# KBService
# ============================================================
class TestKBService:
    def test_build_and_search(self, tmp_db, service, kb, src_file) -> None:
        course = service.create_course(tmp_db, "Test")
        service.upload_document(tmp_db, course.id, src_file)

        result = kb.build_kb(tmp_db, course.id)

        assert result.ok
        assert result.chunks_added >= 1

        hits = kb.search(course.id, "机器学习", k=3)
        assert len(hits) >= 1
        assert "机器学习" in hits[0].content

    def test_search_with_sources(self, tmp_db, service, kb, src_file) -> None:
        """⭐ 检索结果必须能溯源到真实存在的文件。"""
        course = service.create_course(tmp_db, "Test")
        service.upload_document(tmp_db, course.id, src_file)
        kb.build_kb(tmp_db, course.id)

        hits = kb.search_with_sources(course.id, "机器学习", k=1)

        assert len(hits) == 1
        hit = hits[0]
        assert hit.location.exists is True
        assert hit.location.file_path.exists()
        assert hit.location.source_name == "lecture.txt"
        assert "lecture.txt" in hit.label

    def test_incremental_then_full(self, tmp_db, service, kb, src_file) -> None:
        course = service.create_course(tmp_db, "Test")
        service.upload_document(tmp_db, course.id, src_file)

        kb.build_kb(tmp_db, course.id, incremental=True)
        n1 = kb.stats(tmp_db, course.id)["vectors_in_store"]

        kb.build_kb(tmp_db, course.id, incremental=False)
        n2 = kb.stats(tmp_db, course.id)["vectors_in_store"]

        assert n1 == n2  # 全量重建不该改变向量数量

    def test_build_context(self, tmp_db, service, kb, src_file) -> None:
        course = service.create_course(tmp_db, "Test")
        service.upload_document(tmp_db, course.id, src_file)
        kb.build_kb(tmp_db, course.id)

        context, used = kb.build_context(course.id, "机器学习", k=2)

        assert "[1]" in context
        assert len(used) >= 1

    def test_clear_kb(self, tmp_db, service, kb, src_file) -> None:
        course = service.create_course(tmp_db, "Test")
        service.upload_document(tmp_db, course.id, src_file)
        kb.build_kb(tmp_db, course.id)

        kb.clear_kb(tmp_db, course.id)

        assert kb.is_ready(course.id) is False
        assert document_repo.list_unindexed(tmp_db, course.id) != []

    def test_search_empty_kb(self, kb) -> None:
        assert kb.search("nobody", "查询") == []
        assert kb.search_with_sources("nobody", "查询") == []
```

---

## 11. Streamlit UI

### 11.1 目标

两个页面：
1. **📚 课件管理**：上传/删除/查看课件
2. **🧠 知识库**：构建/检索知识库

### 11.2 文件内容指引

#### `src/private_teacher/app.py`（主入口）

```python
import streamlit as st
from private_teacher.config import AppSettings
from private_teacher.storage.db import Database, init_db
from private_teacher.utils.logging import configure_logging

settings = AppSettings.load()
configure_logging(settings.logging)

# 初始化 DB（首次启动时）
data_dir = settings.paths.data_dir
data_dir.mkdir(parents=True, exist_ok=True)
with Database(data_dir / "app.db") as conn:
    init_db(conn)

# 用 st.session_state 存 conn
if "db_conn" not in st.session_state:
    st.session_state.db_conn = Database(data_dir / "app.db")
    init_db(st.session_state.db_conn.__enter__())
```

#### `src/private_teacher/ui/pages/1_📚_courseware.py`

```python
import streamlit as st
from private_teacher.services.course_service import CourseService
from private_teacher.storage.course_repo import list_all, create
from private_teacher.config import AppSettings

st.title("📚 课件管理")
settings = AppSettings.load()
service = CourseService(settings.paths)
conn = st.session_state.db_conn

# 课程选择
courses = list_all(conn)
course_names = [c.name for c in courses]
selected = st.selectbox("选择课程", course_names or ["(新建)"])
if selected == "(新建)":
    name = st.text_input("新课程名")
    if st.button("创建") and name:
        course = service.create_course(conn, name)
        st.success(f"已创建：{course.name}")
        st.rerun()
else:
    course = next(c for c in courses if c.name == selected)

    # 上传文件
    files = st.file_uploader(
        "上传课件", accept_multiple_files=True,
        type=["pdf", "pptx", "txt", "md", "py"]
    )
    category = st.radio("分类", ["main", "auxiliary"], horizontal=True)

    if st.button("上传") and files:
        for f in files:
            tmp = settings.paths.data_dir / "tmp" / f.name
            tmp.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_bytes(f.read())
            try:
                doc = service.upload_document(conn, course.id, tmp, category)
                st.success(f"已上传：{f.name}")
            except FileExistsError as e:
                st.warning(f"跳过：{e}")
            finally:
                tmp.unlink()
```

#### `src/private_teacher/ui/pages/2_🧠_knowledge_base.py`

```python
import streamlit as st
from private_teacher.services.kb_service import KBService
from private_teacher.services.course_service import CourseService
from private_teacher.storage.course_repo import list_all
from private_teacher.config import AppSettings

st.title("🧠 知识库")
settings = AppSettings.load()
kb_service = KBService(settings.llm, settings.paths)
conn = st.session_state.db_conn

# 选课程
courses = list_all(conn)
course = st.selectbox("课程", courses, format_func=lambda c: c.name)

if course:
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔨 增量构建知识库"):
            with st.spinner("构建中..."):
                result = kb_service.build_kb(conn, course.id, incremental=True)
            st.success(f"新增 {result['chunks']} 个 chunks")
    with col2:
        if st.button("♻️ 全量重建"):
            with st.spinner("重建中..."):
                result = kb_service.build_kb(conn, course.id, incremental=False)
            st.success(f"重建完成：{result['chunks']} 个 chunks")

    # 检索
    query = st.text_input("🔍 检索")
    if query:
        chunks = kb_service.search(course.id, query, k=5)
        sources = kb_service.resolve_sources(course.id, chunks)
        for i, (chunk, src) in enumerate(zip(chunks, sources), 1):
            with st.expander(f"[{i}] {src.file_path.name} (page={src.page})"):
                st.markdown(chunk.page_content)
                st.caption(f"📄 {src.file_path}")
                if src.line_start:
                    st.caption(f"📍 第 {src.line_start}-{src.line_end} 行")
```

### 11.3 验证

```bash
uv run streamlit run src/private_teacher/app.py
```

打开浏览器 `http://localhost:8501` 测试。

### 11.4 完整代码实现

> **先装依赖**：`uv add streamlit`
>
> **一个关键的 Streamlit 知识点**：默认的多页应用要求 `pages/` 目录与入口脚本同级。
> 但我们的入口是 `src/private_teacher/app.py`，页面想放在 `ui/pages/` 下。
> 解决办法是用 **`st.navigation` + `st.Page`**（Streamlit ≥ 1.36 的新 API），
> 它允许显式声明页面文件路径，不受目录约定限制。

#### 11.4.1 `src/private_teacher/ui/state.py`（共享状态）

```python
"""
Streamlit 共享状态
==================

【Streamlit 的执行模型 —— 必须先理解这个】
每次用户点按钮、输入文字，Streamlit 会**从头到尾重新执行整个脚本**。
这意味着：
  - 普通变量每次都会重新创建（数据库连接会被反复打开！）
  - 想跨次数保留的东西，必须放进 st.session_state 或用缓存装饰器

【两个缓存装饰器的区别】
  @st.cache_resource : 全局单例，所有用户共享。用于连接、模型、客户端
  @st.cache_data     : 缓存"数据"，会做序列化拷贝。用于查询结果、DataFrame

数据库连接、embedding 模型属于"资源"，必须用 cache_resource ——
用 cache_data 会尝试 pickle 连接对象，直接报错。
"""

from __future__ import annotations

import sqlite3

import streamlit as st

from private_teacher.config import AppSettings
from private_teacher.services import CourseService, KBService
from private_teacher.storage.db import get_connection, init_db
from private_teacher.utils.logging import configure_logging


@st.cache_resource
def get_settings() -> AppSettings:
    """加载配置（全进程只做一次）。"""
    settings = AppSettings.load()
    configure_logging(settings.logging)
    settings.paths.ensure_data_dir()
    return settings


@st.cache_resource
def get_conn() -> sqlite3.Connection:
    """全局共享的数据库连接。

    ⚠️ check_same_thread=False 是必须的：
       Streamlit 用线程池处理请求，每次 rerun 可能落在不同线程上，
       默认的 sqlite3 会抛
       "SQLite objects created in a thread can only be used in that same thread"。

    传 False 的安全性：
       Streamlit 单个 session 的脚本执行是串行的，
       同一时刻不会有两个线程操作同一个连接。
       加上我们开了 WAL 模式 + busy_timeout，多标签页也能撑住。
    """
    settings = get_settings()
    db_path = settings.paths.data_dir / "app.db"

    conn = get_connection(db_path, check_same_thread=False)
    init_db(conn)  # 幂等，每次启动跑一下保证表存在
    return conn


@st.cache_resource
def get_course_service() -> CourseService:
    return CourseService(get_settings().paths)


@st.cache_resource
def get_kb_service() -> KBService:
    """知识库服务。

    注意它内部持有 embedder —— 用 cache_resource 才能避免
    每次 rerun 都重新加载几百 MB 的本地模型。
    """
    settings = get_settings()
    return KBService(settings.llm, settings.paths)


# ============================================================
# 当前选中课程（跨页面共享）
# ============================================================
CURRENT_COURSE_KEY = "current_course_id"


def get_current_course_id() -> str | None:
    """读当前选中的课程 id。

    st.session_state 是**每个浏览器标签页独立**的字典，
    刷新页面会保留，关掉标签页就清空。
    """
    return st.session_state.get(CURRENT_COURSE_KEY)


def set_current_course_id(course_id: str | None) -> None:
    st.session_state[CURRENT_COURSE_KEY] = course_id


def course_selector(label: str = "当前课程") -> str | None:
    """侧边栏的课程选择器（两个页面共用）。

    Returns:
        选中的 course_id；没有任何课程时返回 None
    """
    conn = get_conn()
    courses = get_course_service().list_courses(conn)

    if not courses:
        st.sidebar.info("还没有课程，请先在「课件管理」页创建")
        return None

    ids = [c.id for c in courses]
    current = get_current_course_id()
    # 上次选的课可能已被删除，要做存在性检查
    index = ids.index(current) if current in ids else 0

    selected = st.sidebar.selectbox(
        label,
        options=ids,
        index=index,
        # format_func 控制显示文本，实际值仍是 id
        format_func=lambda cid: next(c.name for c in courses if c.id == cid),
    )
    set_current_course_id(selected)
    return selected
```

#### 11.4.2 `src/private_teacher/app.py`（入口）

```python
"""
Private Teacher —— Streamlit 入口
=================================

运行：
    uv run streamlit run src/private_teacher/app.py
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from private_teacher.ui.state import get_conn, get_course_service, get_settings

# ============================================================
# 全局页面配置
# ============================================================
# ⚠️ set_page_config 必须是**第一个** Streamlit 调用，否则报错
st.set_page_config(
    page_title="Private Teacher",
    page_icon="🎓",
    layout="wide",  # 宽屏：检索结果表格才不会挤成一团
    initial_sidebar_state="expanded",
)

# 触发一次初始化（配置 + 建库 + 建目录）
settings = get_settings()
get_conn()

# 页面文件目录
PAGES_DIR = Path(__file__).parent / "ui" / "pages"


def render_home() -> None:
    """首页：项目介绍 + 当前状态总览。"""
    st.title("🎓 Private Teacher")
    st.caption("本地优先的私人 AI 教师 —— 上传课件，构建知识库，随时提问")

    conn = get_conn()
    courses = get_course_service().list_courses(conn)

    # ---------- 指标卡片 ----------
    col1, col2, col3 = st.columns(3)
    col1.metric("课程数", len(courses))

    total_docs = sum(
        get_course_service().get_stats(conn, c.id).total_documents for c in courses
    )
    col2.metric("课件总数", total_docs)
    col3.metric("LLM Provider", settings.llm.provider)

    st.divider()

    if not courses:
        st.info("👈 从左侧「📚 课件管理」开始，创建你的第一门课程")
        return

    # ---------- 课程列表 ----------
    st.subheader("我的课程")
    for course in courses:
        stats = get_course_service().get_stats(conn, course.id)
        with st.container(border=True):
            left, right = st.columns([3, 1])
            with left:
                st.markdown(f"**{course.name}**")
                st.caption(course.description or "（暂无描述）")
            with right:
                st.metric(
                    "已索引",
                    f"{stats.indexed_documents}/{stats.total_documents}",
                )
            # progress 需要 0.0~1.0 的浮点数
            st.progress(stats.index_progress)


# ============================================================
# 多页面导航
# ============================================================
# st.Page 显式声明页面：路径 + 标题 + 图标 + url 路径
# 好处：页面文件可以放在任意目录，不受 "pages/ 必须与入口同级" 的约束
home = st.Page(render_home, title="首页", icon="🏠", default=True)
courseware = st.Page(
    str(PAGES_DIR / "1_courseware.py"), title="课件管理", icon="📚"
)
knowledge_base = st.Page(
    str(PAGES_DIR / "2_knowledge_base.py"), title="知识库", icon="🧠"
)

# navigation 返回被选中的页面对象，run() 执行它
pg = st.navigation([home, courseware, knowledge_base])

# ---------- 侧边栏底部信息 ----------
with st.sidebar:
    st.divider()
    st.caption(f"数据目录：`{settings.paths.data_dir}`")
    st.caption(f"Embedding：`{settings.llm.embedding_provider}`")

pg.run()
```

#### 11.4.3 `src/private_teacher/ui/pages/1_courseware.py`

```python
"""
📚 课件管理页
=============

功能：创建课程 / 上传课件 / 查看列表 / 删除。
"""

from __future__ import annotations

import streamlit as st

from private_teacher.loaders import SUPPORTED_EXTENSIONS
from private_teacher.ui.state import (
    course_selector,
    get_conn,
    get_course_service,
)
from private_teacher.utils.exceptions import (
    DocumentLoadError,
    DuplicateDocumentError,
    ServiceError,
)

st.title("📚 课件管理")

conn = get_conn()
service = get_course_service()

# ============================================================
# 侧边栏：课程选择 + 新建
# ============================================================
course_id = course_selector()

with st.sidebar.expander("➕ 新建课程"):
    # 用 form 把多个输入框打包：
    # 不用 form 的话，每敲一个字符都会触发整个脚本 rerun（很卡）
    with st.form("new_course", clear_on_submit=True):
        new_name = st.text_input("课程名称")
        new_desc = st.text_area("课程描述", height=80)

        if st.form_submit_button("创建", use_container_width=True):
            try:
                course = service.create_course(conn, new_name, new_desc)
                st.success(f"已创建：{course.name}")
                # rerun 让新课程立刻出现在选择器里
                st.rerun()
            except ServiceError as exc:
                st.error(str(exc))

if course_id is None:
    st.info("请先在左侧创建一门课程")
    st.stop()  # 中止本次脚本执行，下面的代码不会跑

course = service.get_course(conn, course_id)
stats = service.get_stats(conn, course_id)

# ============================================================
# 顶部：课程概览
# ============================================================
st.subheader(course.name)
if course.description:
    st.caption(course.description)

c1, c2, c3, c4 = st.columns(4)
c1.metric("总课件", stats.total_documents)
c2.metric("主干", stats.main_documents)
c3.metric("辅助", stats.auxiliary_documents)
c4.metric("已索引", stats.indexed_documents)

st.divider()

# ============================================================
# 上传区
# ============================================================
st.subheader("上传课件")

upload_col, option_col = st.columns([3, 1])

with option_col:
    category = st.radio(
        "分类",
        options=["main", "auxiliary"],
        format_func=lambda c: "主干课程" if c == "main" else "辅助资料",
        help="主干 = 教材/讲义；辅助 = 代码/扩展阅读",
    )

with upload_col:
    files = st.file_uploader(
        "选择文件（可多选）",
        accept_multiple_files=True,
        # type 只接受不带点的扩展名
        type=[ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS],
    )

if files and st.button("⬆️ 开始上传", type="primary"):
    progress = st.progress(0.0, text="准备上传...")
    ok, skipped, failed = 0, 0, 0

    for i, uploaded in enumerate(files, start=1):
        progress.progress(i / len(files), text=f"上传 {uploaded.name}")
        try:
            # getvalue() 拿到完整字节流（read() 只能读一次，容易踩坑）
            service.upload_bytes(
                conn, course_id, uploaded.name, uploaded.getvalue(), category
            )
            ok += 1
        except DuplicateDocumentError:
            skipped += 1
        except (DocumentLoadError, ServiceError) as exc:
            st.error(f"{uploaded.name}: {exc}")
            failed += 1

    progress.empty()  # 清掉进度条

    msg = f"成功 {ok}"
    if skipped:
        msg += f"，跳过重复 {skipped}"
    if failed:
        msg += f"，失败 {failed}"
    st.success(msg)

    if ok:
        st.info("💡 别忘了去「🧠 知识库」页面构建索引")
    st.rerun()

st.divider()

# ============================================================
# 课件列表
# ============================================================
st.subheader("课件列表")

tab_main, tab_aux = st.tabs(["📘 主干课程", "📗 辅助资料"])

for tab, cat in ((tab_main, "main"), (tab_aux, "auxiliary")):
    with tab:
        docs = service.list_documents(conn, course_id, cat)

        if not docs:
            st.caption("（空）")
            continue

        for doc in docs:
            with st.container(border=True):
                info, status, action = st.columns([5, 2, 1])

                with info:
                    st.markdown(f"**{doc.path.name}**")
                    size_kb = doc.size / 1024
                    st.caption(
                        f"{size_kb:.1f} KB · "
                        f"{doc.mtime:%Y-%m-%d %H:%M} · "
                        f"`{doc.sha256[:8]}`"
                    )

                with status:
                    if not doc.exists:
                        st.warning("文件丢失")
                    elif doc.is_indexed:
                        st.success("已索引")
                    else:
                        st.info("待索引")

                with action:
                    # key 必须唯一，否则多个按钮会互相干扰
                    if st.button("🗑️", key=f"del_{doc.id}", help="删除"):
                        service.delete_document(conn, doc.id)
                        st.rerun()

st.divider()

# ============================================================
# 危险操作
# ============================================================
with st.expander("⚠️ 危险区域"):
    st.warning("删除课程会同时删除所有课件文件和知识库，**不可恢复**")

    # 要求输入课程名确认 —— 防误触的标准做法（GitHub 删仓库同款）
    confirm = st.text_input(f"输入课程名「{course.name}」以确认删除")

    if st.button("永久删除本课程", type="primary", disabled=confirm != course.name):
        service.delete_course(conn, course_id)
        st.success("课程已删除")
        st.rerun()
```

#### 11.4.4 `src/private_teacher/ui/pages/2_knowledge_base.py`

```python
"""
🧠 知识库页
===========

功能：构建/重建索引 + 检索 + 结果溯源。
"""

from __future__ import annotations

import streamlit as st

from private_teacher.ui.state import (
    course_selector,
    get_conn,
    get_course_service,
    get_kb_service,
)

st.title("🧠 知识库")

conn = get_conn()
course_service = get_course_service()
kb = get_kb_service()

course_id = course_selector()
if course_id is None:
    st.info("请先创建课程并上传课件")
    st.stop()

course = course_service.get_course(conn, course_id)
stats = kb.stats(conn, course_id)

# ============================================================
# 状态面板
# ============================================================
st.subheader(f"{course.name} · 知识库状态")

c1, c2, c3, c4 = st.columns(4)
c1.metric("课件总数", stats["documents_total"])
c2.metric("已索引", stats["documents_indexed"])
c3.metric("待处理", stats["documents_pending"])
c4.metric("向量条数", stats["vectors_in_store"])

if stats["documents_total"] == 0:
    st.info("这门课还没有课件，请先去「📚 课件管理」上传")
    st.stop()

# ============================================================
# 构建
# ============================================================
st.divider()
build_col1, build_col2 = st.columns(2)


def run_build(incremental: bool) -> None:
    """执行构建并展示进度。"""
    progress = st.progress(0.0, text="准备中...")

    def on_file(idx: int, total: int, name: str) -> None:
        # 回调里更新进度条 —— Indexer 每处理一个文件就调一次
        progress.progress(idx / max(total, 1), text=f"处理 {name}（{idx}/{total}）")

    # spinner 在耗时操作外面包一层，给用户"正在忙"的反馈
    with st.spinner("构建中，请勿关闭页面..."):
        result = kb.build_kb(conn, course_id, incremental=incremental, on_file=on_file)

    progress.empty()

    if result.ok:
        st.success(result.summary())
    else:
        st.warning(result.summary())
        with st.expander(f"查看 {len(result.errors)} 个失败项"):
            for err in result.errors:
                st.text(f"• {err}")


with build_col1:
    if st.button(
        "🔨 增量构建",
        type="primary",
        use_container_width=True,
        help="只处理新增和修改过的文件，速度快",
        disabled=stats["documents_pending"] == 0,
    ):
        run_build(incremental=True)
        st.rerun()

with build_col2:
    if st.button(
        "♻️ 全量重建",
        use_container_width=True,
        help="清空后重新索引全部文件。换了 embedding 模型或改了切分参数时用",
    ):
        run_build(incremental=False)
        st.rerun()

if stats["documents_pending"] == 0 and stats["documents_total"] > 0:
    st.caption("✅ 所有课件都已索引，无需增量构建")

# ============================================================
# 检索
# ============================================================
st.divider()
st.subheader("🔍 检索")

if stats["vectors_in_store"] == 0:
    st.info("知识库还是空的，请先点击上方「增量构建」")
    st.stop()

query = st.text_input(
    "输入你想查找的内容",
    placeholder="例如：什么是决策树的信息增益？",
)

with st.expander("⚙️ 高级选项"):
    opt1, opt2, opt3 = st.columns(3)
    top_k = opt1.slider("返回条数", 1, 20, 5)
    category_filter = opt2.selectbox(
        "范围",
        options=[None, "main", "auxiliary"],
        format_func=lambda c: {
            None: "全部",
            "main": "仅主干",
            "auxiliary": "仅辅助",
        }[c],
    )
    min_score = opt3.slider(
        "最低相似度", 0.0, 1.0, 0.0, 0.05,
        help="过滤掉不太相关的结果。0 表示不过滤",
    )

if query:
    hits = kb.search_with_sources(
        course_id,
        query,
        k=top_k,
        category=category_filter,
        min_score=min_score or None,
    )

    if not hits:
        st.warning("没有找到相关内容。试试换个说法，或降低最低相似度。")
    else:
        st.caption(f"找到 {len(hits)} 条结果")

        for i, hit in enumerate(hits, start=1):
            # expanded=(i == 1)：默认只展开第一条，其余折叠
            with st.expander(
                f"**[{i}]** {hit.label} · 相似度 {hit.score:.2%}",
                expanded=(i == 1),
            ):
                # ---------- 内容 ----------
                st.markdown(hit.content)

                st.divider()

                # ---------- 溯源信息 ----------
                loc = hit.location
                meta_col, link_col = st.columns([3, 1])

                with meta_col:
                    st.caption(f"📄 `{loc.file_path}`")
                    if loc.has_lines:
                        st.caption(f"📍 第 {loc.line_start}-{loc.line_end} 行")
                    elif loc.has_pages:
                        st.caption(f"📖 第 {loc.page}/{loc.total_pages} 页")

                with link_col:
                    if loc.exists and loc.uri():
                        # Streamlit 的 link_button 会在新标签页打开
                        st.link_button("打开源文件", loc.uri())
                    else:
                        st.caption("⚠️ 源文件已移除")

                # ---------- 源文件上下文（仅文本类） ----------
                if loc.exists and loc.has_lines:
                    resolver = kb._resolver(course_id)  # noqa: SLF001
                    snippet = resolver.read_numbered_snippet(loc, context_lines=3)
                    with st.popover("查看源文件上下文"):
                        st.code(snippet, language=loc.file_type or "text")
```

#### 11.4.5 `.streamlit/config.toml`

```toml
# Streamlit 全局配置（放在项目根目录）

[server]
# 上传文件大小上限（MB）。默认 200，课件 PDF 有时会超
maxUploadSize = 500
# 关掉自动打开浏览器（在服务器上跑时很烦）
headless = true

[browser]
# 不上报使用统计 —— 本项目是 local-first
gatherUsageStats = false

[theme]
base = "light"
primaryColor = "#4F8BF9"

[runner]
# 保存文件后自动重跑，开发时很方便
fastReruns = true
```

#### 11.4.6 `Makefile` 追加

```makefile
# ---------- Phase 1 ----------
.PHONY: streamlit samples

streamlit:  ## 启动 Streamlit 应用
	uv run streamlit run src/private_teacher/app.py

samples:  ## 生成测试用课件样本
	uv run python scripts/make_sample_courseware.py
```

#### 11.4.7 UI 层的踩坑清单

| 现象 | 原因 | 解决 |
| --- | --- | --- |
| `SQLite objects created in a thread...` | Streamlit 多线程 rerun | `get_connection(..., check_same_thread=False)` |
| 每次点按钮都重新加载 embedding 模型 | 没缓存 | `@st.cache_resource` 包住 service |
| `set_page_config() can only be called once` | 不是第一个 st 调用 | 放在 import 之后的第一行 |
| 上传大文件报 `AxiosError 413` | 超过 maxUploadSize | 改 `.streamlit/config.toml` |
| 按钮点了没反应 | 多个按钮 `key` 重复 | 每个按钮给唯一 `key=f"xx_{id}"` |
| 输入框每敲一个字就卡一下 | 每次输入都 rerun | 用 `st.form` 包起来批量提交 |
| 修改代码后页面没变 | 缓存了旧对象 | 侧边栏菜单 → Clear cache，或重启 |

---

## 12. 集成测试 + 收尾验证

### 12.1 集成测试

**tests/integration/test_rag_pipeline.py**

```python
def test_full_rag_pipeline(tmp_path, fake_embedder):
    """端到端测试：上传 → 索引 → 检索 → 解析。"""
    # 1. 创建课程
    conn = create_test_db(tmp_path)
    service = CourseService(PathSettings(data_dir=tmp_path))
    course = service.create_course(conn, "Test")

    # 2. 上传 PDF（TXT 模拟）
    f = tmp_path / "doc.txt"
    f.write_text("Python 是一种广泛使用的编程语言。它强调代码可读性。")
    service.upload_document(conn, course.id, f, "main")

    # 3. 构建知识库
    settings = LLMSettings(provider="ollama")  # 用 fake embedder
    kb = KBService(settings, PathSettings(data_dir=tmp_path))
    result = kb.build_kb(conn, course.id)
    assert result["chunks"] > 0

    # 4. 检索
    chunks = kb.search(course.id, "Python")
    assert len(chunks) > 0

    # 5. 源文件解析
    sources = kb.resolve_sources(course.id, chunks)
    assert sources[0].file_path.exists()
```

#### 12.1.1 完整代码实现

**`tests/integration/test_rag_pipeline.py`**

```python
"""
端到端集成测试
==============

覆盖真实用户路径：创建课程 → 上传课件 → 建索引 → 检索 → 跳回源文件。

【与单元测试的区别】
  单元测试：一次只测一个函数，其他全用假的
  集成测试：所有真实模块串起来跑，只有 embedding 用 HashEmbedder 替身
            （因为真 embedding 要联网 + 花钱，且结果不稳定）

集成测试跑得慢（每个用例要建 Chroma 库），所以数量要少而精 ——
只覆盖"多个模块协作时才可能出现"的 bug。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from private_teacher.config import LLMSettings, PathSettings
from private_teacher.rag.embeddings import HashEmbedder
from private_teacher.services import CourseService, KBService
from private_teacher.storage import document_repo
from private_teacher.storage.db import Database, init_db

# 整个模块都标记为集成测试：
# 想跳过它们： uv run pytest -m "not integration"
pytestmark = pytest.mark.integration


# ============================================================
# 环境搭建
# ============================================================
@pytest.fixture
def env(tmp_path: Path):
    """一套完整的隔离环境：数据库 + 数据目录 + 两个 service。"""
    data_dir = tmp_path / "data"
    paths = PathSettings(data_dir=data_dir)
    llm = LLMSettings(embedding_provider="hash", embedding_dimension=256)
    embedder = HashEmbedder(dimension=256)

    db = Database(data_dir / "app.db")
    conn = db.connect()
    init_db(conn)

    yield {
        "conn": conn,
        "paths": paths,
        "course": CourseService(paths),
        "kb": KBService(llm, paths, embedder=embedder),
        "tmp": tmp_path,
    }

    db.close()


def write_file(tmp: Path, name: str, content: str) -> Path:
    """在临时目录造一个待上传的源文件。"""
    f = tmp / "incoming" / name
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(content, encoding="utf-8")
    return f


# ============================================================
# 主流程
# ============================================================
def test_full_rag_pipeline(env) -> None:
    """⭐ 完整链路：上传 → 索引 → 检索 → 溯源。"""
    conn, course_svc, kb, tmp = env["conn"], env["course"], env["kb"], env["tmp"]

    # ---------- 1. 创建课程 ----------
    course = course_svc.create_course(conn, "机器学习导论", "本科生入门课")
    assert env["paths"].course_dir(course.id).exists()

    # ---------- 2. 上传课件 ----------
    lecture = write_file(
        tmp,
        "lecture.txt",
        "机器学习是人工智能的核心分支，研究计算机如何从数据中学习。\n"
        "监督学习需要带标签的训练数据，常见任务是分类与回归。\n"
        "决策树通过信息增益选择划分特征，是一种可解释性很强的模型。",
    )
    code = write_file(
        tmp,
        "train.py",
        "\n".join(
            [
                "def load_data(path):",
                "    return open(path).read()",
                "",
                "def train_model(X, y):",
                "    # 训练决策树分类器",
                "    return DecisionTree().fit(X, y)",
            ]
        ),
    )

    doc1 = course_svc.upload_document(conn, course.id, lecture, "main")
    doc2 = course_svc.upload_document(conn, course.id, code, "auxiliary")

    assert doc1.path.exists()
    assert doc2.category == "auxiliary"

    # ---------- 3. 构建知识库 ----------
    result = kb.build_kb(conn, course.id, incremental=True)

    assert result.ok, f"索引失败：{result.errors}"
    assert result.documents_indexed == 2
    assert result.chunks_added >= 2

    # 数据库状态应同步更新
    assert document_repo.list_unindexed(conn, course.id) == []

    # ---------- 4. 检索 ----------
    hits = kb.search_with_sources(course.id, "决策树", k=3)

    assert len(hits) >= 1
    assert "决策树" in hits[0].content

    # ---------- 5. 溯源 ----------
    loc = hits[0].location
    assert loc.exists is True
    assert loc.file_path.exists()
    assert loc.source_name in {"lecture.txt", "train.py"}
    assert loc.label()  # 能生成人类可读的位置描述


def test_source_jump_precision(env) -> None:
    """⭐ 代码检索必须能精确定位到行，并读回真实源文件内容。"""
    conn, course_svc, kb, tmp = env["conn"], env["course"], env["kb"], env["tmp"]

    course = course_svc.create_course(conn, "编程课")
    # 造一个 150 行的文件，目标函数在第 120 行附近
    lines = [f"# 填充第 {i} 行注释内容" for i in range(1, 120)]
    lines.append("def calculate_gini_impurity(labels):")
    lines.append("    # 计算基尼不纯度指标")
    lines.append("    return 1 - sum(p * p for p in probabilities)")
    lines.extend(f"# 尾部第 {i} 行" for i in range(1, 30))

    src = write_file(tmp, "tree.py", "\n".join(lines))
    course_svc.upload_document(conn, course.id, src, "auxiliary")
    kb.build_kb(conn, course.id)

    hits = kb.search_with_sources(course.id, "calculate_gini_impurity", k=1)

    assert len(hits) == 1
    loc = hits[0].location

    # 必须有行号
    assert loc.has_lines
    assert loc.line_start is not None

    # 从源文件读回来的内容里应该真的包含目标函数
    resolver = kb._resolver(course.id)  # noqa: SLF001
    snippet = resolver.read_snippet(loc, context_lines=2)
    assert "calculate_gini_impurity" in snippet


def test_incremental_workflow(env) -> None:
    """⭐ 真实使用节奏：先传两个 → 建库 → 再传一个 → 增量。"""
    conn, course_svc, kb, tmp = env["conn"], env["course"], env["kb"], env["tmp"]

    course = course_svc.create_course(conn, "增量测试")

    course_svc.upload_document(
        conn, course.id, write_file(tmp, "a.txt", "第一份课件内容，讲的是线性回归。")
    )
    course_svc.upload_document(
        conn, course.id, write_file(tmp, "b.txt", "第二份课件内容，讲的是逻辑回归。")
    )

    first = kb.build_kb(conn, course.id)
    assert first.documents_indexed == 2
    vectors_after_first = kb.stats(conn, course.id)["vectors_in_store"]

    # 再传一个
    course_svc.upload_document(
        conn, course.id, write_file(tmp, "c.txt", "第三份课件，讲支持向量机 SVM。")
    )

    second = kb.build_kb(conn, course.id)

    assert second.documents_indexed == 1  # 只处理新的
    assert second.documents_skipped == 2  # 老的跳过
    assert kb.stats(conn, course.id)["vectors_in_store"] > vectors_after_first

    # 新内容立刻可检索
    hits = kb.search(course.id, "支持向量机", k=1)
    assert "SVM" in hits[0].content or "支持向量机" in hits[0].content


def test_delete_document_cleans_index(env) -> None:
    """⭐ 删掉课件后，它的内容必须彻底从检索结果里消失。"""
    conn, course_svc, kb, tmp = env["conn"], env["course"], env["kb"], env["tmp"]

    course = course_svc.create_course(conn, "删除测试")
    doc_a = course_svc.upload_document(
        conn, course.id, write_file(tmp, "a.txt", "量子计算使用量子比特进行运算。")
    )
    course_svc.upload_document(
        conn, course.id, write_file(tmp, "b.txt", "经典计算机使用二进制比特。")
    )
    kb.build_kb(conn, course.id)

    assert kb.search(course.id, "量子计算", k=5)

    course_svc.delete_document(conn, doc_a.id)

    remaining = kb.search(course.id, "量子计算", k=5)
    assert all("量子" not in r.content for r in remaining)


def test_modified_file_reflected_in_search(env) -> None:
    """⭐ 改了课件内容并重新索引后，检索结果要跟着更新。"""
    conn, course_svc, kb, tmp = env["conn"], env["course"], env["kb"], env["tmp"]
    from private_teacher.utils.hashing import compute_sha256

    course = course_svc.create_course(conn, "更新测试")
    src = write_file(tmp, "note.txt", "原始内容：介绍朴素贝叶斯分类器。")
    doc = course_svc.upload_document(conn, course.id, src, "main")
    kb.build_kb(conn, course.id)

    # 直接改课件目录里的那份（doc.path 才是真正被索引的文件）
    doc.path.write_text("更新内容：改为介绍随机森林集成方法。", encoding="utf-8")
    conn.execute(
        "UPDATE documents SET sha256 = ? WHERE id = ?",
        (compute_sha256(doc.path), doc.id),
    )
    conn.commit()

    kb.build_kb(conn, course.id)

    assert kb.search(course.id, "随机森林", k=3)
    old_hits = kb.search(course.id, "朴素贝叶斯", k=3)
    assert all("朴素贝叶斯" not in r.content for r in old_hits)


def test_multi_course_isolation(env) -> None:
    """⭐ 两门课的知识库必须完全隔离，不能串味。"""
    conn, course_svc, kb, tmp = env["conn"], env["course"], env["kb"], env["tmp"]

    math_course = course_svc.create_course(conn, "高等数学")
    history = course_svc.create_course(conn, "世界历史")

    course_svc.upload_document(
        conn, math_course.id, write_file(tmp, "m.txt", "微积分研究导数与积分。")
    )
    course_svc.upload_document(
        conn, history.id, write_file(tmp, "h.txt", "文艺复兴起源于意大利佛罗伦萨。")
    )

    kb.build_kb(conn, math_course.id)
    kb.build_kb(conn, history.id)

    # 在数学课里搜历史内容，应该搜不到
    math_hits = kb.search(math_course.id, "文艺复兴", k=5)
    assert all("文艺复兴" not in r.content for r in math_hits)

    history_hits = kb.search(history.id, "文艺复兴", k=1)
    assert "文艺复兴" in history_hits[0].content


def test_delete_course_wipes_everything(env) -> None:
    """⭐ 删课程后：文件、数据库、向量库、manifest 全部干净。"""
    conn, course_svc, kb, tmp, paths = (
        env["conn"], env["course"], env["kb"], env["tmp"], env["paths"],
    )

    course = course_svc.create_course(conn, "待删除课程")
    course_svc.upload_document(
        conn, course.id, write_file(tmp, "a.txt", "一些将被彻底删除的内容。")
    )
    kb.build_kb(conn, course.id)

    course_svc.delete_course(conn, course.id)

    assert not paths.course_dir(course.id).exists()
    assert not (paths.data_dir / "knowledge_bases" / course.id).exists()
    assert document_repo.list_by_course(conn, course.id) == []
    assert kb.search(course.id, "内容") == []


def test_partial_failure_does_not_block(env) -> None:
    """⭐ 一个坏文件不能拖垮整批索引。"""
    conn, course_svc, kb, tmp = env["conn"], env["course"], env["kb"], env["tmp"]

    course = course_svc.create_course(conn, "容错测试")
    good = course_svc.upload_document(
        conn, course.id, write_file(tmp, "good.txt", "这份文件完全正常可以索引。")
    )
    bad = course_svc.upload_document(
        conn, course.id, write_file(tmp, "bad.txt", "这份文件会被偷偷删掉。")
    )

    # 模拟：数据库有记录，但物理文件被用户手动删了
    bad.path.unlink()

    result = kb.build_kb(conn, course.id)

    assert result.documents_indexed == 1  # good 成功
    assert len(result.errors) == 1  # bad 记录在案
    assert "bad.txt" in result.errors[0]
    assert kb.search(course.id, "正常", k=1)  # 好文件照样能搜到


def test_persistence_across_restart(env, tmp_path: Path) -> None:
    """⭐ 模拟重启：新建连接和 service，数据必须都还在。"""
    conn, course_svc, kb, tmp = env["conn"], env["course"], env["kb"], env["tmp"]

    course = course_svc.create_course(conn, "持久化测试")
    course_svc.upload_document(
        conn, course.id, write_file(tmp, "a.txt", "重启之后这段内容还要能被搜到。")
    )
    kb.build_kb(conn, course.id)
    conn.commit()

    # ---------- 模拟重启 ----------
    paths = env["paths"]
    new_db = Database(paths.data_dir / "app.db")
    new_conn = new_db.connect()
    new_kb = KBService(
        LLMSettings(embedding_provider="hash", embedding_dimension=256),
        paths,
        embedder=HashEmbedder(dimension=256),
    )

    try:
        assert len(CourseService(paths).list_courses(new_conn)) == 1
        hits = new_kb.search(course.id, "重启", k=1)
        assert len(hits) == 1
    finally:
        new_db.close()
```

**运行方式**：

```bash
# 只跑集成测试
uv run pytest tests/integration -v

# 跳过集成测试（日常快速反馈）
uv run pytest -m "not integration" -q
```


### 12.2 收尾清单

- [ ] 所有 loader 都实现并测过
- [ ] Splitter 测试通过
- [ ] Chroma 集成 OK
- [ ] Indexer 支持全量 + 增量
- [ ] Retriever 检索准确
- [ ] Source Resolver 跳转准确
- [ ] Streamlit 上传 + 检索 UI 跑通
- [ ] `make ci` 全绿
- [ ] `uv run streamlit run src/private_teacher/app.py` 能打开

### 12.3 阶段 1 → 阶段 2 衔接

**给 Phase 2 准备**：
- `KBService.search()` 是 Phase 2 Agent 的基础（Agent 会用 RAG 检索）
- `data/units/{course_id}/plan.json` 的目录建好占位
- `SourceResolver` 会在 Phase 2 单元学习里频繁用

**Phase 2 主要做**：
- `agents/course_analyzer.py`：用 RAG 概览全库，LLM 切分单元
- `agents/unit_planner.py`：每个单元生成 todo
- `agents/learn_content_generator.py`：生成学习 Markdown
- Streamlit 单元页面

### 12.4 完整命令清单

```bash
# 开发
make install
make test          # 跑所有测试
make lint-fix      # 自动修格式
make typecheck     # mypy
make ci            # 跑全套

# 跑 Streamlit
uv run streamlit run src/private_teacher/app.py

# 生成测试样本
uv run python scripts/make_sample_courseware.py
```

### 12.5 5 个常见踩坑

| 现象 | 原因 | 解决 |
| --- | --- | --- |
| Chroma 写入慢 | embedding 阻塞 | Phase 5 加 async 批处理 |
| 检索结果不相关 | embedding 模型差 | 换 text-embedding-3-large |
| 源文件路径失效 | 文件被移动/删除 | 跳错处理显示「文件已移除」 |
| PDF 解析乱码 | 字体缺失 | 用 OCR 或换 unstructured |
| 上传大文件超时 | Streamlit 默认 200MB | `.streamlit/config.toml` 改 limit |

---

## 13. 一句话总结

> **Phase 1 = 「上传 → 切 → 向量化 → 存 → 检索」全链路**。  
> 8 个模块（loaders / splitters / embeddings / vector_store / indexer / retriever / source_resolver / services），2 个 Streamlit 页面（课件 + 知识库）。  
> 跑通后 `make hello-llm` 还能用（Phase 0 冒烟测试），加 `make streamlit` 跑 Phase 1 演示。  
> **完成后可以搜任何课件内容并跳回源文件**，这是整个项目最核心的能力之一。

---

## 14. 附录：完整文件清单与落地顺序

### 14.1 本文实现的全部文件

| # | 文件 | 所在小节 | 说明 |
| --- | --- | --- | --- |
| 1 | `src/private_teacher/utils/hashing.py` | 2.7.1 | sha256 工具 |
| 2 | `src/private_teacher/storage/schema.sql` | 2.7.2 | 建表 SQL |
| 3 | `src/private_teacher/storage/db.py` | 2.7.3 | 连接 + 上下文管理器 |
| 4 | `src/private_teacher/storage/models.py` | 2.7.4 | Course / Document |
| 5 | `src/private_teacher/storage/course_repo.py` | 2.7.5 | 课程 CRUD |
| 6 | `src/private_teacher/storage/document_repo.py` | 2.7.6 | 文档 CRUD + 去重 |
| 7 | `src/private_teacher/storage/__init__.py` | 2.7.7 | re-export |
| 8 | `src/private_teacher/loaders/base.py` | 3.7.1 | metadata 契约 |
| 9 | `src/private_teacher/loaders/text_loader.py` | 3.7.2 | 智能编码识别 |
| 10 | `src/private_teacher/loaders/markdown_loader.py` | 3.7.3 | frontmatter 处理 |
| 11 | `src/private_teacher/loaders/code_loader.py` | 3.7.4 | 行号保留 |
| 12 | `src/private_teacher/loaders/pdf_loader.py` | 3.7.5 | pypdf + pdfplumber 降级 |
| 13 | `src/private_teacher/loaders/pptx_loader.py` | 3.7.6 | 表格 + 备注 |
| 14 | `src/private_teacher/loaders/registry.py` | 3.7.7 | 统一入口 |
| 15 | `src/private_teacher/loaders/directory_loader.py` | 3.7.8 | 递归扫描 |
| 16 | `src/private_teacher/loaders/__init__.py` | 3.7.9 | re-export |
| 17 | `scripts/make_sample_courseware.py` | 3.7.10 | 生成测试样本 |
| 18 | `src/private_teacher/rag/splitters.py` | 4.7.1 | 递归切分 + chunk_id |
| 19 | `src/private_teacher/rag/__init__.py` | 4.7.3 | re-export |
| 20 | `src/private_teacher/rag/embeddings.py` | 5.6.2 | 四种 provider + HashEmbedder |
| 21 | `src/private_teacher/rag/vector_store.py` | 6.6.1 | Chroma 封装 |
| 22 | `src/private_teacher/rag/manifest.py` | 7.6.1 | 索引清单 |
| 23 | `src/private_teacher/rag/indexer.py` | 7.6.2 | 全量 + 增量构建 |
| 24 | `src/private_teacher/rag/retriever.py` | 8.6.1/8.6.2 | 向量检索 + 混合检索 |
| 25 | `src/private_teacher/rag/source_resolver.py` | 9.4.1 | 溯源 |
| 26 | `src/private_teacher/services/course_service.py` | 10.4.2 | 课程编排 |
| 27 | `src/private_teacher/services/kb_service.py` | 10.4.3 | 知识库编排 |
| 28 | `src/private_teacher/services/__init__.py` | 10.4.4 | re-export |
| 29 | `src/private_teacher/ui/state.py` | 11.4.1 | Streamlit 缓存状态 |
| 30 | `src/private_teacher/app.py` | 11.4.2 | 入口 + 导航 |
| 31 | `src/private_teacher/ui/pages/1_courseware.py` | 11.4.3 | 课件管理页 |
| 32 | `src/private_teacher/ui/pages/2_knowledge_base.py` | 11.4.4 | 知识库页 |
| 33 | `.streamlit/config.toml` | 11.4.5 | Streamlit 配置 |

**测试文件**：`test_hashing.py`(2.7.10) / `test_storage.py`(2.7.9) / `test_loaders.py`(3.7.11) /
`test_splitters.py`(4.7.2) / `test_embeddings.py`(5.6.4) / `test_vector_store.py`(6.6.2) /
`test_indexer.py`(7.6.3) / `test_retriever.py`(8.6.3) / `test_source_resolver.py`(9.4.2) /
`test_services.py`(10.4.5) / `test_rag_pipeline.py`(12.1.1)

**需要修改的既有文件**：
- `src/private_teacher/config.py` → 加 5 个 embedding 字段（5.6.1）
- `src/private_teacher/utils/exceptions.py` → 加 `DuplicateDocumentError` / `ServiceError`（10.4.1）
- `tests/conftest.py` → 加 `tmp_db` / `sample_course`（2.7.8）、`fake_embedder` / `hash_llm_settings`（5.6.3）
- `.env.example` → 加 embedding 配置（5.6.1）
- `Makefile` → 加 `streamlit` / `samples` 目标（11.4.6）
- `pyproject.toml` → `uv add streamlit pdfplumber`

### 14.2 推荐的落地顺序（每一步都能独立验证）

```bash
# Day 1 —— 存储层
#   写 2.7.1 ~ 2.7.7，然后：
uv run pytest tests/unit/test_hashing.py tests/unit/test_storage.py -v

# Day 2 —— 加载器
#   写 3.7.1 ~ 3.7.9，先生成样本再测：
uv run python scripts/make_sample_courseware.py
uv run pytest tests/unit/test_loaders.py -v

# Day 3 —— 切分 + 向量化
uv run pytest tests/unit/test_splitters.py tests/unit/test_embeddings.py -v

# Day 4 —— 向量库 + 索引器
uv run pytest tests/unit/test_vector_store.py tests/unit/test_indexer.py -v

# Day 5 —— 检索 + 溯源
uv run pytest tests/unit/test_retriever.py tests/unit/test_source_resolver.py -v

# Day 6 —— Service + UI
uv run pytest tests/unit/test_services.py -v
make streamlit

# Day 7 —— 端到端 + 收尾
uv run pytest tests/integration -v
make ci
```

### 14.3 自检：这些不变量必须成立

写完之后逐条对照，任何一条不成立都说明有 bug：

1. **chunk_id 稳定**：同一份文件，重跑 100 次 `split_documents`，id 完全一致
2. **重复索引不膨胀**：连续 `build_kb` 三次，`vectors_in_store` 数量不变
3. **删除即消失**：删掉课件后立刻检索它的关键词，返回空
4. **改小即收缩**：把长文件改短后重建，向量条数下降
5. **课程隔离**：A 课的检索永远不会返回 B 课的内容
6. **溯源可达**：每条检索结果的 `location.exists` 为 True 且 `file_path.exists()`
7. **部分失败不阻塞**：一个坏文件只进 `errors`，其余文件照常索引
8. **重启不丢**：关掉进程重开，课程、课件、向量全都还在
9. **元数据纯净**：写进 Chroma 的 metadata 只有 str/int/float/bool，没有 None
10. **无泄漏**：`make ci` 全绿，覆盖率不低于 80%

---

> **下一步**：等用户确认 Phase 1 收尾，进入 Phase 2（课程分析 + 单元学习）。