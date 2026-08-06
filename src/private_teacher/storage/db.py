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
