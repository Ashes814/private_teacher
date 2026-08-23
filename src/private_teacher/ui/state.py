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

