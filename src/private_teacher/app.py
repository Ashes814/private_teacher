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