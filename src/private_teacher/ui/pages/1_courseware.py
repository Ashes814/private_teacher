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

with st.sidebar.expander("➕ 新建课程"), st.form("new_course", clear_on_submit=True):
    # 用 form 把多个输入框打包：
    # 不用 form 的话，每敲一个字符都会触发整个脚本 rerun（很卡）
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
            service.upload_bytes(conn, course_id, uploaded.name, uploaded.getvalue(), category)
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
                        f"{size_kb:.1f} KB · {doc.mtime:%Y-%m-%d %H:%M} · `{doc.sha256[:8]}`"
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
