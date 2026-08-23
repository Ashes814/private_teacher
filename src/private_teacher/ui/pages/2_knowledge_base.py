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