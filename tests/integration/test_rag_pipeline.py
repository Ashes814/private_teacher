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
    resolver = kb._resolver(course.id)
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
        env["conn"],
        env["course"],
        env["kb"],
        env["tmp"],
        env["paths"],
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
    course_svc.upload_document(
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
