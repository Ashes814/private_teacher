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
        f'def helper_{i}(x: float) -> float:\n    """第 {i} 个辅助函数。"""\n    return x * {i}\n'
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
