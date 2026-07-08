# Private Teacher — 详细开发计划（Build Plan）

> 本文基于 `private_teacher_plan.md` 的功能描述，对「私人教师大模型 App」进行系统化的工程拆解，涵盖形态选型、技术栈、目录结构、各功能模块设计、数据流、状态持久化、提示词工程、开发阶段、按功能粒度的测试方案、风险与缓解措施。**本文仅做规划，不包含实现代码。**

---

## 1. 项目概述

### 1.1 一句话定位
一个本地优先（local-first）的私人教师 Agent 系统：用户上传课件 → 自动构建本地 RAG 知识库 → 大模型基于课件规划单元 → 生成学习内容 / 单元测试 / 期末项目 → 自动批改与反馈 → 形成可追溯的学习档案。

### 1.2 核心目标
| 目标 | 衡量标准 |
| --- | --- |
| 课件即用 | PDF/PPTX/TXT/代码 等异构材料可一键入知识库 |
| 学习闭环 | 单元学习 → 测试 → 批改 → 答案解析 → 进度更新，全自动串联 |
| 本地可控 | 向量库、单元文件、答题记录、得分全部存在用户机器 |
| 来源可查 | 任意一段学习内容/答案解析都能回链到原始课件页码/行号 |
| 可扩展 | 同一门课可多次提交（最多 3 次）取最高分，跨单元累积形成终期评价 |

### 1.3 范围与非目标
**范围内**（v1.0）：课件管理、RAG 知识库、单元生成、单元测试、单元批改、期末项目、终期评价、答疑、源文件查询。
**非目标**（v1.0 不做）：多用户/账号系统、云同步、移动端、原生 OCR 复杂排版、多语言切换（中文为主）、实时协作。

---

## 2. 形态选型与技术栈

### 2.1 形态选型对比

| 形态 | 优势 | 劣势 | 推荐度 |
| --- | --- | --- | --- |
| **A. Streamlit Web（本地启动）** | 全 Python，1 天即可搭出 UI，自带文件上传 / 进度条 / 聊天，调试方便 | UI 美观度受限，复杂布局写起来累 | ⭐⭐⭐⭐⭐（MVP） |
| B. Gradio Web | 更适合对话与演示场景 | 复杂流程（多步骤表单）较弱 | ⭐⭐⭐ |
| C. Tauri（Rust + Web 前端）桌面 | 真正原生安装包，体验最好 | Python ↔ Rust 通信成本高，首版投入大 | ⭐⭐⭐⭐（v1.1 升级路径） |
| D. Electron + Python 后端 | 生态成熟 | 体积大、启动慢 | ⭐⭐ |
| E. 纯 CLI（Typer/Rich） | 极致简单，自动化友好 | 学习者友好度差 | ⭐⭐（仅作脚本入口） |

**推荐**：MVP 采用 **A（Streamlit）** 形态，保留 **C（Tauri）** 作为后续 v1.1 的桌面封装路径。所有 UI 状态以「页面/组件」为粒度，组件内调用后端 service，方便后期替换前端。

### 2.2 技术栈总览

| 维度 | 选型 | 理由 |
| --- | --- | --- |
| 语言 | Python 3.11+ | LangChain/LangGraph、Jupyter 生态原生支持 |
| 包管理 | **uv**（首选）/ Poetry | uv 速度快、锁文件简洁 |
| Agent 编排 | **LangGraph** | 计划书已点名，支持循环、状态分支、人机协作 |
| LLM 接口 | `langchain-anthropic` / `langchain-openai` / `langchain-ollama` 适配器 + 自研 factory | 多 provider 可切换 |
| 文档加载 | `unstructured`（PDF/PPTX）+ 自研 code/text loader | 覆盖课件常见格式 |
| 切分 | `RecursiveCharacterTextSplitter` + 自研 code-aware splitter | 代码文件按语义块切分 |
| Embedding | `BAAI/bge-m3`（本地 HuggingFace）或 OpenAI `text-embedding-3-large` | 离线/在线双模式 |
| 向量库 | **ChromaDB**（MVP，本地持久化）→ **Milvus Lite / Milvus**（v1.1 升级） | MVP 用 Chroma 零运维 |
| 传统元数据 | SQLite（`sqlite3` 标准库） | 存课程/单元/成绩 |
| Notebook | `nbformat` 生成与解析、`papermill` 执行、`pytest` 在子进程跑编程题 | 工业级 Jupyter 工具链 |
| 沙箱执行 | `subprocess` + `RestrictedPython`（可选） + 资源限制 | 编程题安全执行 |
| 前端 | Streamlit 1.35+，自定义组件（`streamlit-extras`、`streamlit-chat`） | 见 2.1 |
| 配置 | `pydantic-settings` + `.env` | 类型安全的配置 |
| 日志 | `loguru` | 简洁、可观测 |
| 测试 | `pytest` + `pytest-asyncio` + `pytest-mock` + `vcrpy`（录制 LLM HTTP） | 覆盖单元/集成/E2E |
| Lint/Format | `ruff` + `mypy` | 一体化 |
| CI | GitHub Actions | 跑测试 + 静态检查 |
| 文档 | MkDocs（Material 主题） | 用户与开发双视角 |

### 2.3 LLM Provider 策略

通过统一 `LLMFactory` 暴露 3 个 provider：

| Provider | 用途 | 备注 |
| --- | --- | --- |
| Anthropic Claude（Sonnet 5） | 主推理（课程分析、内容生成、批改） | 中文表现稳，质量高 |
| Ollama（本地 Qwen2.5 / DeepSeek） | 离线场景、隐私敏感用户 | 速度较慢，需 ≥16GB 显存 |
| OpenAI 兼容 | 备选，便于国内中转 | 通过 `base_url` 配置 |

默认配置走 Claude，通过 `.env` 切换。

---

## 3. 系统架构

### 3.1 逻辑视图

```
┌────────────────────────────────────────────────────────────┐
│                   Streamlit UI（前端）                      │
│ 课件 │ 知识库 │ 单元 │ 测试 │ 期末 │ 答疑 │ 进度  (Pages)   │
└────────────────────┬───────────────────────────────────────┘
                     │  service 调用（同步 / 异步任务）
┌────────────────────▼───────────────────────────────────────┐
│              Service Layer（业务编排）                      │
│  CourseService  RAGService  UnitService  TestService ...    │
└────┬──────────┬──────────┬──────────┬──────────┬──────────┘
     │          │          │          │          │
┌────▼───┐ ┌────▼────┐ ┌───▼────┐ ┌───▼────┐ ┌───▼─────┐
│ Loaders│ │   RAG   │ │ Agents │ │Notebook│ │ Storage │
│        │ │ Embed+VS│ │LangGrph│ │  nbformat│ │SQLite+FS│
└────────┘ └─────────┘ └────────┘ └────────┘ └────────┘
```

### 3.2 关键设计原则
1. **本地优先**：所有用户数据（课件、知识库、单元、成绩）落地到 `data/`，绝不外发。
2. **LLM 可替换**：所有 Agent 节点只依赖 LangChain `BaseChatModel` 抽象，provider 切换零改动。
3. **可重入**：单元测试允许多次提交，状态机记录每次得分。
4. **来源可追溯**：知识库 chunk 必须带 `(source_path, page, line_range, chunk_id)` 元数据。
5. **失败可恢复**：每个长任务（KB 构建、单元生成、批改）有 check-point 与断点续跑能力。
6. **测试可回放**：所有 LLM HTTP 调用用 `vcrpy` 录制，离线回归。

### 3.3 进程模型
- **UI 进程**：Streamlit 主进程。
- **后台任务**：通过 `asyncio` + `BackgroundTasks` 或独立 `ThreadPoolExecutor` 跑长任务（KB 构建、大单元生成）。Streamlit 用 `st.status` / `st.progress` 轮询状态。
- **沙箱子进程**：编程题执行时 `subprocess.Popen` + 资源限制（`resource`/`prlimit`）。

---

## 4. 数据流设计

### 4.1 整体数据流

```
[课件文件]──load──▶[Document]──split──▶[Chunk]──embed──▶[VectorDB]
                                                          │
                                                          │ retrieve
[学习请求]──▶[LangGraph Agent]──▶[生成内容（含 source refs）]──▶[落盘]
                                                          │
                                                          │ 用户作答
                                                          ▼
                                                [Notebook 解析]
                                                          │
                                                          ▼
                                                [自动 + LLM 混合批改]
                                                          │
                                                          ▼
                                                [成绩 → SQLite]
                                                          │
                                                          ▼
                                                [终期评价 → 报告]
```

### 4.2 关键流

**流 1：知识库构建**
1. 用户在「课件」页选择目录或上传文件。
2. `CoursewareIndexer` 扫描 → `LoaderRegistry` 选 loader → 产出 `Document` 列表（带 `source_path`、`page`）。
3. `SplitterRegistry` 选 splitter → 产出 `Chunk`（带 `chunk_id`、`line_range`）。
4. `Embedder` 批量编码 → 写入 Chroma collection（`course_{course_id}`）。
5. SQLite `courses` / `documents` 表写入元信息。

**流 2：单元学习闭环**
1. `CourseAnalyzer` Agent 调用 RAG 概览全库 → 切分单元。
2. `UnitPlanner` 为每单元生成 todo + 学习点。
3. `LearnContentGenerator` 按学习点逐条生成 Markdown，引用 chunk 来源。
4. `TestGenerator` 按题型（MC/QA/Code）调用专门 prompt 生成结构化题目 → `NotebookBuilder` 拼装 `.ipynb`。
5. 用户在 Jupyter（VSCode / 本地 JupyterLab）作答后上传。
6. `NotebookParser` 抽取作答 → `Grader`（MC 用规则、QA/Code 用 LLM+pytest 混合）→ 写 `scores`。
7. 通过阈值（≥60）后，`AnswerExplainer` 生成 `{unit}_answer.ipynb`。
8. 单元状态置 `passed`，解锁下一单元。

**流 3：终期评价**
1. 所有单元 `passed` 后，触发 `FinalProjectGenerator` 生成 `final_project.ipynb`。
2. 用户提交 → `FinalEvaluator` 综合单元分 + 项目完成度 + 答题表现 → 输出评级（A/B/C/D/F）。

---

## 5. 文件目录结构（详细）

```
private_teacher/
├── README.md
├── private_teacher_plan.md          # 原始需求
├── Build_plan.md                    # 本文档
├── pyproject.toml                   # uv/pip 项目元信息
├── uv.lock
├── Makefile                         # 常用命令：make dev / test / lint / run
├── .env.example                     # 环境变量模板
├── .gitignore
├── .python-version                  # 3.11
│
├── data/                            # 全部用户数据（.gitignore）
│   ├── courseware/                  # 用户上传的课件
│   │   └── {course_id}/
│   │       ├── main/                # 主干课程（PDF/PPTX）
│   │       └── auxiliary/           # 辅助资料（代码/TXT/扩展）
│   ├── knowledge_bases/             # 向量库持久化
│   │   └── {course_id}/
│   │       ├── chroma/              # Chroma 持久化目录
│   │       └── manifest.json        # 文档清单 + 哈希
│   ├── units/                       # 单元产物
│   │   └── {course_id}/
│   │       ├── plan.json            # 单元划分结果
│   │       ├── {unit_id}_learn.md
│   │       ├── {unit_id}_test.ipynb
│   │       ├── {unit_id}_answer.ipynb
│   │       └── scores.json          # 三次提交得分
│   ├── final/                       # 期末产物
│   │   └── {course_id}/
│   │       ├── final_project.ipynb
│   │       └── final_score.json
│   ├── submissions/                 # 用户原始作答（归档）
│   │   └── {course_id}/{unit_id}/{attempt_no}.ipynb
│   └── app.db                       # SQLite（课程/单元/成绩/日志索引）
│
├── src/
│   └── private_teacher/
│       ├── __init__.py
│       ├── main.py                  # 入口：python -m private_teacher
│       ├── app.py                   # Streamlit 入口
│       ├── config.py                # Settings（pydantic-settings）
│       │
│       ├── llm/                     # LLM 抽象层
│       │   ├── __init__.py
│       │   ├── factory.py           # LLMFactory.build(provider)
│       │   ├── claude_provider.py
│       │   ├── ollama_provider.py
│       │   ├── openai_compat_provider.py
│       │   └── token_counter.py
│       │
│       ├── loaders/                 # 课件加载
│       │   ├── __init__.py
│       │   ├── registry.py          # 根据扩展名分发
│       │   ├── pdf_loader.py        # 基于 unstructured
│       │   ├── pptx_loader.py
│       │   ├── text_loader.py
│       │   ├── code_loader.py       # 按语言切分，保留行号
│       │   ├── markdown_loader.py
│       │   └── directory_loader.py  # 递归扫描
│       │
│       ├── rag/                     # RAG 核心
│       │   ├── __init__.py
│       │   ├── embeddings.py        # Embedder 工厂
│       │   ├── vector_store.py      # ChromaStore / MilvusStore 抽象
│       │   ├── splitters.py         # Recursive + CodeAware
│       │   ├── indexer.py           # 从课件构建 KB
│       │   ├── retriever.py         # 混合检索（向量 + BM25）
│       │   ├── source_resolver.py   # chunk_id → 源文件路径/页码
│       │   └── reranker.py          # 可选 cross-encoder 重排
│       │
│       ├── agents/                  # LangGraph 智能体
│       │   ├── __init__.py
│       │   ├── state.py             # TypedDict 状态定义
│       │   ├── graphs.py            # 图装配
│       │   ├── nodes/
│       │   │   ├── __init__.py
│       │   │   ├── course_analyzer.py
│       │   │   ├── unit_planner.py
│       │   │   ├── learn_content_generator.py
│       │   │   ├── test_generator.py
│       │   │   ├── grader.py
│       │   │   ├── answer_explainer.py
│       │   │   ├── final_project_generator.py
│       │   │   └── final_evaluator.py
│       │   ├── prompts/             # 全部 prompt 模板
│       │   │   ├── course_analyzer.md
│       │   │   ├── unit_planner.md
│       │   │   ├── learn_content.md
│       │   │   ├── test_mc.md
│       │   │   ├── test_qa.md
│       │   │   ├── test_code.md
│       │   │   ├── grader_qa.md
│       │   │   ├── grader_code.md
│       │   │   ├── answer_explain.md
│       │   │   ├── final_project.md
│       │   │   ├── final_evaluate.md
│       │   │   └── qa_chat.md
│       │   └── utils/
│       │       ├── json_parser.py   # 容错解析 LLM JSON
│       │       └── validator.py     # 题目结构校验
│       │
│       ├── notebook/                # Jupyter 工具
│       │   ├── __init__.py
│       │   ├── builder.py           # 结构化 JSON → .ipynb
│       │   ├── parser.py            # .ipynb → 学生作答
│       │   ├── executor.py          # 沙箱执行 + 资源限制
│       │   ├── grader_runner.py     # 用 pytest 跑隐藏测试
│       │   └── validator.py         # 校验生成 nb 是否合法
│       │
│       ├── services/                # UI 调用的服务层
│       │   ├── __init__.py
│       │   ├── course_service.py
│       │   ├── kb_service.py
│       │   ├── unit_service.py
│       │   ├── test_service.py
│       │   ├── final_service.py
│       │   ├── qa_service.py
│       │   └── progress_service.py
│       │
│       ├── storage/                 # 持久化
│       │   ├── __init__.py
│       │   ├── db.py                # SQLite 连接 + migration
│       │   ├── models.py            # dataclass 领域模型
│       │   ├── course_repo.py
│       │   ├── unit_repo.py
│       │   ├── score_repo.py
│       │   └── path_utils.py        # data/ 下路径解析
│       │
│       ├── ui/                      # Streamlit 页面
│       │   ├── __init__.py
│       │   ├── shared.py            # 共用组件
│       │   ├── pages/
│       │   │   ├── 1_📚_courseware.py
│       │   │   ├── 2_🧠_knowledge_base.py
│       │   │   ├── 3_🗂_units.py
│       │   │   ├── 4_📝_test.py
│       │   │   ├── 5_🏆_final.py
│       │   │   ├── 6_💬_qa.py
│       │   │   └── 7_📊_progress.py
│       │   └── components/
│       │       ├── file_uploader.py
│       │       ├── source_viewer.py
│       │       ├── notebook_runner.py
│       │       ├── chat_box.py
│       │       └── score_card.py
│       │
│       └── utils/
│           ├── __init__.py
│           ├── logging.py           # loguru 配置
│           ├── hashing.py           # 文件/内容哈希
│           ├── retry.py             # tenacity 包装
│           ├── async_runner.py      # 后台任务封装
│           └── exceptions.py        # 自定义异常体系
│
├── tests/
│   ├── conftest.py                  # 共享 fixtures（假 LLM / 临时目录）
│   ├── fixtures/
│   │   ├── courseware/
│   │   │   ├── sample.pdf           # 合成小 PDF
│   │   │   ├── sample.pptx
│   │   │   ├── sample.txt
│   │   │   └── sample.py
│   │   └── cassettes/               # vcrpy 录制
│   ├── unit/
│   │   ├── test_loaders.py
│   │   ├── test_splitters.py
│   │   ├── test_vector_store.py
│   │   ├── test_indexer.py
│   │   ├── test_notebook_builder.py
│   │   ├── test_notebook_parser.py
│   │   ├── test_notebook_executor.py
│   │   ├── test_agents_course_analyzer.py
│   │   ├── test_agents_test_generator.py
│   │   ├── test_agents_grader.py
│   │   ├── test_services.py
│   │   └── test_storage.py
│   ├── integration/
│   │   ├── test_kb_build_pipeline.py
│   │   ├── test_unit_workflow.py
│   │   ├── test_grading_pipeline.py
│   │   └── test_final_evaluation.py
│   ├── e2e/
│   │   ├── test_full_course_cli.py
│   │   └── test_streamlit_smoke.py
│   └── llm_eval/
│       ├── rubric_unit.json         # 题目质量 rubric
│       └── eval_runner.py           # 离线评分脚本
│
├── scripts/
│   ├── setup_dev.sh                 # 一键安装依赖
│   ├── run_app.sh                   # streamlit run
│   ├── run_tests.sh                 # pytest
│   ├── make_sample_courseware.py    # 生成测试用课件
│   └── clean_data.sh                # 清理 data/（危险）
│
└── docs/
    ├── architecture.md
    ├── dataflow.md
    ├── prompts.md
    ├── testing.md
    ├── user_guide.md
    └── dev_guide.md
```

---

## 6. 功能模块详细设计

> 每个模块按 **目标 / 输入 / 输出 / 关键流程 / 错误处理 / 测试点** 描述。

### 6.1 课件管理（Courseware Module）

- **目标**：让用户以最少步骤把任意格式课件放到指定位置，并能预览 / 删除 / 重建索引。
- **输入**：用户选择的文件或目录（PDF/PPTX/TXT/MD/PY/JS/TS/Java/…）。
- **输出**：
  - 文件复制到 `data/courseware/{course_id}/main|auxiliary/`。
  - 写入 `documents` 表（`id`, `course_id`, `path`, `category`, `sha256`, `size`, `mtime`）。
- **关键流程**：
  1. 选择课程（若无则新建 `course_id = uuid4().hex[:8]`）。
  2. 拖拽上传 / 选择目录。
  3. 后台 `LoaderRegistry` 预校验（不支持格式给出提示）。
  4. 复制文件 + 写库。
- **错误处理**：
  - 重复文件（同 sha256）跳过并提示。
  - 格式不支持给出明确错误。
  - 单文件失败不阻塞整体。
- **测试点**：
  - 上传 PDF/TXT/PY/PPTX 各类样本均入库成功。
  - 同一文件重复上传幂等。
  - 不支持格式（.exe）返回明确错误。
  - 课程切换/删除不影响其他课程。

### 6.2 RAG 知识库（RAG Module）

- **目标**：从课件生成可检索、可溯源的本地向量库。
- **输入**：`documents` 表 + 课程 ID。
- **输出**：Chroma 集合 `course_{course_id}` + `manifest.json`。
- **关键流程**：
  1. 增量检测：仅对新增/修改的 `sha256` 文档构建。
  2. 加载 → 切分 → embedding → upsert。
  3. 保留 `metadata: {source_path, page, line_start, line_end, chunk_id, doc_id}`。
  4. BM25 索引（用 `rank_bm25`）落盘，作为混合检索补充。
- **错误处理**：
  - 单 chunk embedding 失败：标记并跳过，不阻塞整库。
  - Chroma 写失败：事务回滚 manifest。
- **测试点**：
  - 同一份课件两次构建结果一致（确定性）。
  - 修改文件后增量重建只重做变更文件。
  - 删除课件后对应 chunk 被清除。
  - chunk 元数据完整（`source_path` / `page` 不为空）。
  - 检索召回片段能反查到源文件行号/页码。

### 6.3 课程分析与单元划分（Course Analyzer + Unit Planner）

- **目标**：自动把一门课拆成 3~10 个有序单元，每个单元有 todo list。
- **输入**：RAG 概览（取全库代表性片段） + 课件元信息。
- **输出**：`data/units/{course_id}/plan.json`，结构：
  ```
  {
    "course_id": "...",
    "units": [
      {
        "id": "u1",
        "name": "Python 基础语法",
        "objectives": [...],
        "todo": [{"topic": "...", "refs": [{"chunk_id": "..."}]}],
        "prerequisites": []
      }
    ]
  }
  ```
- **关键流程**：
  1. `CourseAnalyzer` 节点：先用 RAG 取全库摘要 → LLM 输出单元结构。
  2. 输出用 JSON schema 校验，失败重试最多 3 次。
  3. `UnitPlanner` 节点：基于 RAG 检索每个单元的 top-k chunk，写 todo。
  4. 落盘 `plan.json`，触发 UI 列表。
- **错误处理**：
  - JSON 解析失败：调用 `json_parser.fix_and_parse`。
  - 单元数 < 2 或 > 15：自动合并/拆分。
  - todo 引用了不存在的 chunk：过滤。
- **测试点**：
  - 样本课程输出单元数在 [2, 15] 区间。
  - 每个单元 todo 不为空且都引用了真实 chunk。
  - 单元顺序合理（先导知识在前）。
  - 二次运行结果稳定（mock LLM）。

### 6.4 单元学习内容生成（Learn Content Generator）

- **目标**：按单元 todo 逐条生成学习 Markdown，标注来源。
- **输入**：`plan.json` 中某单元 + RAG 检索。
- **输出**：`{unit_id}_learn.md`：
  ```
  # 单元名
  ## 学习目标
  ## 知识点 1
  内容...
  > 📖 来源：main/第一章.pdf#page=12
  ## 知识点 2
  ...
  ## 知识图谱（可选）
  ```
- **关键流程**：
  1. 按 todo 逐条调用 LLM（一次一个知识点，控上下文）。
  2. prompt 强制要求返回「内容 + source_chunk_ids」。
  3. 后处理把 chunk_id 解析为「文件相对路径 + 页码 / 行号」并渲染为脚注。
  4. 整体用模板组装为最终 Markdown。
- **错误处理**：
  - LLM 漏标来源：检索回填 top-1。
  - Markdown 语法错误：用 `markdown-it-py` 校验，不通过则回退上一版。
- **测试点**：
  - 每个 todo 都有对应章节。
  - 至少 80% 段落带可点开的来源链接。
  - Markdown 可被正确渲染。
  - 重复生成内容差异在合理范围（用 LLM-eval 评分）。

### 6.5 单元测试生成（Test Generator）

- **目标**：按题型生成结构化测试题并拼装成 `.ipynb`。
- **输入**：单元 ID + 单元 todo + RAG。
- **输出**：`{unit_id}_test.ipynb`，包含：
  - **Markdown cell**：题目说明
  - **Code cell (MC)**：定义 `answer = ""` 等选项变量
  - **Markdown cell (QA)**：作答文本框
  - **Code cell (Code)**：函数骨架 + `assert` 隐藏测试入口
- **关键流程**：
  1. 调 LLM 三次分别生成 MC / QA / Code 题目（JSON）。
  2. `validator.py` 校验 JSON schema：
     - MC：选项数 ≥ 4，答案唯一性，解析唯一。
     - QA：rubric 含分值细则。
     - Code：函数签名 + 至少 1 个隐藏测试用例。
  3. `NotebookBuilder` 组装 ipynb，保留 cell tags（如 `mc`、`qa`、`code`）。
  4. 元数据写入 `plan.json`（题目数 / 分值 / 题型）。
- **错误处理**：
  - JSON 不合规：重试 + 修复 prompt。
  - 总分不为 100：自动按比例重缩放并提示。
  - 生成代码 `ast.parse` 失败：标记为 draft，不写入。
- **测试点**：
  - 生成 ipynb 能用 `jupyter nbconvert --execute` 打开。
  - 题型比例与计划一致。
  - 总分恰为 100。
  - 题目去重（不与历史重复）。
  - 题目可读性 / 答案唯一性人工 + LLM 评分（`tests/llm_eval/`）。

### 6.6 单元测试批改（Grader）

- **目标**：自动批改三类题型，给出 0~100 分。
- **输入**：用户提交的 `{unit_id}_test.ipynb`。
- **输出**：
  - `submissions/.../{attempt_no}.ipynb`（归档）
  - `scores.json` 更新：
    ```
    {
      "u1": {
        "attempts": [
          {"no":1, "mc":30, "qa":40, "code":20, "total":90, "passed":true, "ts":"..."}
        ],
        "best": 90,
        "passed": true
      }
    }
    ```
- **关键流程**：
  1. `NotebookParser` 按 cell tag 提取作答。
  2. **MC 批改**：字符串/正则匹配规则答案，1 题 X 分。
  3. **QA 批改**：调用 LLM 按 rubric 打分（prompt `grader_qa.md`），输出每题分 + 评语。
  4. **Code 批改**：
     - `executor` 在子进程（资源限制：30s / 256MB）跑用户函数。
     - `grader_runner` 用 `pytest` 跑隐藏测试，统计通过率。
     - 失败时记录 traceback。
  5. 加权汇总 → 写库 → UI 展示。
- **错误处理**：
  - 提交文件损坏/格式错误：明确错误信息并允许重新提交（不占次数？需决定，建议占次数以促使用户谨慎）。
  - 子进程崩溃：捕获并记 0 分。
  - LLM QA 评分 JSON 失败：重试。
- **测试点**：
  - MC 全对/全错/部分对分数正确。
  - QA 评分与人工评分相关性 > 0.7（用样本）。
  - Code 题：正常解 / 错误解 / 超时解 / 抛异常，分别得分符合预期。
  - 三次提交取最高分逻辑正确。
  - 沙箱：恶意代码（`os.system('rm -rf /')`）不生效。

### 6.7 答案解析生成（Answer Explainer）

- **目标**：通过测试后生成完整解题过程。
- **输入**：原测试题 + 用户最佳作答 + RAG。
- **输出**：`{unit_id}_answer.ipynb`，每题一个 cell 块含「思路 / 步骤 / 答案 / 知识点关联」。
- **关键流程**：
  1. LLM 读题 + 作答 → 输出一致性校验。
  2. 引用相关 RAG chunk 作知识点回链。
  3. 拼装 ipynb，渲染 LaTeX 公式。
- **错误处理**：
  - LLM 输出与答案不一致：以答案为准并标注存疑。
- **测试点**：
  - 每个原题都有对应解析。
  - 至少 50% 解析引用了 RAG 来源。
  - ipynb 渲染无错。

### 6.8 期末项目生成（Final Project Generator）

- **目标**：所有单元通过后，给出综合性项目/论文指导。
- **输入**：完整 `plan.json` + 全库 RAG。
- **输出**：`final_project.ipynb`：
  - 项目背景 / 目标
  - 任务拆解（多阶段，每阶段对应一个或多个单元知识点）
  - 评分标准（创新 / 完成度 / 代码质量 / 文档）
  - 提交要求 cell
- **关键流程**：
  1. 调 LLM 设计项目主题（与课程强相关、跨单元、可量化）。
  2. 二次生成项目骨架代码 / 论文大纲。
  3. 嵌入评分 rubric。
- **错误处理**：
  - 主题太宽泛/太狭窄：LLM 自评 + 自动迭代。
- **测试点**：
  - 项目覆盖 ≥ 80% 单元。
  - 评分项可量化。
  - ipynb 可执行。

### 6.9 终期评价（Final Evaluator）

- **目标**：综合单元分 + 项目完成度 → 总评。
- **输入**：
  - 全部单元 best 得分
  - 项目提交 + 自评
- **输出**：`final_score.json`：
  ```
  {
    "unit_avg": 85,
    "project": {"score": 80, "breakdown": {...}},
    "final_grade": "B",
    "summary": "..."
  }
  ```
- **关键流程**：
  1. 计算单元平均分（权重可配）。
  2. 项目按 rubric 跑代码 + LLM 评阅产出文字部分。
  3. LLM 生成「综合评语 + 改进建议」。
  4. 等级映射：A(≥90) / B(80-89) / C(70-79) / D(60-69) / F(<60)。
- **错误处理**：
  - 项目未提交：仅按单元分评。
- **测试点**：
  - 分数与等级映射正确。
  - 评语引用了具体单元/题目。
  - 边界值（恰好 60、恰好 90）正确归类。

### 6.10 答疑对话（QA Chat）

- **目标**：学生可就课件内容自由提问，答案有出处。
- **输入**：用户问题 + 选中的课程。
- **输出**：带 source 链接的回答。
- **关键流程**：
  1. RAG 检索 top-k → rerank → 拼上下文。
  2. LLM 生成回答 + 引用 chunk_id。
  3. UI 渲染时把 chunk_id 解析为可点击的源文件链接。
- **错误处理**：
  - 检索无结果：诚实回答「未在课件中找到」并建议补充课件。
  - 引用 id 失效：过滤并提示。
- **测试点**：
  - 简单事实问题能精确回链到 PDF 页码。
  - 多跳问题（如「A 概念和 B 概念的关系」）能综合多 chunk。
  - 幻觉率：抽样 50 题人工评估 < 15%。

### 6.11 源文件查询（Source Viewer）

- **目标**：在学习/测试/答案任意界面都能跳到源文件对应位置。
- **输入**：chunk 元数据 `(source_path, page, line_start, line_end)`。
- **输出**：
  - 文本文件：渲染对应行（带高亮）。
  - PDF：用 `streamlit-pdf-viewer` 或 `pdf.js` 跳页。
  - PPTX：转图片 + 跳页。
  - 代码：渲染对应行区间。
- **错误处理**：
  - 文件已删除：友好提示「课件已移除」。
- **测试点**：
  - 每种文件类型都能正确跳转。
  - 高亮位置准确（行号偏差 ≤ 1）。

### 6.12 进度与档案（Progress）

- **目标**：可视化学习曲线、单元状态、得分趋势。
- **输入**：`scores` / `units` / `final_score`。
- **输出**：仪表盘（单元完成度、雷达图、时间线）。
- **测试点**：
  - 图表数据与 SQLite 一致。
  - 无数据时给出空态引导。

---

## 7. 状态管理与持久化

### 7.1 SQLite Schema（`data/app.db`）

```sql
CREATE TABLE courses (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  description TEXT,
  created_at TIMESTAMP,
  updated_at TIMESTAMP
);

CREATE TABLE documents (
  id TEXT PRIMARY KEY,
  course_id TEXT REFERENCES courses(id) ON DELETE CASCADE,
  path TEXT NOT NULL,
  category TEXT CHECK(category IN ('main','auxiliary')),
  sha256 TEXT NOT NULL,
  size INTEGER,
  mtime TIMESTAMP,
  indexed_at TIMESTAMP
);

CREATE TABLE units (
  id TEXT PRIMARY KEY,           -- u1/u2/...
  course_id TEXT REFERENCES courses(id) ON DELETE CASCADE,
  name TEXT,
  objectives TEXT,               -- JSON
  todo TEXT,                     -- JSON
  order_idx INTEGER,
  status TEXT CHECK(status IN ('locked','available','in_progress','passed')),
  best_score INTEGER,
  attempts INTEGER DEFAULT 0,
  updated_at TIMESTAMP
);

CREATE TABLE scores (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  unit_id TEXT REFERENCES units(id) ON DELETE CASCADE,
  attempt_no INTEGER,
  mc_score INTEGER,
  qa_score INTEGER,
  code_score INTEGER,
  total INTEGER,
  passed INTEGER,
  details TEXT,                  -- JSON
  ts TIMESTAMP
);

CREATE TABLE final_results (
  course_id TEXT PRIMARY KEY REFERENCES courses(id) ON DELETE CASCADE,
  unit_avg REAL,
  project_score REAL,
  final_grade TEXT,
  summary TEXT,
  ts TIMESTAMP
);
```

### 7.2 文件系统约定
- 所有用户数据落 `data/`。
- 命名规范：`{course_id}/{unit_id}_{type}.{ext}`。
- 不可变文件：用户原始课件、用户提交（仅追加）。
- 可重建文件：单元 Markdown、测试、答案、终期 → 都可一键重生成（不可变的是分数与提交）。

### 7.3 迁移
- 使用简单 `schema_migrations` 表 + 顺序 SQL 文件（不引入 Alembic，保持轻量）。
- 每次启动应用跑 `db.migrate()`。

---

## 8. 提示词工程

### 8.1 提示词管理
- 所有 prompt 模板放在 `src/private_teacher/agents/prompts/*.md`。
- 用 frontmatter 记录版本、模型推荐、token 估算。
- 关键 prompt 走 LangChain `ChatPromptTemplate`，支持 `{chat_history}` / `{context}` / `{variable}`。

### 8.2 提示词关键约束
- **结构化输出**：要求 LLM 返回严格 JSON（schema in prompt）。
- **回链强制**：每条内容必须带 `source_chunk_ids` 字段，否则后处理回填。
- **题目防泄漏**：QA/Code 题目 prompt 中禁止暴露答案。
- **可重入**：prompt 包含「之前生成过、避免重复」的历史摘要。

### 8.3 关键 prompt 清单（见 `prompts/`）
| 文件 | 用途 |
| --- | --- |
| `course_analyzer.md` | 切分单元 |
| `unit_planner.md` | 生成 todo |
| `learn_content.md` | 知识点学习内容 |
| `test_mc.md` | 选择题生成 |
| `test_qa.md` | 问答题生成 |
| `test_code.md` | 编程题生成 |
| `grader_qa.md` | 问答题评分 |
| `grader_code.md` | 编程题辅助评分（rubric） |
| `answer_explain.md` | 答案解析 |
| `final_project.md` | 期末项目设计 |
| `final_evaluate.md` | 终期评价 |
| `qa_chat.md` | 答疑 |

---

## 9. 开发阶段（按周推进）

> 总周期建议 **8 周**（单人；双人 5 周）。每个阶段结束都有可演示产物。

### Phase 0 — 基础设施（Week 1）
- 项目骨架、`pyproject.toml`、`uv` 初始化。
- `config.py` + `.env.example`。
- `llm/factory.py` 三个 provider 跑通最小 demo。
- `utils/logging`、`utils/retry`、自定义异常。
- 测试基建：`conftest.py`、假 LLM fixture、CI 跑 lint + 单测。
- **可演示**：`python -m private_teacher hello-llm` 输出 LLM 回复。

### Phase 1 — 课件 + RAG（Week 2）
- `loaders/*` 实现 PDF/PPTX/TXT/Code/MD。
- `rag/splitters` + `embeddings` + `vector_store` (Chroma)。
- `rag/indexer` 增量构建。
- `rag/retriever` + `source_resolver`。
- Streamlit 课件页 + 知识库页（显示文档列表 / 检索 demo）。
- **可演示**：上传 PDF → 检索「机器学习」→ 给出片段 + 来源。

### Phase 2 — 单元学习（Week 3）
- `agents/course_analyzer` + `unit_planner`。
- `agents/learn_content_generator`。
- Streamlit 单元列表 + 学习页（渲染 Markdown + 来源链接）。
- **可演示**：点击某单元 → 看到完整学习内容 + 可跳转源文件。

### Phase 3 — 单元测试（Week 4）
- `agents/test_generator`（MC/QA/Code 三个 prompt）。
- `notebook/builder` 生成 ipynb。
- Streamlit 测试下载页。
- **可演示**：下载 `{unit}_test.ipynb` → 用 VSCode 打开看到题目。

### Phase 4 — 批改与答案（Week 5）
- `notebook/parser` 抽取作答。
- `notebook/executor` 沙箱 + `grader_runner`。
- `agents/grader`（MC/QA/Code）。
- `agents/answer_explainer`。
- Streamlit 提交 + 成绩页 + 答案查看。
- **可演示**：提交测试 → 看到分数 → 通过后查看解析。

### Phase 5 — 期末（Week 6）
- `final_project_generator`。
- `final_evaluator`。
- 终期报告页 + 进度仪表盘。
- **可演示**：完成所有单元 → 生成期末项目 → 提交 → 拿评级。

### Phase 6 — 答疑 + 源文件（Week 7）
- `qa_service` + `qa_chat` prompt。
- `ui/components/source_viewer` 多格式。
- 流式回答 + 对话历史。
- **可演示**：在答疑页提问 → 拿到带源链接的回答。

### Phase 7 — 打磨与发布（Week 8）
- 异常处理、撤销、加载状态、移动端适配。
- 文档（`docs/`）+ README + 截图。
- `make install && make run` 一键启动脚本。
- v0.1.0 tag 发布。

---

## 10. 测试策略（按功能粒度）

> 总原则：**每个小功能至少有 1 个单测 + 1 个集成测**；LLM 交互用 `vcrpy` 录制或假 LLM 替换；端到端用样本课件跑通。

### 10.1 通用测试基建
- `conftest.py`：
  - `tmp_course_dir` 临时课程目录 fixture
  - `fake_llm` 返回预设 JSON 的假 LLM
  - `sample_pdf/pptx/txt/py` 样本文件
  - `cassettes/` 录制真实 LLM 调用
- `pytest -m unit|integration|e2e` 区分快慢测试。

### 10.2 各模块测试点（与第 6 章一一对应）

| 模块 | 单测 | 集成测 |
| --- | --- | --- |
| 课件管理 | 各类文件入库；不支持格式报错；幂等 | 上传 → 落盘 → 列表正确 |
| RAG | 单文件构建；增量构建；元数据完整 | 真实样本课构建 → 检索命中 |
| 课程分析 | JSON schema 校验；chunk 引用过滤 | 样本课产出 plan.json 结构合理 |
| 学习内容 | Markdown 渲染；来源回链 | 样本单元产出 learn.md |
| 测试生成 | 题目 JSON 校验；总分=100 | 端到端生成可执行 ipynb |
| 批改 | MC/QA/Code 单题打分；沙箱防逃逸 | 提交样本 ipynb → 分数符合预期 |
| 答案解析 | 题目-解析一一对应 | 与测试题同源 |
| 期末项目 | 覆盖单元数；rubric 可量化 | 完整课程生成项目 ipynb |
| 终期评价 | 等级映射；边界值 | 完整数据生成报告 |
| 答疑 | 检索 top-k；引用渲染 | 真实问答得分（LLM eval） |
| 源文件 | 每类文件跳转坐标正确 | UI 跳转链路 |
| 进度 | 空态；图表数据一致 | 全流程跑完数据对齐 |

### 10.3 专项测试
- **沙箱安全测试**（`tests/integration/test_sandbox.py`）：
  - `os.system`、`subprocess`、`open('/etc/passwd')`、死循环、爆内存均被拦截/超时。
- **LLM 质量评测**（`tests/llm_eval/`）：
  - 维护 50~100 题「金标准」题库，跑 `eval_runner.py` 给出平均分。
  - MC 答案唯一性 / QA 与人工评分相关系数 / Code 通过率。
- **回归测试**：
  - 用 vcrpy 录制所有 LLM HTTP 调用，CI 离线跑（节省成本）。
  - 每次 prompt 改版必须跑 `tests/llm_eval/` 评分。
- **性能基线**（`tests/perf/`，可选）：
  - 100 页 PDF 索引时间、单元生成端到端时间、答疑延迟。

### 10.4 CI 流水线
- PR 触发：`ruff check`、`mypy`、`pytest -m unit`。
- 每日定时：`pytest -m integration` + `llm_eval`（夜间，离线 cassettes）。
- 手动触发：E2E（需要真 LLM key）。

---

## 11. 风险与缓解

| 风险 | 影响 | 缓解 |
| --- | --- | --- |
| LLM 生成题目答案错或可被搜到 | 教学价值受损 | 题干重写 + 抽检人工 + LLM 评分脚本 |
| 编程题沙箱被逃逸 | 主机安全 | `subprocess` + 资源限制 + 网络隔离 + 容器（v1.1） |
| Chroma 写入失败导致知识库损坏 | 学习中断 | 写入前 manifest 备份；定期导出 |
| LLM API 限流/中断 | 长时间任务失败 | tenacity 指数退避 + 断点续跑 + 多 provider 切换 |
| 大课件（>1000 页）首次索引慢 | 体验差 | 流式构建 + 进度条 + 缓存中间结果 |
| Notebook 执行结果不确定 | 批改不稳定 | 设置随机种子 + 多次执行取众数（v1.1） |
| 不同 PDF 解析质量差异 | RAG 质量不稳 | `unstructured` 分层 fallback；样本人工抽检 |
| 提示词漂移 | 题目/内容质量波动 | prompt 版本号 + eval 基线对比 |
| LLM 中文输出偶发英文 | 体验割裂 | prompt 显式约束 + 后处理替换 |

---

## 12. 文档与交付物

| 文档 | 读者 | 内容 |
| --- | --- | --- |
| `docs/architecture.md` | 开发者 | 模块图、数据流、依赖 |
| `docs/dataflow.md` | 开发者 | 端到端时序图 |
| `docs/prompts.md` | 开发者 | prompt 编写规范、当前版本 |
| `docs/testing.md` | 开发者 | 测试结构、跑测命令 |
| `docs/user_guide.md` | 终端用户 | 上传课件、学习、查看成绩 |
| `docs/dev_guide.md` | 贡献者 | 本地开发、提 PR 流程 |
| `README.md` | 所有人 | 项目简介 + 快速开始 |

---

## 13. 验收清单（MVP 完成定义）

- [x] 用户可上传 PDF/PPTX/TXT/代码，并触发知识库构建
- [ ] 一门样本课可在 60s 内构建完成（取决于 LLM）
- [ ] 自动产出 3~10 个单元及对应的 `learn.md`
- [ ] 每个单元产出可执行 `test.ipynb`（含 MC/QA/Code）
- [ ] 用户提交 3 次后系统记录最高分，≥60 通过
- [ ] 通过后产出 `answer.ipynb`
- [ ] 所有单元通过后可生成 `final_project.ipynb` 并得到终期评级
- [ ] 答疑页能基于 RAG 给出带源链接的回答
- [ ] 任意内容可跳转到源文件对应页/行
- [ ] 全流程 E2E 测试通过
- [ ] CI 跑通 lint + 单测 + 集成测
- [ ] 文档齐备

---

## 14. 后续迭代路线（v1.1+）

- Tauri 桌面封装（脱离浏览器）
- 多用户/多课程并行
- 题目难度自适应（根据历史表现调题）
- 学习时间统计 + 学习曲线
- 课程市场（用户共享课件 + RAG）
- 容器化沙箱（Firecracker / gVisor）
- 多模态：图片/公式 OCR 入库
- 微信/邮件订阅学习提醒

---

> **下一步**：等用户确认本计划后，进入 Phase 0 实施（搭骨架 + 写最小 demo）。
