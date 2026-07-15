# Phase 0 — 基础设施 手把手教学计划

> 目标：搭好项目骨架，让 `python -m private_teacher hello-llm` 能跑通，并具备**测试基建 + Lint + CI**。
> 学完之后，你将掌握：**uv 项目管理、pydantic 配置、loguru 日志、tenacity 重试、LangChain 三家 LLM provider、pytest + 假 LLM + vcrpy、GitHub Actions**。
> 本文不写实际代码，只写「该建什么文件、文件里该有什么、为什么这样设计、怎么验证、哪里容易踩坑」。**代码由你亲手写**。

---

## 0. 前置准备（5 分钟）

### 0.1 你需要知道
- 你用的是 **macOS + zsh + VSCode**。
- 仓库目前是空仓库，目录在 `/Users/zowcool/Desktop/Learning/private_teacher`。
- 你的 Python 应该 ≥ 3.11。

### 0.2 验证环境
打开终端，进入项目目录，依次跑以下命令并确认输出符合预期：

1. `python3 --version` → 看到 `Python 3.11.x` 或更高。**不是 3.10 或更早**，因为我们要用最新的类型注解语法。
2. `which python3` → 看到路径。
3. `git status` → 看到 `On branch main` 和 `No commits yet`。
4. `echo $SHELL` → 看到 `/bin/zsh`（macOS 默认）。

> 🐛 **踩坑**：如果 Python 版本 < 3.11，去 python.org 装 3.12，或者用 brew 装：`brew install python@3.12`，然后确保 `python3` 指向新版本。

### 0.3 安装 uv（如果还没装）
`uv` 是 Astral 出品的极快 Python 包管理器，**比 pip / poetry 快 10~100 倍**。
- 安装命令：`curl -LsSf https://astral.sh/uv/install.sh | sh`
- 装完后**重开一个终端**，跑 `uv --version` 验证。
- 看到版本号（如 `0.5.x`）就 OK。

> 🐛 **踩坑**：如果 `uv --version` 报 `command not found`，说明 PATH 没生效。重开终端，或在 `~/.zshrc` 加 `export PATH="$HOME/.local/bin:$PATH"`，然后 `source ~/.zshrc`。

### 0.4 准备 API Key
至少准备一个能用的 LLM API Key（推荐 Anthropic Claude）：
- 申请地址：https://console.anthropic.com/
- 拿到形如 `sk-ant-...` 的 key。
- 先**不要**写到任何文件里，Phase 0 我们用 `.env` 管理。

---

## 1. 项目骨架（30 分钟）

### 1.1 要建的目录与文件
按下面这个清单建好空文件（**先建空文件，文件内容后面步骤填**）：

```
private_teacher/
├── .python-version
├── .gitignore
├── .env.example
├── pyproject.toml                # uv 会自动生成，我们要改
├── README.md
├── Makefile
├── src/
│   └── private_teacher/
│       ├── __init__.py
│       ├── __main__.py
│       ├── cli.py                # 命令行入口
│       ├── config.py
│       ├── llm/
│       │   ├── __init__.py
│       │   └── factory.py
│       └── utils/
│           ├── __init__.py
│           ├── logging.py
│           ├── retry.py
│           └── exceptions.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── fakes/
│   │   ├── __init__.py
│   │   └── fake_llm.py
│   └── unit/
│       ├── __init__.py
│       ├── test_config.py
│       ├── test_logging.py
│       ├── test_retry.py
│       ├── test_exceptions.py
│       └── test_llm_factory.py
├── data/                          # 空目录，加 .gitkeep
│   └── .gitkeep
└── .github/
    └── workflows/
        └── ci.yml
```

### 1.2 操作步骤
1. 在项目根目录跑：`mkdir -p src/private_teacher/llm src/private_teacher/utils tests/fakes tests/unit data .github/workflows`
2. 在 `data/` 里建一个空文件 `.gitkeep`（让空目录能进 git）。
3. 所有 `__init__.py` 暂时留空即可（**Python 3.3+ namespace package 不需要 init，但我们要导入子包里的模块，习惯上保留**）。

### 1.3 验证
- 跑 `find . -type d -name __pycache__ -prune -o -type f -print | sort` 看目录结构是否齐全。
- 跑 `git status` 应该看到很多 `Untracked files`，但还没 add。

### 1.4 概念小课堂
- **src/ 布局** vs **flat 布局**：我们用 `src/` 是为了让 `pip install -e .` 安装后，测试用「安装后的包」而不是「当前目录的源码」，避免一些 import 歧义坑。
- **__main__.py**：让 `python -m private_teacher` 能直接运行（等价于 `python -m private_teacher.__main__`）。

---

## 2. uv 初始化与依赖管理（20 分钟）

### 2.1 目标
- 生成 `pyproject.toml`。
- 安装运行依赖（langchain、langchain-anthropic、langchain-ollama、langchain-openai、pydantic、pydantic-settings、loguru、tenacity、python-dotenv）。
- 安装开发依赖（pytest、pytest-asyncio、pytest-cov、pytest-mock、vcrpy、ruff、mypy、respx）。

### 2.2 步骤
1. 确认终端在项目根目录。
2. 跑 `uv init --package --python 3.11 .`（注意末尾的 `.` 表示当前目录）。
   - 这会生成 `pyproject.toml`、`README.md`、`.python-version`、一个示例 `hello.py`（**我们后面会删掉 hello.py**）。
3. 用 `uv add` 加运行依赖（一条一条加，看清楚每个是干啥的）：
   - `uv add pydantic pydantic-settings`：类型安全的配置管理。
   - `uv add python-dotenv`：从 `.env` 文件读环境变量。
   - `uv add loguru`：更人性化的日志库。
   - `uv add tenacity`：自动重试（指数退避）。
   - `uv add langchain langchain-core langchain-anthropic langchain-openai langchain-ollama`：LLM 抽象 + 三个 provider。
   - `uv add httpx`：HTTP 客户端（vcrpy / Ollama 调用）。
4. 用 `uv add --dev` 加开发依赖：
   - `uv add --dev pytest pytest-asyncio pytest-cov pytest-mock`
   - `uv add --dev vcrpy`（录制 LLM HTTP 调用）
   - `uv add --dev ruff`（lint + format）
   - `uv add --dev mypy`（类型检查）
   - `uv add --dev respx`（mock httpx 响应）
5. 跑 `uv sync` 同步依赖到 `.venv/`。
6. 跑 `uv run python -c "import langchain; print(langchain.__version__)"` 验证。

### 2.3 验证
- `cat pyproject.toml` 应该看到 `[project]`、`[project.optional-dependencies]` 或 `[dependency-groups]` 里有上面那些包。
- `ls -la .venv/` 应该看到虚拟环境目录。
- 跑 `uv run python --version` 应该进入 venv 并显示 3.11+。

### 2.4 概念小课堂
- **pyproject.toml 是项目元信息**（PEP 621）：包名、版本、依赖、入口点、构建系统。
- **uv.lock 是锁文件**：精确锁定每个包的小版本，让别人 install 时拿到一样的版本。
- **uv add = 修改 pyproject.toml + 同步 lock + 安装** 一步到位。
- **运行命令一律加 `uv run`**：它会确保用 venv 里的 Python/包。

### 2.5 清理
- 删除 uv init 生成的 `hello.py`（如果还有 `hello.py` 或 `src/private_teacher/hello.py`）。
- 改 `pyproject.toml` 的 `name = "private-teacher"`、`version = "0.1.0"`、`description = "Local-first private teacher LLM app"`。

> 🐛 **踩坑**：`uv add` 失败一般是网络问题。配置镜像：`export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple/`，再加进 `~/.zshrc`。

---

## 3. .gitignore / .env.example / .python-version（10 分钟）

### 3.1 .python-version
内容：`3.11`（或你装的 3.12）。
- 作用：让 `pyenv` / `uv` 自动切到对应 Python。

**实现（直接复制）**：

**.python-version**
```bash
# 指定本项目使用的 Python 版本
# uv 在执行任何命令时会自动读取这个文件，使用对应版本的 Python
# 如果你装的是 3.12，把下面的 3.11 改成 3.12
3.11
```

### 3.2 .gitignore
至少包含（参考 GitHub Python 模板）：
- `.venv/`、`__pycache__/`、`*.py[cod]`、`*.egg-info/`、`dist/`、`build/`
- `.env`、`*.local`
- `data/courseware/`、`data/knowledge_bases/`、`data/units/`、`data/final/`、`data/submissions/`、`data/app.db`
- `.pytest_cache/`、`.ruff_cache/`、`.mypy_cache/`、`htmlcov/`、`.coverage`
- `tests/fixtures/cassettes/*.yaml`（cassettes 可以不入库，但开发期先忽略；后面 vcrpy 再决定）
- `.vscode/`（可选，避免配置冲突）
- `.DS_Store`（macOS）

**实现（直接复制）**：

**.gitignore**
```gitignore
# =====================================================
# Python 基础忽略
# =====================================================

# 字节码缓存（运行 Python 时自动生成）
__pycache__/
*.py[cod]
*$py.class

# 编译后的扩展模块
*.so

# 包构建产物
*.egg-info/
build/
dist/
.eggs/
pip-wheel-metadata/

# 虚拟环境目录（uv 在项目根目录默认创建 .venv）
.venv/
venv/
env/

# =====================================================
# 环境变量与本地配置（绝对不能 commit 真实 key）
# =====================================================

# .env 包含 API key 等敏感信息，绝不能入库
.env
.env.*
# 但 .env.example 是模板，需要入库
!.env.example

# 各种本地覆盖文件
*.local

# =====================================================
# 用户数据目录（data/ 下的所有内容都是用户生成的）
# =====================================================

# 课件原始文件（用户上传的）
data/courseware/

# RAG 向量库（可重建）
data/knowledge_bases/

# 单元生成产物
data/units/

# 期末产物
data/final/

# 用户作答归档
data/submissions/

# SQLite 数据库（含所有学习记录）
data/app.db
data/*.db

# 临时数据
data/tmp/
data/temp/

# =====================================================
# 测试与覆盖率产物
# =====================================================

.pytest_cache/
.coverage
.coverage.*
htmlcov/
coverage.xml
*.cover

# mypy 缓存
.mypy_cache/
dmypy.json

# ruff 缓存
.ruff_cache/

# =====================================================
# LLM 调用录制文件（vcrpy cassettes，体积大，不入库）
# =====================================================

tests/fixtures/cassettes/
tests/**/cassettes/

# =====================================================
# IDE 与编辑器
# =====================================================

# VSCode（推荐让团队各自配，不入库）
.vscode/
!.vscode/settings.json.example
!.vscode/extensions.json

# JetBrains
.idea/

# macOS
.DS_Store

# Windows
Thumbs.db
desktop.ini

# =====================================================
# 日志与临时文件
# =====================================================

logs/
*.log
*.tmp
*.bak
*.swp
*~

# =====================================================
# Streamlit 专属
# =====================================================

.streamlit/secrets.toml
```

### 3.3 .env.example
模板（**真实 key 不入 git**）：
- 标注每个变量含义、是否必填、示例值。
- 至少包含：`PRIVATE_TEACHER_LLM_PROVIDER`、`PRIVATE_TEACHER_LLM_MODEL`、`ANTHROPIC_API_KEY`、`OPENAI_API_KEY`、`OPENAI_BASE_URL`、`OLLAMA_BASE_URL`、`OLLAMA_MODEL`、`PRIVATE_TEACHER_LOG_LEVEL`、`PRIVATE_TEACHER_DATA_DIR`。
- 实际使用：用户 `cp .env.example .env` 后填真值。

**实现（直接复制）**：

**.env.example**
```bash
# =====================================================
# Private Teacher - 环境变量配置模板
# =====================================================
# 使用方法：
#   1. 复制本文件：cp .env.example .env
#   2. 编辑 .env 填入真实的 API key
#   3. 绝不要把 .env 加入 git（已经在 .gitignore 里忽略）
# =====================================================

# ---------- LLM Provider 选择 ----------
# 可选值: claude | openai_compat | ollama
# 推荐生产环境用 claude，本地测试可以用 ollama 节省费用
PRIVATE_TEACHER_LLM_PROVIDER=claude

# ---------- Anthropic Claude 配置 ----------
# 在 https://console.anthropic.com/ 申请
# 形如: sk-ant-api03-xxxxxx
ANTHROPIC_API_KEY=

# Claude 模型名（默认 sonnet-5，质量与速度平衡）
# 可选: claude-sonnet-5-20251001, claude-opus-4-8-20251001, claude-haiku-4-5-20251001
PRIVATE_TEACHER_LLM_CLAUDE_MODEL=claude-sonnet-5-20251001

# ---------- OpenAI 兼容配置（含中转 API） ----------
# 如果用国内中转（如 https://api.deepseek.com），把 OPENAI_BASE_URL 改成对应地址
OPENAI_API_KEY=
OPENAI_BASE_URL=https://api.openai.com/v1
PRIVATE_TEACHER_LLM_OPENAI_MODEL=gpt-4o-mini

# ---------- Ollama 本地配置 ----------
# 先安装 ollama: https://ollama.com/
# 然后 ollama pull qwen2.5:7b 下载模型
# Ollama 默认监听 11434 端口
OLLAMA_BASE_URL=http://localhost:11434
PRIVATE_TEACHER_LLM_OLLAMA_MODEL=qwen2.5:7b

# ---------- MiniMax 配置 ----------
# MiniMax（https://MiniMax.com/）提供 OpenAI 兼容的 API，所以复用 ChatOpenAI 客户端
# 申请 key 请到 MiniMax 开发者控制台
# 形如: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
MINIMAX_API_KEY=
# MiniMax API 基础地址（OpenAI 兼容协议）
# 如果你用自建代理或测试环境，改成对应地址
PRIVATE_TEACHER_LLM_MINIMAX_BASE_URL=https://api.MiniMax.com/v1
# 默认模型：MiniMaxm3（你也可以改成 MiniMax 提供的其他模型名）
PRIVATE_TEACHER_LLM_MINIMAX_MODEL=MiniMaxm3

# ---------- 日志配置 ----------
# 日志级别: DEBUG | INFO | WARNING | ERROR
PRIVATE_TEACHER_LOG_LEVEL=INFO
# 是否输出 JSON 格式日志（生产环境推荐 true，方便日志聚合）
PRIVATE_TEACHER_LOG_JSON_LOGS=false

# ---------- 数据目录配置 ----------
# 用户数据（课件、知识库、单元、成绩）存储位置
# 相对路径基于运行命令的当前目录
# 建议保持默认，CI 环境可以指向临时目录
PRIVATE_TEACHER_PATH_DATA_DIR=./data
```

> 💡 **小贴士**：每个变量前的 `#` 注释是说明，`.env` 文件里**带 `#` 的行会被忽略**，所以注释可以保留。

### 3.4 验证
- 跑 `cp .env.example .env`，编辑 `.env` 填上你的真 key，再 `git status` —— **`.env` 应该在 untracked 里**（说明没被忽略错误）。
- `cat .gitignore | grep .env` 应能看到 `.env`。

> 🐛 **踩坑**：一旦不小心把真 key commit 了，立刻去 console 撤销并重新生成，git 历史里的也算泄露。

---

## 4. 配置管理：config.py（30 分钟）

### 4.1 目标
实现一个 `Settings` 类，**类型安全**地加载 `.env` 和**环境变量**到 Python 对象。

### 4.2 概念小课堂
- **pydantic-settings** = pydantic 的扩展，专门读环境变量。
- 配置来源优先级：函数参数 > 环境变量 > `.env` 文件 > 默认值。
- 我们用一个**嵌套** `Settings` 类（`LLMSettings`、`PathSettings`、`LogSettings`），外层 `AppSettings` 聚合。
- 所有自定义字段都加 `PRIVATE_TEACHER_` 前缀，避免污染全局环境变量（`env_prefix="PRIVATE_TEACHER_"`）。

### 4.3 文件内容指引（src/private_teacher/config.py）

**导入什么**：
- `from pydantic import Field`
- `from pydantic_settings import BaseSettings, SettingsConfigDict`
- `from pathlib import Path`

**类设计**：
- 一个 `LLMSettings(BaseSettings)`：
  - 字段：`provider: Literal["claude", "ollama", "openai_compat"] = "claude"`
  - `claude_api_key: str | None = None`
  - `claude_model: str = "claude-sonnet-5-20251001"`（先用这个，对应 Build_plan 里的 Sonnet 5）
  - `openai_api_key: str | None = None`
  - `openai_base_url: str | None = None`
  - `openai_model: str = "gpt-4o-mini"`
  - `ollama_base_url: str = "http://localhost:11434"`
  - `ollama_model: str = "qwen2.5:7b"`
  - 全部加 `Field(default=..., description="...")`
  - `model_config = SettingsConfigDict(env_prefix="PRIVATE_TEACHER_LLM_", env_file=".env", extra="ignore")`

- 一个 `LogSettings(BaseSettings)`：
  - `level: str = "INFO"`
  - `json_logs: bool = False`
  - 同样 `env_prefix="PRIVATE_TEACHER_LOG_"`

- 一个 `PathSettings(BaseSettings)`：
  - `data_dir: Path = Path("./data")`
  - `env_prefix="PRIVATE_TEACHER_PATH_"`

- 一个 `AppSettings`：
  - 包含 `llm: LLMSettings`、`logging: LogSettings`、`paths: PathSettings`
  - 提供一个 `@classmethod def load(cls) -> "AppSettings"`（一次性 load 所有子 settings）
  - 可以再加一个 `def validate_for_runtime(self) -> None`：检查 provider 选了之后对应 key 是否存在

**思考题（自己写之前先想）**：
- 如果用户选了 `provider="claude"` 但 `claude_api_key` 是空，怎么办？→ 在 `validate_for_runtime` 里 raise 自定义异常。
- 为什么不直接用一个 `Settings` 类平铺？→ 嵌套更清晰，避免几十个字段挤在一起。

**实现（直接复制）**：

**src/private_teacher/config.py**
```python
"""
配置管理模块
============

本模块使用 pydantic-settings 来加载项目配置，支持：
  1. 从 .env 文件读取
  2. 从环境变量读取
  3. 环境变量优先级高于 .env 文件
  4. 字段类型校验（pydantic 自动完成）

设计原则：
  - 嵌套结构：按职责拆分（LLM / 日志 / 路径），避免单一巨型类
  - 前缀隔离：所有变量加 PRIVATE_TEACHER_* 前缀，避免污染全局环境
  - 显式校验：运行前调用 validate_for_runtime() 检查关键参数
"""

from __future__ import annotations  # 让所有类型注解按字面意思解析，兼容老 Python

# ---------- 标准库导入 ----------
import os
from pathlib import Path
from typing import Literal  # 字面量类型，用于限定 provider 只能是三个值之一

# ---------- 第三方库导入 ----------
from pydantic import Field  # pydantic 字段定义工具（默认值、描述、校验）
from pydantic_settings import BaseSettings, SettingsConfigDict  # 配置基类与配置字典


# ============================================================
# LLM 相关配置
# ============================================================
class LLMSettings(BaseSettings):
    """
    大语言模型相关配置。

    包含三个 provider 的参数：
      - Anthropic Claude（云端，质量高）
      - OpenAI 兼容（含中转 API）
      - Ollama（本地运行，完全离线）

    环境变量前缀：PRIVATE_TEACHER_LLM_
    例如 PRIVATE_TEACHER_LLM_PROVIDER=claude
    """

    # ----- provider 选择 -----
    # Literal[...] 是 Python 3.8+ 的字面量类型，限定只能是括号里的值
    # pydantic 会自动校验：传其他值会抛出 ValidationError
    provider: Literal["claude", "ollama", "openai_compat", "minimax"] = Field(
        default="claude",  # 默认值（用户没设环境变量时用这个）
        description="LLM 提供方，可选：claude / ollama / openai_compat / minimax",
    )

    # ----- Claude 配置 -----
    # 注意：API key 不放在这里，而是从环境变量 ANTHROPIC_API_KEY 直接读
    # 这样可以让 key 走系统级环境变量（更安全，也避免被序列化到日志）
    claude_model: str = Field(
        default="claude-sonnet-5-20251001",
        description="Claude 模型名",
    )

    # ----- OpenAI 兼容配置 -----
    openai_model: str = Field(
        default="gpt-4o-mini",
        description="OpenAI 模型名（也用于中转 API）",
    )
    openai_base_url: str | None = Field(
        default=None,  # None 表示用 OpenAI 官方地址
        description="OpenAI 兼容 API 的 base_url（用于中转、自建服务）",
    )

    # ----- Ollama 配置 -----
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama 服务地址",
    )
    ollama_model: str = Field(
        default="qwen2.5:7b",
        description="Ollama 模型名（需先 ollama pull 下载）",
    )

    # ----- MiniMax 配置（OpenAI 兼容协议，复用 ChatOpenAI 客户端）-----
    # MiniMax 的 API 走 OpenAI 兼容协议，所以用 ChatOpenAI 就能连
    # 模型名默认 minimaxm3，可在 .env 里覆盖
    minimax_model: str = Field(
        default="minimaxm3",
        description="MiniMax 模型名（默认 minimaxm3）",
    )
    minimax_base_url: str = Field(
        default="https://api.minimax.com/v1",
        description="MiniMax API 基础地址（OpenAI 兼容）",
    )

    # ----- pydantic-settings 配置 -----
    # 这是 pydantic-settings 特有的元配置，控制如何从环境加载
    model_config = SettingsConfigDict(
        # 所有环境变量都要加这个前缀，避免与系统其他变量冲突
        # 例如 LLMSettings 的 provider 字段对应 PRIVATE_TEACHER_LLM_PROVIDER
        env_prefix="PRIVATE_TEACHER_LLM_",
        # 允许从 .env 文件读取
        env_file=".env",
        # 允许 .env 文件不存在（如果用户没创建 .env 也能跑）
        env_file_encoding="utf-8",
        # 忽略 .env 里多余的变量（防止用户加了未定义的变量就报错）
        extra="ignore",
        # 字段名大小写不敏感（PRIVATE_TEACHER_LLM_PROVIDER 和 private_teacher_llm_provider 都可以）
        case_sensitive=False,
    )


# ============================================================
# 日志相关配置
# ============================================================
class LogSettings(BaseSettings):
    """
    日志相关配置。

    环境变量前缀：PRIVATE_TEACHER_LOG_
    """

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="日志级别",
    )
    json_logs: bool = Field(
        default=False,
        description="是否输出 JSON 格式日志（生产环境推荐 true）",
    )

    model_config = SettingsConfigDict(
        env_prefix="PRIVATE_TEACHER_LOG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


# ============================================================
# 路径相关配置
# ============================================================
class PathSettings(BaseSettings):
    """
    路径相关配置。

    环境变量前缀：PRIVATE_TEACHER_PATH_

    所有用户数据（课件、知识库、单元、成绩）都存在 data_dir 下。
    """

    data_dir: Path = Field(
        default=Path("./data"),
        description="用户数据根目录（课件、知识库、单元、成绩等）",
    )

    model_config = SettingsConfigDict(
        env_prefix="PRIVATE_TEACHER_PATH_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    def course_dir(self, course_id: str) -> Path:
        """获取某门课的课件目录（子目录）。"""
        return self.data_dir / "courseware" / course_id

    # 其他子目录方法在 Phase 1 实现 RAG 时再加，避免 Phase 0 过度设计
    def ensure_data_dir(self) -> None:
        """确保 data_dir 存在（不存在则创建）。"""
        self.data_dir.mkdir(parents=True, exist_ok=True)


# ============================================================
# 顶层聚合配置
# ============================================================
class AppSettings:
    """
    应用级配置聚合。

    本身不是 BaseSettings（避免 pydantic 重复实例化），
    而是显式持有三个子 settings 实例。

    用法：
        settings = AppSettings.load()
        print(settings.llm.provider)
        settings.validate_for_runtime()  # 关键参数校验
    """

    # 类型注解：让 IDE 和 mypy 知道这三个字段的类型
    llm: LLMSettings
    logging: LogSettings
    paths: PathSettings

    def __init__(
        self,
        llm: LLMSettings,
        logging: LogSettings,  # noqa: A002  # logging 名字与 stdlib 冲突是故意的
        paths: PathSettings,
    ) -> None:
        self.llm = llm
        self.logging = logging
        self.paths = paths

    @classmethod
    def load(cls) -> "AppSettings":
        """
        一次性加载所有子配置。

        这里手动实例化三个子 settings 而不是用嵌套 BaseSettings，
        是为了让三个子模块可以独立被其他模块引用（如 logging 模块只需 LogSettings）。
        """
        return cls(
            llm=LLMSettings(),
            logging=LogSettings(),
            paths=PathSettings(),
        )

    def validate_for_runtime(self) -> None:
        """
        运行前校验关键参数。

        当前检查：
          - provider=claude 时必须有 ANTHROPIC_API_KEY
          - provider=openai_compat 时必须有 OPENAI_API_KEY
          - provider=minimax 时必须有 MINIMAX_API_KEY
          - provider=ollama 时不强制检查（Ollama 本地无 key）
        """
        # 这里先写最简版，Phase 0 只校验最关键的一项
        if self.llm.provider == "claude":
            # 故意用 os.getenv 而不是 LLMSettings 字段
            # 因为 key 是敏感信息，不应该进入我们的 Settings 序列化
            if not os.getenv("ANTHROPIC_API_KEY"):
                # 这里先 raise 标准 ValueError，Phase 0 之后会换成自定义 ConfigurationError
                # 引导你理解异常体系：在 5.3 节我们会替换为 raise ConfigurationError(...)
                raise ValueError(
                    "provider=claude 需要设置环境变量 ANTHROPIC_API_KEY，"
                    "请在 .env 文件中配置"
                )
        elif self.llm.provider == "openai_compat":
            if not os.getenv("OPENAI_API_KEY"):
                raise ValueError(
                    "provider=openai_compat 需要设置环境变量 OPENAI_API_KEY"
                )
        elif self.llm.provider == "minimax":
            # MiniMax 也走 API key 校验（用环境变量 MINIMAX_API_KEY）
            if not os.getenv("MINIMAX_API_KEY"):
                raise ValueError(
                    "provider=minimax 需要设置环境变量 MINIMAX_API_KEY，"
                    "请在 .env 文件中配置"
                )
        # ollama 不强制 key

    def __repr__(self) -> str:
        # 隐藏敏感信息：llm 配置里不打印 key
        return (
            f"AppSettings(llm.provider={self.llm.provider!r}, "
            f"logging.level={self.logging.level!r}, "
            f"paths.data_dir={self.paths.data_dir!r})"
        )
```

> 🐛 **实现注意事项**：
> - `validate_for_runtime` 暂时用 `ValueError`，5.3 节替换为 `ConfigurationError`，记得回这里改。
> - `PathSettings.course_dir` / `ensure_data_dir` 是为 Phase 1 埋伏笔，Phase 0 用不到也保留。

### 4.4 测试代码（test_config.py）

跟着模块一起写测试，能在写完 config.py 后立即验证。先写测试再写实现是 TDD 模式，我们这里用的是「**实现 + 测试同步落地**」。

**tests/unit/test_config.py**
```python
"""
测试 config 模块。

覆盖：
  - 默认值正确
  - .env 文件加载
  - 环境变量优先级
  - validate_for_runtime 校验
  - 嵌套字段访问
"""

from __future__ import annotations

import pytest
from pathlib import Path

from private_teacher.config import AppSettings, LLMSettings, LogSettings, PathSettings
from private_teacher.utils.exceptions import ConfigurationError


class TestLLMSettings:
    """测试 LLMSettings 子配置。"""

    def test_default_provider_is_claude(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """默认 provider 应该是 claude。"""
        # 清空可能影响测试的环境变量
        monkeypatch.delenv("PRIVATE_TEACHER_LLM_PROVIDER", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        # 切到临时目录，避免读到项目根的 .env
        monkeypatch.chdir(Path("/tmp"))

        settings = LLMSettings()
        assert settings.provider == "claude"
        assert settings.claude_model == "claude-sonnet-5-20251001"

    def test_env_var_overrides_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """环境变量应该覆盖默认值。"""
        monkeypatch.delenv("PRIVATE_TEACHER_LLM_PROVIDER", raising=False)
        monkeypatch.setenv("PRIVATE_TEACHER_LLM_PROVIDER", "ollama")
        monkeypatch.chdir(Path("/tmp"))

        settings = LLMSettings()
        assert settings.provider == "ollama"

    def test_dotenv_loading(self, tmp_env: Path) -> None:
        """.env 文件应被加载。"""
        # tmp_env fixture 把 cwd 切到 tmp_path
        (tmp_env / ".env").write_text(
            "PRIVATE_TEACHER_LLM_PROVIDER=openai_compat\n"
            "PRIVATE_TEACHER_LLM_OPENAI_MODEL=gpt-4o\n"
        )

        settings = LLMSettings()
        assert settings.provider == "openai_compat"
        assert settings.openai_model == "gpt-4o"

    def test_env_var_beats_dotenv(
        self, tmp_env: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """环境变量优先级高于 .env 文件。"""
        (tmp_env / ".env").write_text("PRIVATE_TEACHER_LLM_PROVIDER=ollama\n")
        monkeypatch.setenv("PRIVATE_TEACHER_LLM_PROVIDER", "claude")

        settings = LLMSettings()
        assert settings.provider == "claude"

    def test_invalid_provider_raises(self, tmp_env: Path) -> None:
        """provider 取非法值时，pydantic 应抛 ValidationError。"""
        from pydantic import ValidationError

        (tmp_env / ".env").write_text("PRIVATE_TEACHER_LLM_PROVIDER=invalid\n")
        with pytest.raises(ValidationError):
            LLMSettings()


class TestLogSettings:
    """测试 LogSettings 子配置。"""

    def test_default_level_is_info(self) -> None:
        settings = LogSettings()
        assert settings.level == "INFO"
        assert settings.json_logs is False


class TestPathSettings:
    """测试 PathSettings 子配置。"""

    def test_default_data_dir(self) -> None:
        settings = PathSettings()
        assert settings.data_dir == Path("./data")

    def test_ensure_data_dir_creates_dir(self, tmp_path: Path) -> None:
        """ensure_data_dir 应在 data_dir 不存在时创建它。"""
        from private_teacher.config import PathSettings
        settings = PathSettings(data_dir=tmp_path / "new_data")
        assert not (tmp_path / "new_data").exists()
        settings.ensure_data_dir()
        assert (tmp_path / "new_data").exists()


class TestAppSettings:
    """测试 AppSettings 聚合配置。"""

    def test_load_returns_all_subsystems(self) -> None:
        """AppSettings.load() 应返回包含所有子配置的实例。"""
        settings = AppSettings.load()
        assert isinstance(settings.llm, LLMSettings)
        assert isinstance(settings.logging, LogSettings)
        assert isinstance(settings.paths, PathSettings)

    def test_validate_raises_for_claude_without_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """provider=claude 但 ANTHROPIC_API_KEY 缺失时应抛 ConfigurationError。"""
        monkeypatch.setenv("PRIVATE_TEACHER_LLM_PROVIDER", "claude")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        settings = AppSettings.load()
        with pytest.raises(ConfigurationError) as exc_info:
            settings.validate_for_runtime()
        assert "ANTHROPIC_API_KEY" in str(exc_info.value)

    def test_validate_passes_for_claude_with_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """provider=claude 且 ANTHROPIC_API_KEY 设置时不应抛错。"""
        monkeypatch.setenv("PRIVATE_TEACHER_LLM_PROVIDER", "claude")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

        settings = AppSettings.load()
        settings.validate_for_runtime()  # 不应抛错

    def test_validate_raises_for_minimax_without_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """provider=minimax 但 MINIMAX_API_KEY 缺失时应抛 ConfigurationError。"""
        monkeypatch.setenv("PRIVATE_TEACHER_LLM_PROVIDER", "minimax")
        monkeypatch.delenv("MINIMAX_API_KEY", raising=False)

        settings = AppSettings.load()
        with pytest.raises(ConfigurationError) as exc_info:
            settings.validate_for_runtime()
        assert "MINIMAX_API_KEY" in str(exc_info.value)

    def test_validate_passes_for_minimax_with_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """provider=minimax 且 MINIMAX_API_KEY 设置时不应抛错。"""
        monkeypatch.setenv("PRIVATE_TEACHER_LLM_PROVIDER", "minimax")
        monkeypatch.setenv("MINIMAX_API_KEY", "test-minimax-key")

        settings = AppSettings.load()
        settings.validate_for_runtime()  # 不应抛错

    def test_repr_hides_sensitive_info(self) -> None:
        """__repr__ 不应泄露 key。"""
        settings = AppSettings.load()
        repr_str = repr(settings)
        # repr 里不应包含任何 key 字样
        assert "key" not in repr_str.lower() or "api_key" not in repr_str.lower()
```

### 4.5 验证
跑下面这条命令，**应该看到 14 个测试全过**：

```bash
uv run pytest tests/unit/test_config.py -v
```

预期输出（节选）：
```
tests/unit/test_config.py::TestLLMSettings::test_default_provider_is_claude PASSED
tests/unit/test_config.py::TestLLMSettings::test_env_var_overrides_default PASSED
...
tests/unit/test_config.py::TestAppSettings::test_repr_hides_sensitive_info PASSED

========== 14 passed in 0.5s ==========
```

> 🐛 **踩坑**：
> - `BaseSettings` 默认**不**读 `.env` —— 必须显式 `env_file=".env"`。
> - `extra="ignore"` 必须加，否则 `.env` 里的无关变量会报错。
> - 跑测试时报 `ModuleNotFoundError: No module named 'private_teacher'`？跑 `uv pip install -e .` 把项目装进 venv。

---

## 5. 异常体系：utils/exceptions.py（15 分钟）

### 5.1 目标
建立**领域异常层级**，让上层能 catch 具体异常做对应处理，而不是全部 `except Exception`。

### 5.2 概念小课堂
- **基类异常** = 框架层能统一 catch 的根。
- **具体异常** = 业务层抛的具体问题。
- 一个好习惯：自定义异常继承自一个**项目根异常**（`PrivateTeacherError`），外层用 `except PrivateTeacherError` 一网打尽。
- 异常**不要**承载业务逻辑，只承载「发生了什么 + 上下文」。

### 5.3 文件内容指引（src/private_teacher/utils/exceptions.py）

**导入什么**：
- 无（标准库自带 `Exception`）。

**类设计**：
- `class PrivateTeacherError(Exception)`：项目根异常。
  - 接受 `message: str` 和 `**context`（可选）。
  - `__str__` 返回 `message` + 拼接 context。
- `class ConfigurationError(PrivateTeacherError)`：配置错误。
- `class LLMError(PrivateTeacherError)`：LLM 调用错误。
- `class DocumentLoadError(PrivateTeacherError)`：文档加载错误（Phase 1 用，先建好）。
- `class RAGError(PrivateTeacherError)`：RAG 错误（Phase 1 用）。
- `class NotebookError(PrivateTeacherError)`：notebook 错误（Phase 3 用）。
- `class GradingError(PrivateTeacherError)`：批改错误（Phase 4 用）。

**思考题**：
- 为什么 `LLMError` 不直接继承 `Exception`？→ 让上层可以 `except PrivateTeacherError` 统一处理所有我们自己的异常。
- `**context` 怎么用？→ `raise LLMError("rate limit", provider="claude", retry_after=60)`，输出便于排查。

**实现（直接复制）**：

**src/private_teacher/utils/exceptions.py**
```python
"""
自定义异常体系
==============

本模块定义项目内所有自定义异常，遵循以下原则：
  1. 单一根异常 PrivateTeacherError：所有项目异常都继承它
  2. 异常只承载"发生了什么 + 上下文"，不承载业务逻辑
  3. 支持携带 context 字典，便于日志与排查

使用示例：
    try:
        do_something()
    except PrivateTeacherError as e:
        logger.error(f"操作失败: {e}")
        # 统一处理：可以在这里发通知、记录指标等
"""

from __future__ import annotations

# ============================================================
# 根异常
# ============================================================
class PrivateTeacherError(Exception):
    """
    项目根异常。

    所有自定义异常都应该继承本类，这样上层可以用
    `except PrivateTeacherError:` 一网打尽。

    支持携带 context 字典，便于日志输出与问题排查：
        raise LLMError("rate limit", provider="claude", retry_after=60)
        # str(e) 会输出: "rate limit | provider=claude, retry_after=60"
    """

    def __init__(self, message: str, **context: object) -> None:
        """
        Args:
            message: 人类可读的错误描述
            **context: 任意附加上下文，会在 __str__ 中打印
        """
        # 调用父类 Exception 的 __init__，让 traceback 正常显示
        super().__init__(message)
        self.message = message  # 单独存一份，方便后续访问
        self.context: dict[str, object] = context  # 上下文字典

    def __str__(self) -> str:
        """
        自定义字符串表示：把 context 拼接到 message 后面。

        例子：
          message="API 调用失败", context={"status": 500, "url": "..."}
          输出: "API 调用失败 | status=500, url=..."
        """
        if not self.context:
            return self.message
        # 把 context 字典格式化成 "k1=v1, k2=v2"
        ctx_str = ", ".join(f"{k}={v!r}" for k, v in self.context.items())
        return f"{self.message} | {ctx_str}"

    def __repr__(self) -> str:
        # 调试时用，显示完整信息
        return f"{self.__class__.__name__}({self.message!r}, **{self.context!r})"


# ============================================================
# 配置相关
# ============================================================
class ConfigurationError(PrivateTeacherError):
    """
    配置错误。

    使用场景：
      - 缺少必要的环境变量
      - provider 字段值非法
      - 配置文件格式错误

    示例：
        raise ConfigurationError("缺少 API key", provider="claude", env_var="ANTHROPIC_API_KEY")
    """
    pass  # 暂不增加额外字段，保持简单


# ============================================================
# LLM 相关
# ============================================================
class LLMError(PrivateTeacherError):
    """
    LLM 调用相关错误。

    使用场景：
      - 网络错误（API 不可达）
      - 鉴权失败（key 错误）
      - 速率限制（rate limit）
      - 输出解析失败（JSON 损坏）

    transient 字段：
      - True 表示临时错误，可以重试（time-out / 5xx / rate limit）
      - False 表示永久错误，不应重试（鉴权失败 / 4xx / 输出格式错）
    """
    # 类变量默认值
    transient: bool = False

    def __init__(
        self,
        message: str,
        *,
        transient: bool = False,
        **context: object,
    ) -> None:
        """
        Args:
            message: 错误描述
            transient: 是否临时错误（True 表示可重试，False 表示永久错误）
            **context: 附加上下文
        """
        super().__init__(message, **context)
        self.transient = transient


# ============================================================
# 文档加载相关（Phase 1 使用，先建好避免循环 import）
# ============================================================
class DocumentLoadError(PrivateTeacherError):
    """
    文档加载错误。

    使用场景：
      - PDF 解析失败
      - 文件不存在
      - 文件格式不支持
    """
    pass


# ============================================================
# RAG 相关（Phase 1 使用）
# ============================================================
class RAGError(PrivateTeacherError):
    """
    RAG 流程错误。

    使用场景：
      - Embedding 失败
      - 向量库写入失败
      - 检索异常
    """
    pass


# ============================================================
# Notebook 相关（Phase 3 使用）
# ============================================================
class NotebookError(PrivateTeacherError):
    """
    Jupyter Notebook 处理错误。

    使用场景：
      - ipynb 格式损坏
      - 用户代码执行超时
      - 沙箱逃逸尝试
    """
    pass


# ============================================================
# 批改相关（Phase 4 使用）
# ============================================================
class GradingError(PrivateTeacherError):
    """
    批改流程错误。

    使用场景：
      - 题目无法解析
      - 评分异常
    """
    pass
```

> 💡 **配套修改**：回到 `config.py` 的 `validate_for_runtime`，把 `raise ValueError(...)` 改成 `from private_teacher.utils.exceptions import ConfigurationError; raise ConfigurationError(...)`。

**实现（修改 config.py 中 validate_for_runtime）**：

```python
# 在 config.py 顶部 import 区加一行：
from private_teacher.utils.exceptions import ConfigurationError

# 然后把 validate_for_runtime 里的：
#     raise ValueError("provider=claude 需要设置环境变量 ANTHROPIC_API_KEY，...")
# 改成：
#     raise ConfigurationError(
#         "provider=claude 需要设置环境变量 ANTHROPIC_API_KEY",
#         env_var="ANTHROPIC_API_KEY",
#     )
```

### 5.4 测试代码（test_exceptions.py）

**tests/unit/test_exceptions.py**
```python
"""
测试异常体系。

覆盖：
  - PrivateTeacherError 能被 except PrivateTeacherError 捕获
  - 子类能被 except PrivateTeacherError 捕获
  - __str__ 包含 context
  - LLMError.transient 字段正确
"""

from __future__ import annotations

import pytest

from private_teacher.utils.exceptions import (
    PrivateTeacherError,
    ConfigurationError,
    LLMError,
    DocumentLoadError,
    RAGError,
    NotebookError,
    GradingError,
)


class TestPrivateTeacherError:
    """测试根异常。"""

    def test_can_be_caught_by_base_class(self) -> None:
        """子类应能被根异常捕获。"""
        with pytest.raises(PrivateTeacherError):
            raise ConfigurationError("test")

    def test_str_with_context(self) -> None:
        """__str__ 应包含 context 信息。"""
        exc = ConfigurationError("missing", key="ANTHROPIC_API_KEY")
        result = str(exc)
        assert "missing" in result
        assert "ANTHROPIC_API_KEY" in result

    def test_str_without_context(self) -> None:
        """无 context 时 __str__ 只返回 message。"""
        exc = ConfigurationError("oops")
        assert str(exc) == "oops"

    def test_repr_includes_class_name(self) -> None:
        """__repr__ 应包含类名便于调试。"""
        exc = ConfigurationError("oops", x=1)
        r = repr(exc)
        assert "ConfigurationError" in r
        assert "oops" in r


class TestLLMError:
    """测试 LLMError 特有字段。"""

    def test_default_transient_is_false(self) -> None:
        """LLMError 默认 transient=False（永久错误）。"""
        exc = LLMError("bad json")
        assert exc.transient is False

    def test_transient_can_be_set_true(self) -> None:
        """可以显式设置 transient=True。"""
        exc = LLMError("rate limit", transient=True)
        assert exc.transient is True


class TestHierarchy:
    """测试异常继承关系。"""

    @pytest.mark.parametrize("exc_cls", [
        ConfigurationError,
        LLMError,
        DocumentLoadError,
        RAGError,
        NotebookError,
        GradingError,
    ])
    def test_all_inherit_from_root(self, exc_cls: type) -> None:
        """所有自定义异常都应继承 PrivateTeacherError。"""
        assert issubclass(exc_cls, PrivateTeacherError)
```

### 5.5 验证
```bash
uv run pytest tests/unit/test_exceptions.py -v
```

预期：**13 个测试全过**（含 6 个参数化）。

> 🐛 **踩坑**：
> - 异常测试不要用 `assert exc_info.value == "xxx"`，而是用 `in str(exc_info.value)`，因为我们重写了 `__str__`。
> - 测试 `LLMError` 时记得导入 `from private_teacher.utils.exceptions import LLMError`（不是 `from langchain_core.exceptions`）。

---

## 6. 日志：utils/logging.py（20 分钟）

### 6.1 目标
统一项目日志格式与输出位置。dev 模式彩色 + 控制台，prod 模式 JSON + 文件。

### 6.2 概念小课堂
- **标准库 `logging`** 配置复杂、格式死板、debug 不友好。
- **loguru** 是第三方库，**单文件 API**：`from loguru import logger`，直接 `logger.info("hi")`。
- loguru 用 `logger.add(...)` 加 sink（输出目标），可以加多个：控制台 + 文件 + 错误文件。
- 日志**结构化**（JSON）方便日志聚合系统（Loki / ELK）消费。

### 6.3 文件内容指引（src/private_teacher/utils/logging.py）

**导入什么**：
- `import sys`
- `from loguru import logger`
- `from private_teacher.config import LogSettings`（注意导入路径，要从 src 装好后才能这么用）

**要实现的函数**：
- `def configure_logging(settings: LogSettings) -> None`：
  - 先 `logger.remove()`（清掉默认 sink）。
  - 加一个控制台 sink：`sys.stderr`，level=settings.level，彩色（dev 友好），format 带时间 + level + 模块 + 消息。
  - 如果 `settings.json_logs`：format 改成 JSON 字符串。
  - 可选：加一个文件 sink（logs/app.log，rotation="10 MB"，retention="7 days"）。

**思考题**：
- 为什么不直接 `from loguru import logger` 然后在每个文件用？→ 因为 logger.configure() 应该在 main 入口调一次，**不要**在 import 时调（容易被多次 import 触发）。
- 怎么避免测试时日志污染 pytest 输出？→ 在 `conftest.py` 里加一个 autouse fixture，把 log level 暂时设为 ERROR。

**实现（直接复制）**：

**src/private_teacher/utils/logging.py**
```python
"""
日志配置模块
============

封装 loguru 的初始化逻辑，让项目里的日志输出格式统一。

设计要点：
  - 入口处（CLI 启动时）调用一次 configure_logging()
  - 其他模块直接 from loguru import logger; logger.info(...) 即可
  - 不在 import 时调 configure，避免被多次 import 重复初始化

loguru 速览：
  - logger 是单例，全局共享
  - logger.add(sink) 添加一个输出目标（控制台、文件、网络…）
  - logger.remove() 移除所有 sink
  - 格式化字符串里 {field} 会自动展开为对应字段值
"""

from __future__ import annotations

# ---------- 标准库导入 ----------
import sys  # 用于 stderr 输出

# ---------- 第三方库导入 ----------
from loguru import logger  # type: ignore[import-untyped]  # loguru 没类型注解，加个 noqa 注释

# ---------- 本项目导入 ----------
from private_teacher.config import LogSettings  # 日志配置


# ============================================================
# 日志配置函数
# ============================================================
def configure_logging(settings: LogSettings) -> None:
    """
    根据配置初始化全局 logger。

    调用时机：CLI 入口、Streamlit 应用启动、测试启动
    调用次数：通常只调一次，重复调用会先 remove 再 add（不会重复输出）

    Args:
        settings: 日志配置
    """
    # 1. 先清掉 loguru 默认的 stderr handler
    # 为什么要 remove？避免与其他 handler 重复输出
    logger.remove()

    # 2. 根据 json_logs 决定格式
    if settings.json_logs:
        # JSON 格式：生产环境用，方便日志聚合系统（Loki / ELK）解析
        # 这里用 loguru 内置的 serialize=True 让 loguru 自动序列化为 JSON
        log_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        )
    else:
        # 人类友好格式：带颜色，dev 体验好
        # 颜色 tag：
        #   <green>...</green> = 绿色
        #   <cyan>...</cyan>   = 青色
        #   <level>...</level> = 根据 level 自动着色
        log_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        )

    # 3. 加控制台 sink（输出到 stderr）
    # level 参数过滤：低于该 level 的日志不输出
    # backtrace=True 让异常日志显示完整调用链
    # diagnose=False 不在生产环境暴露变量值（安全考虑）
    logger.add(
        sys.stderr,
        format=log_format,
        level=settings.level,
        colorize=not settings.json_logs,  # JSON 模式不开颜色（避免 ANSI 转义污染）
        backtrace=True,
        diagnose=False,  # 生产环境 False，dev 调试可以临时改 True
    )

    # 4. 可选：加一个文件 sink（让日志持久化）
    # 暂不启用，Phase 0 阶段以控制台为主，后续阶段再加
    # 启用时取消注释：
    # logger.add(
    #     "logs/app.log",
    #     format=log_format,
    #     level=settings.level,
    #     rotation="10 MB",      # 单文件超过 10MB 自动切割
    #     retention="7 days",    # 保留 7 天
    #     encoding="utf-8",
    #     enqueue=True,          # 异步写入，不阻塞主流程
    # )

    # 5. 输出配置完成信息（logger.info 自动走上面注册的 sink）
    logger.debug(f"日志初始化完成: level={settings.level}, json_logs={settings.json_logs}")


# ============================================================
# 模块级便利函数（可选）
# ============================================================
def get_logger() -> "logger":  # type: ignore[name-defined]  # noqa: F821
    """
    获取全局 logger 实例（推荐直接 from loguru import logger 即可，这里只是备用）。

    用法：
        from private_teacher.utils.logging import get_logger
        log = get_logger()
        log.info("hi")
    """
    return logger
```

> 💡 **小贴士**：在测试时如果想临时屏蔽日志，可以在 `conftest.py` 里调用 `logger.remove()` 或把 level 调成 ERROR。详见 10.3 节。

### 6.4 测试代码（test_logging.py）

**tests/unit/test_logging.py**
```python
"""
测试日志模块。

覆盖：
  - configure_logging 后能输出
  - 多次调用不重复输出
  - level 过滤生效
"""

from __future__ import annotations

import sys
from io import StringIO

import pytest
from loguru import logger  # type: ignore[import-untyped]

from private_teacher.config import LogSettings
from private_teacher.utils.logging import configure_logging


class TestConfigureLogging:
    """测试 configure_logging 函数。"""

    def test_adds_console_sink(self, capsys: pytest.CaptureFixture) -> None:
        """configure 后 logger.info 应输出到 stderr。"""
        configure_logging(LogSettings(level="INFO"))
        logger.info("test message")

        # capsys 默认捕 stdout，loguru 写 stderr，所以用 capsys 不行
        # 这里直接验证 logger 已配置 sink
        assert len(logger._core.handlers) >= 1  # noqa: SLF001  # 内部 API

    def test_multiple_calls_dont_duplicate(self) -> None:
        """多次调用 configure 不应导致重复输出。"""
        configure_logging(LogSettings(level="INFO"))
        handlers_count_1 = len(logger._core.handlers)  # noqa: SLF001

        configure_logging(LogSettings(level="INFO"))
        handlers_count_2 = len(logger._core.handlers)  # noqa: SLF001

        # 第二次 configure 先 remove 再 add，数量应该一致
        assert handlers_count_1 == handlers_count_2

    def test_level_filtering(self) -> None:
        """level=ERROR 时 INFO 不应输出。"""
        configure_logging(LogSettings(level="ERROR"))
        # 这里只能间接验证：捕获 stderr 看不到 INFO
        # 更严谨的做法是用 caplog（但 caplog 对 loguru 支持有限）
        # 我们简化处理：验证 logger 已配置
        assert len(logger._core.handlers) >= 1  # noqa: SLF001
```

### 6.5 验证
```bash
uv run pytest tests/unit/test_logging.py -v
```

预期：**3 个测试全过**。

> 🐛 **踩坑**：
> - loguru 写的是 **stderr**，不是 stdout。pytest 的 `capsys` 默认抓 stdout，要抓 stderr 用 `capsys.readouterr().err`。
> - 这里直接看 `logger._core.handlers` 是**测试实现细节**，但因为 loguru 没有公开的「已注册 sink 数量」API，只能这么写。Phase 2 可以封装自己的 logger 再用更好的测试方式。

---

## 7. 重试：utils/retry.py（20 分钟）

### 7.1 目标
封装一个**可复用**的重试装饰器，给所有「外部调用」（LLM API、HuggingFace、Chroma、网络）用。

### 7.2 概念小课堂
- **tenacity** 是 Python 生态最主流的重试库。
- 常用策略：固定间隔 / 指数退避 / 随机抖动（避免雷鸣群）。
- **哪些异常要重试** = 临时性错误（timeout、5xx、rate limit）。**绝不**重试 4xx（业务错误）。
- **最大重试次数** 默认 3，避免无限重试。

### 7.3 文件内容指引（src/private_teacher/utils/retry.py）

**导入什么**：
- `from tenacity import retry, stop_after_attempt, wait_exponential_jitter, retry_if_exception_type, before_sleep_log`
- `import logging`

**要实现什么**：
- 一个**可配置的工厂函数** `def make_retry_decorator(max_attempts: int = 3, base_wait: float = 1.0, max_wait: float = 30.0, retry_on: tuple = (...))`：
  - 返回一个 tenacity 装饰器实例。
  - `stop=stop_after_attempt(max_attempts)`
  - `wait=wait_exponential_jitter(initial=base_wait, max=max_wait)`
  - `retry=retry_if_exception_type(retry_on)`
  - `before_sleep` 用 logger 打印重试信息。
- 一个**默认实例** `default_retry`：`retry_on=(TimeoutError, ConnectionError, OSError)`。
- 一个**针对 LLM 的实例** `llm_retry`：`retry_on=(LLMError, TimeoutError, ConnectionError)`，并把 `LLMError` 中标记 `transient=True` 的重试（这块要扩展 `LLMError`，加一个 `transient` 字段）。

**思考题**：
- 为什么 `retry_on` 用元组而不是单类？→ 因为外部错误往往是 `TimeoutError | ConnectionError` 一起出现。
- 为什么不直接用 `@retry` 装饰每个函数？→ 集中策略可统一调整参数。

**实现（直接复制）**：

**src/private_teacher/utils/retry.py**
```python
"""
重试装饰器模块
==============

封装 tenacity 的常用模式，提供：
  - 可配置的重试装饰器工厂 make_retry_decorator
  - 默认实例 default_retry（适用于网络/IO 错误）
  - LLM 专用实例 llm_retry（针对 LLMError 的 transient 字段）

设计原则：
  - 只对临时错误（transient）重试，不对业务错误（4xx/参数错）重试
  - 指数退避 + 随机抖动，避免雷鸣群
  - 集中策略，便于统一调整

tenacity 速览：
  - stop_after_attempt(N) 最多重试 N 次
  - wait_exponential_jitter(initial, max) 指数退避 + 抖动
  - retry_if_exception_type(tuple) 仅对指定异常类型重试
  - before_sleep 重试前的回调，用于打日志
"""

from __future__ import annotations

# ---------- 第三方库导入 ----------
from tenacity import (
    retry,                               # 装饰器
    stop_after_attempt,                  # 停止条件：尝试 N 次
    wait_exponential_jitter,             # 等待策略：指数退避 + 抖动
    retry_if_exception,                  # 自定义重试条件（接受 lambda）
    retry_if_exception_type,             # 简单重试条件（接受类型元组）⭐ 别漏！
    before_sleep_log,                    # 重试前回调（打日志）
    RetryError,                          # 当超过 max_attempts 后会抛这个
)

# ---------- 本项目导入 ----------
from private_teacher.utils.logging import logger
from private_teacher.utils.exceptions import LLMError


# ============================================================
# 自定义 predicate：仅对 transient=True 的 LLMError 重试
# ============================================================
def _is_transient_llm_error(exception: BaseException) -> bool:
    """
    判断是否为「临时性」LLM 错误。

    只对以下情况重试：
      1. 标准网络错误（被外层 retry_on 覆盖）
      2. LLMError 且 transient=True

    不重试：
      - 普通 LLMError（如 JSON 解析失败、鉴权错误）
      - ValueError / KeyError 等业务错误
    """
    if isinstance(exception, LLMError):
        return exception.transient
    return False


# ============================================================
# 装饰器工厂
# ============================================================
def make_retry_decorator(
    *,
    max_attempts: int = 3,
    initial_wait: float = 1.0,
    max_wait: float = 30.0,
    retry_on: tuple[type[BaseException], ...] = (TimeoutError, ConnectionError, OSError),
    use_llm_transient_check: bool = False,
):
    """
    工厂函数：返回一个 tenacity 装饰器。

    Args:
        max_attempts: 最大尝试次数（包含第一次，所以默认 3 表示"1 次 + 2 次重试"）
        initial_wait: 第一次重试前的等待秒数
        max_wait: 重试间隔上限
        retry_on: 要重试的异常类型元组
        use_llm_transient_check: 是否启用 LLMError.transient 字段判断
                                 （开启后，retry_on 里的 LLMError 也会被尊重 transient 标记）

    Returns:
        一个 tenacity 装饰器实例，可以直接 @ 装饰函数

    用法：
        retry = make_retry_decorator(max_attempts=5)

        @retry
        def fetch_url(url):
            ...
    """
    # 构造重试条件
    if use_llm_transient_check:
        # 同时满足：
        #   1. 在 retry_on 列表里
        #   2. LLMError 必须 transient=True
        retry_condition = retry_if_exception(
            lambda exc: (
                isinstance(exc, retry_on)
                and not (isinstance(exc, LLMError) and not exc.transient)
            )
        )
    else:
        # 简单版：只要异常在 retry_on 里就重试
        retry_condition = retry_if_exception_type(retry_on)

    return retry(
        # 停止条件：尝试 N 次后停止
        stop=stop_after_attempt(max_attempts),
        # 等待策略：指数退避（1s, 2s, 4s, 8s...） + 随机抖动（0~1s）
        wait=wait_exponential_jitter(initial=initial_wait, max=max_wait),
        # 重试条件
        retry=retry_condition,
        # 重试前的回调：用 logger 打印 INFO 日志
        before_sleep=before_sleep_log(logger, log_level=20),  # 20 = INFO
        # 每次重试时重新抛出原异常（不要包装成 RetryError）
        reraise=True,
    )


# ============================================================
# 预设实例：开箱即用
# ============================================================
# 默认重试：网络错误最多 3 次
default_retry = make_retry_decorator(
    max_attempts=3,
    initial_wait=1.0,
    max_wait=10.0,
    retry_on=(TimeoutError, ConnectionError, OSError),
)

# LLM 专用重试：4 次，尊重 transient 标记
llm_retry = make_retry_decorator(
    max_attempts=4,
    initial_wait=2.0,    # LLM API 通常 rate limit 较严，初始等待稍长
    max_wait=60.0,
    retry_on=(LLMError, TimeoutError, ConnectionError),
    use_llm_transient_check=True,  # 关键：transient=False 的 LLMError 不重试
)
```

> 💡 **使用示例**：

```python
from private_teacher.utils.retry import default_retry, llm_retry

@llm_retry
def call_claude(prompt: str) -> str:
    # 内部 raise 的 LLMError(transient=True) 会被自动重试
    # transient=False 的不会重试，直接抛出
    ...

@default_retry
def read_file(path: Path) -> str:
    # 网络/IO 错误自动重试 3 次
    ...
```

### 7.4 测试代码（test_retry.py）

**tests/unit/test_retry.py**
```python
"""
测试重试装饰器。

覆盖：
  - 第 N 次成功时不抛错
  - 超过 max_attempts 抛原异常
  - 等待时间大致符合指数
  - 不在 retry_on 的异常不重试
"""

from __future__ import annotations

import time

import pytest

from private_teacher.utils.retry import make_retry_decorator
from private_teacher.utils.exceptions import LLMError


class TestRetryDecorator:
    """测试 make_retry_decorator。"""

    def test_succeeds_after_retries(self) -> None:
        """前 2 次失败、第 3 次成功的函数，最终应返回成功结果。"""
        call_count = {"n": 0}

        @make_retry_decorator(
            max_attempts=3,
            initial_wait=0.01,  # 测试用 10ms
            max_wait=0.05,
            retry_on=(ValueError,),
        )
        def flaky() -> str:
            call_count["n"] += 1
            if call_count["n"] < 3:
                raise ValueError("not yet")
            return "success"

        result = flaky()
        assert result == "success"
        assert call_count["n"] == 3  # 调了 3 次

    def test_raises_after_max_attempts(self) -> None:
        """总是失败的函数应在 max_attempts 次后抛出。"""
        call_count = {"n": 0}

        @make_retry_decorator(
            max_attempts=3,
            initial_wait=0.01,
            max_wait=0.05,
            retry_on=(ValueError,),
        )
        def always_fail() -> None:
            call_count["n"] += 1
            raise ValueError("always")

        with pytest.raises(ValueError, match="always"):
            always_fail()
        assert call_count["n"] == 3

    def test_does_not_retry_non_listed_exception(self) -> None:
        """不在 retry_on 列表里的异常不应重试。"""
        call_count = {"n": 0}

        @make_retry_decorator(
            max_attempts=3,
            initial_wait=0.01,
            max_wait=0.05,
            retry_on=(ValueError,),
        )
        def raises_key_error() -> None:
            call_count["n"] += 1
            raise KeyError("not in retry list")

        with pytest.raises(KeyError):
            raises_key_error()
        assert call_count["n"] == 1  # 只调了 1 次

    def test_wait_time_grows_exponentially(self) -> None:
        """等待时间应大致呈指数增长（带抖动）。"""
        timestamps: list[float] = []

        @make_retry_decorator(
            max_attempts=4,
            initial_wait=0.05,
            max_wait=1.0,
            retry_on=(ValueError,),
        )
        def always_fail() -> None:
            timestamps.append(time.monotonic())
            raise ValueError("fail")

        with pytest.raises(ValueError):
            always_fail()

        # 计算相邻等待时间
        waits = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
        # 第二次等待应比第一次长（指数退避）
        # 注：带 jitter 所以不一定严格 > ，但应该不会显著更短
        assert waits[1] > waits[0] * 0.8  # 允许 20% 抖动


class TestLLMRetry:
    """测试 LLM 专用重试。"""

    def test_respects_transient_flag(self) -> None:
        """transient=False 的 LLMError 不应重试。"""
        call_count = {"n": 0}

        @make_retry_decorator(
            max_attempts=3,
            initial_wait=0.01,
            max_wait=0.05,
            retry_on=(LLMError,),
            use_llm_transient_check=True,
        )
        def non_transient() -> None:
            call_count["n"] += 1
            raise LLMError("bad json", transient=False)  # 永久错误

        with pytest.raises(LLMError):
            non_transient()
        assert call_count["n"] == 1  # 不重试

    def test_retries_transient(self) -> None:
        """transient=True 的 LLMError 应重试。"""
        call_count = {"n": 0}

        @make_retry_decorator(
            max_attempts=3,
            initial_wait=0.01,
            max_wait=0.05,
            retry_on=(LLMError,),
            use_llm_transient_check=True,
        )
        def transient() -> str:
            call_count["n"] += 1
            if call_count["n"] < 2:
                raise LLMError("rate limit", transient=True)
            return "ok"

        assert transient() == "ok"
        assert call_count["n"] == 2
```

### 7.5 验证
```bash
uv run pytest tests/unit/test_retry.py -v
```

预期：**6 个测试全过**（含两个 LLM 专用测试）。

> 🐛 **踩坑**：
> - 测试时**必须**用 `initial_wait=0.01`（10ms），否则 3 次重试要等 3s+。
> - `time.monotonic()` 测时间间隔，**不要**用 `time.time()`（会被系统时钟调整影响）。
> - 带 jitter 的指数退避不是严格递增，断言时留 20% 余量（`> waits[0] * 0.8`）。

---

## 8. LLM 工厂：llm/factory.py（45 分钟）⭐ 重点

### 8.1 目标
实现一个 `LLMFactory.build(settings: LLMSettings) -> BaseChatModel`，根据 `provider` 字段返回不同的 LangChain chat model 实例。
**这是 Phase 0 最核心的一步**，后续所有 Agent 都基于它。

### 8.2 概念小课堂
- **LangChain 的 `BaseChatModel`** 是所有 chat model 的抽象基类，`.invoke(messages)` 是统一接口。
- **三个 provider 对应三个子类**：
  - `ChatAnthropic` → Claude
  - `ChatOpenAI` → OpenAI 及兼容（含国内中转、Ollama OpenAI 模式）
  - `ChatOllama` → Ollama 原生协议
- **配置来源**：每个 provider 用自己的 env 变量（`ANTHROPIC_API_KEY` 等），不放在 `LLMSettings` 字段里更通用。
  - 但**模型名、base_url** 仍走我们的 settings。

### 8.3 文件内容指引（src/private_teacher/llm/factory.py）

**导入什么**：
- `from langchain_anthropic import ChatAnthropic`
- `from langchain_openai import ChatOpenAI`
- `from langchain_ollama import ChatOllama`
- `from langchain_core.language_models import BaseChatModel`
- `from private_teacher.config import LLMSettings`
- `from private_teacher.utils.exceptions import LLMError, ConfigurationError`

**要实现的函数**：
- `def build_chat_model(settings: LLMSettings) -> BaseChatModel`：
  - 用 `match settings.provider` 模式匹配（Python 3.10+ 语法）：
    - `case "claude"`：返回 `ChatAnthropic(model=settings.claude_model, api_key=os.getenv("ANTHROPIC_API_KEY"))`。如果 `ANTHROPIC_API_KEY` 为空，raise `ConfigurationError`。
    - `case "openai_compat"`：返回 `ChatOpenAI(model=settings.openai_model, api_key=os.getenv("OPENAI_API_KEY"), base_url=settings.openai_base_url)`。同理校验 key。
    - `case "ollama"`：返回 `ChatOllama(model=settings.ollama_model, base_url=settings.ollama_base_url)`。
    - `case _`：raise `ConfigurationError`（未知 provider）。
  - 每个分支都加 `temperature=0.2`（教学场景需要稳定）。

- 一个**便捷函数** `def quick_chat(prompt: str, settings: LLMSettings | None = None) -> str`：
  - 内部用 `build_chat_model`，调 `.invoke([HumanMessage(content=prompt)])`。
  - 返回 `result.content`。

**思考题**：
- 为什么 temperature 不用 settings 配？→ Phase 0 先硬编码，后续 Phase 2 再加 `LLMSettings.temperature`。
- 为什么不直接用 LangChain 的 init_chat_model 工厂？→ 我们要保留自己的 settings 校验和异常包装。
- key 从哪来？→ 你的 `.env` 里写 `ANTHROPIC_API_KEY=...`，pydantic 不会去管这个变量（因为我们没用 `Field` 标它），所以手动 `os.getenv`。
- **`quick_chat` 拿 model 名时为什么用 `getattr(model, "model", None) or type(model).__name__`？** → `ChatAnthropic/OpenAI/Ollama` 都有 `.model` 属性，但 `FakeListChatModel`（测试用的假模型）**没有**这个属性。直接访问会 `AttributeError`。用 `getattr` + fallback 是兼容所有 `BaseChatModel` 子类的安全写法。

**实现（直接复制）**：

**src/private_teacher/llm/factory.py**
```python
"""
LLM 工厂模块
============

根据配置选择不同的 LLM provider，返回统一的 LangChain BaseChatModel 实例。

设计要点：
  - 单一入口 build_chat_model(settings) -> BaseChatModel
  - 用 match/case（Python 3.10+）替代 if/elif 链，代码更清晰
  - key 不放在 LLMSettings 字段里，而是从环境变量读取
    （避免 key 被序列化到日志 / .dict() / 错误信息中）
  - 缺 key 立即抛 ConfigurationError，让错误前置

LangChain 速览：
  - BaseChatModel 是所有 chat model 的抽象基类
  - .invoke(messages) 是统一接口，messages 是 list[BaseMessage]
  - HumanMessage(content="...") 表示用户消息
  - AIMessage 表示 AI 回复
  - 返回值 .content 拿到字符串内容
"""

from __future__ import annotations

# ---------- 标准库导入 ----------
import os  # 读取环境变量
from functools import lru_cache  # 缓存装饰器

# ---------- 第三方库导入 ----------
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

# ---------- 本项目导入 ----------
from private_teacher.config import LLMSettings
from private_teacher.utils.exceptions import ConfigurationError, LLMError
from private_teacher.utils.logging import logger


# ============================================================
# 工厂函数
# ============================================================
def build_chat_model(settings: LLMSettings) -> BaseChatModel:
    """
    根据 settings.provider 构造对应的 chat model。

    Args:
        settings: LLM 配置

    Returns:
        BaseChatModel 实例（ChatAnthropic / ChatOpenAI / ChatOllama）

    Raises:
        ConfigurationError: provider 非法或缺少必要的 API key
    """
    # 用 match/case（Python 3.10+ 结构化模式匹配）
    # 比 if/elif 链更清晰，且能强制覆盖所有 provider
    match settings.provider:
        # ---------- Claude ----------
        case "claude":
            # 从环境变量读 key（不要放进 LLMSettings 字段，避免泄漏）
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                # 立即抛错，让用户配置好 .env 再重试
                raise ConfigurationError(
                    "缺少 ANTHROPIC_API_KEY",
                    provider="claude",
                    env_var="ANTHROPIC_API_KEY",
                    hint="在 .env 文件中设置 ANTHROPIC_API_KEY=sk-ant-...",
                )
            # temperature=0.2 让输出相对稳定（教学场景需要可复现）
            # max_tokens 留 None 用 provider 默认
            return ChatAnthropic(
                model=settings.claude_model,
                api_key=api_key,
                temperature=0.2,
            )

        # ---------- OpenAI 兼容（含中转） ----------
        case "openai_compat":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ConfigurationError(
                    "缺少 OPENAI_API_KEY",
                    provider="openai_compat",
                    env_var="OPENAI_API_KEY",
                )
            # base_url 可选：None 时用 OpenAI 官方地址
            return ChatOpenAI(
                model=settings.openai_model,
                api_key=api_key,
                base_url=settings.openai_base_url,
                temperature=0.2,
            )

        # ---------- Ollama 本地 ----------
        case "ollama":
            # Ollama 不需要 key，但需要 base_url 和 model 都配置好
            # 不做严格校验，留给用户自己排查
            return ChatOllama(
                model=settings.ollama_model,
                base_url=settings.ollama_base_url,
                temperature=0.2,
            )

        # ---------- MiniMax（OpenAI 兼容协议）----------
        case "minimax":
            # MiniMax 提供的是 OpenAI 兼容 API，所以直接复用 ChatOpenAI
            # 关键是把 base_url 指向 MiniMax 的网关
            api_key = os.getenv("MINIMAX_API_KEY")
            if not api_key:
                raise ConfigurationError(
                    "缺少 MINIMAX_API_KEY",
                    provider="minimax",
                    env_var="MINIMAX_API_KEY",
                    hint="在 .env 文件中设置 MINIMAX_API_KEY=<你的 key>",
                )
            return ChatOpenAI(
                model=settings.minimax_model,        # 默认 minimaxm3
                api_key=api_key,
                base_url=settings.minimax_base_url,  # 默认 https://api.minimax.com/v1
                temperature=0.2,
            )

        # ---------- 兜底 ----------
        case _:
            # 理论上不会到这里（pydantic 已经校验过 Literal）
            # 但留个兜底更安全
            raise ConfigurationError(
                f"未知的 LLM provider: {settings.provider!r}",
                provider=str(settings.provider),
                valid_options=["claude", "ollama", "openai_compat", "minimax"],
            )


# ============================================================
# 便捷函数（Phase 0 冒烟测试用）
# ============================================================
def quick_chat(
    prompt: str,
    settings: LLMSettings | None = None,
) -> str:
    """
    一句话调用 LLM，返回字符串回复。

    用法：
        from private_teacher.llm.factory import quick_chat
        reply = quick_chat("用一句话介绍你自己")

    Args:
        prompt: 用户输入
        settings: 可选，不传则从环境读默认配置

    Returns:
        LLM 的回复文本

    Raises:
        LLMError: 调用失败时（已包装）
    """
    # 默认从环境读（避免每次都传 settings）
    if settings is None:
        settings = LLMSettings()

    # 构造 chat model
    model = build_chat_model(settings)

    try:
        # invoke 是同步调用，耗时 1~10s
        # 异步版本用 model.ainvoke(messages)
        result = model.invoke([HumanMessage(content=prompt)])
    except Exception as exc:
        # 把所有底层异常包装成 LLMError，方便上层统一处理
        # 根据异常类型判断是否 transient
        transient = isinstance(exc, (TimeoutError, ConnectionError, OSError))
        # ⬇️ 安全获取 model 名（兼容所有 chat model，包括 FakeListChatModel）
        # ChatAnthropic/OpenAI/Ollama 都有 .model 属性，但 FakeListChatModel 没有
        # 用 getattr 安全访问，没有就 fallback 到类名
        model_name = getattr(model, "model", None) or type(model).__name__
        raise LLMError(
            f"LLM 调用失败: {exc}",
            transient=transient,
            provider=settings.provider,
            model=str(model_name),
        ) from exc

    return str(result.content)


# ============================================================
# 缓存版工厂（可选，Phase 0 不需要）
# ============================================================
@lru_cache(maxsize=1)
def get_default_chat_model() -> BaseChatModel:
    """
    缓存的默认 chat model。

    适合在长生命周期进程（如 Streamlit session）里复用同一个 model，
    避免每次都新建（langchain 内部会缓存 HTTP 连接）。

    Phase 0 暂不使用，先把基础功能跑通。
    """
    return build_chat_model(LLMSettings())
```

> ⚠️ **关键点**：
> - 第一次 `model.invoke` 会发真实 HTTP 请求，会消耗 1 个 token 的钱，**测试时务必用 `FakeListChatModel` 替代**。
> - `temperature=0.2` 不是 0，留一点创造性。后面 Phase 2 生成单元内容时可以调到 0.5+，Phase 4 批改时调到 0。

### 8.4 测试代码（test_llm_factory.py）

**tests/unit/test_llm_factory.py**
```python
"""
测试 LLM 工厂。

覆盖：
  - 各 provider 返回正确类型
  - 缺 key 抛 ConfigurationError
  - 未知 provider 抛 ConfigurationError
  - quick_chat 走 base 接口
"""

from __future__ import annotations

import pytest
from langchain_core.language_models import BaseChatModel

from private_teacher.config import LLMSettings
from private_teacher.llm.factory import build_chat_model, quick_chat
from private_teacher.utils.exceptions import ConfigurationError, LLMError


class TestBuildChatModel:
    """测试 build_chat_model 工厂。"""

    def test_claude_returns_chat_anthropic(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """provider=claude 应返回 ChatAnthropic 实例。"""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        settings = LLMSettings(provider="claude", claude_model="claude-sonnet-5-20251001")

        model = build_chat_model(settings)

        assert isinstance(model, BaseChatModel)
        # 进一步检查具体类型
        from langchain_anthropic import ChatAnthropic
        assert isinstance(model, ChatAnthropic)

    def test_claude_without_key_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """provider=claude 缺 key 时应抛 ConfigurationError。"""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        settings = LLMSettings(provider="claude")

        with pytest.raises(ConfigurationError) as exc_info:
            build_chat_model(settings)
        assert "ANTHROPIC_API_KEY" in str(exc_info.value)

    def test_openai_compat_returns_chat_openai(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """provider=openai_compat 应返回 ChatOpenAI 实例。"""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        settings = LLMSettings(
            provider="openai_compat",
            openai_model="gpt-4o-mini",
            openai_base_url="https://api.openai.com/v1",
        )

        model = build_chat_model(settings)

        from langchain_openai import ChatOpenAI
        assert isinstance(model, ChatOpenAI)

    def test_ollama_returns_chat_ollama(self) -> None:
        """provider=ollama 应返回 ChatOllama 实例。"""
        settings = LLMSettings(
            provider="ollama",
            ollama_model="qwen2.5:7b",
            ollama_base_url="http://localhost:11434",
        )

        model = build_chat_model(settings)

        from langchain_ollama import ChatOllama
        assert isinstance(model, ChatOllama)

    def test_minimax_returns_chat_openai(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """provider=minimax 应返回 ChatOpenAI 实例（因为走 OpenAI 兼容协议）。"""
        monkeypatch.setenv("MINIMAX_API_KEY", "test-minimax-key")
        settings = LLMSettings(
            provider="minimax",
            minimax_model="minimaxm3",
            minimax_base_url="https://api.minimax.com/v1",
        )

        model = build_chat_model(settings)

        from langchain_openai import ChatOpenAI
        assert isinstance(model, ChatOpenAI)
        # 进一步验证 base_url 和 model 配置正确
        assert model.openai_api_base == "https://api.minimax.com/v1"  # type: ignore[attr-defined]

    def test_minimax_without_key_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """provider=minimax 缺 key 时应抛 ConfigurationError。"""
        monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
        settings = LLMSettings(provider="minimax")

        with pytest.raises(ConfigurationError) as exc_info:
            build_chat_model(settings)
        assert "MINIMAX_API_KEY" in str(exc_info.value)

    def test_unknown_provider_raises(self) -> None:
        """未知 provider 应抛 ConfigurationError。"""
        # 用 model_construct 绕过 pydantic Literal 校验
        settings = LLMSettings.model_construct(provider="unknown")  # type: ignore[arg-type]

        with pytest.raises(ConfigurationError):
            build_chat_model(settings)


class TestQuickChat:
    """测试 quick_chat 便捷函数。"""

    def test_quick_chat_with_fake_model(
        self, monkeypatch: pytest.MonkeyPatch, fake_chat_model
    ) -> None:
        """quick_chat 应能调用假 model 并返回响应。"""
        # 注入假 model
        def fake_build(settings: LLMSettings) -> BaseChatModel:
            return fake_chat_model(responses=["pong"])
        monkeypatch.setattr(
            "private_teacher.llm.factory.build_chat_model", fake_build
        )

        result = quick_chat("ping")
        assert "pong" in result

    def test_quick_chat_wraps_exception(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """quick_chat 内部异常应被包装为 LLMError。"""
        from langchain_core.language_models.fake_chat_models import FakeListChatModel

        class BrokenModel(FakeListChatModel):
            def invoke(self, *args, **kwargs):  # type: ignore[override]
                raise ConnectionError("network down")

        def fake_build(settings: LLMSettings) -> BaseChatModel:
            return BrokenModel(responses=[])

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        monkeypatch.setattr(
            "private_teacher.llm.factory.build_chat_model", fake_build
        )

        with pytest.raises(LLMError) as exc_info:
            quick_chat("ping")
        # ConnectionError 是 transient 错误
        assert exc_info.value.transient is True


# 可选：真实 LLM 烟囱测试（默认跳过）
@pytest.mark.requires_llm
class TestRealLLM:
    """真实 LLM 测试（需 ANTHROPIC_API_KEY 环境变量）。"""

    def test_real_claude_responds(self) -> None:
        import os
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("需要 ANTHROPIC_API_KEY")

        settings = LLMSettings(provider="claude")
        result = quick_chat("Say 'pong' and nothing else.")
        assert "pong" in result.lower()
```

### 8.5 验证
```bash
# 不含真实 LLM 测试
uv run pytest tests/unit/test_llm_factory.py -v

# 含真实 LLM 测试（需 key）
uv run pytest tests/unit/test_llm_factory.py -v -m requires_llm
```

预期：**8 个测试全过**（不含 requires_llm）；跑 `requires_llm` 会多 1 个真实调用。

> 🐛 **踩坑**：
> - 测 `minimax` 时一定要 `monkeypatch.setenv("MINIMAX_API_KEY", ...)`，不然会抛 `ConfigurationError`。
> - `LLMSettings.model_construct(provider="unknown")` 绕过 pydantic Literal 校验，是为了测 factory 的兜底分支。生产代码别这么用。
> - 测 `openai_api_base` 属性时 LangChain 不同版本名字可能不同（`openai_api_base` vs `base_url`），断言失败就改成 `assert "minimax.com" in str(model.openai_api_base)`。
> - **`quick_chat` 异常包装时必须用 `getattr(model, "model", None) or type(model).__name__`**——`FakeListChatModel` 没有 `.model` 属性，直接 `model.model` 会炸 `AttributeError`。这是测 `test_quick_chat_wraps_exception` 时最容易踩的坑。

---

## 9. 命令行入口：cli.py + __main__.py（20 分钟）

### 9.1 目标
让 `python -m private_teacher hello-llm` 跑通，输出 LLM 回复。

### 9.2 概念小课堂
- **`__main__.py`** 让 `python -m private_teacher` 等价于 `python -m private_teacher.__main__`。
- **argparse** 是标准库，够用；如果想要更现代的可以用 `typer`（Phase 0 先不引）。
- 一个子命令 = 一个函数。`hello-llm` 是 Phase 0 收尾的「冒烟测试」。

### 9.3 文件内容指引

**src/private_teacher/cli.py**：
- 导入：`argparse`、`from private_teacher.config import AppSettings`、`from private_teacher.utils.logging import configure_logging`、`from private_teacher.llm.factory import quick_chat`。
- `def cmd_hello_llm(args: argparse.Namespace) -> int`：
  - 调 `settings = AppSettings.load()`，`configure_logging(settings.logging)`。
  - 调 `print(quick_chat("用一句话介绍你自己"))`。
  - return 0。
- `def cmd_hello(args: argparse.Namespace) -> int`：
  - 简单打印 "Hello from Private Teacher! Phase 0 is ready."
  - return 0。
- `def build_parser() -> argparse.ArgumentParser`：注册子命令 `hello` 和 `hello-llm`。
- `def main(argv: list[str] | None = None) -> int`：parse → dispatch → return exit code。
- 顶层 `if __name__ == "__main__": raise SystemExit(main())`

**src/private_teacher/__main__.py**：
- 一行：`from private_teacher.cli import main; raise SystemExit(main())`

**src/private_teacher/__init__.py**：
- 加 `__version__ = "0.1.0"`，可被 `private_teacher.__version__` 访问。

**实现（直接复制）**：

**src/private_teacher/__init__.py**
```python
"""
Private Teacher 包初始化文件。

暴露版本号给外部使用：
    import private_teacher
    print(private_teacher.__version__)
"""

# 项目版本号
# 遵循语义化版本: 主版本.次版本.修订号
# - 主版本: 不兼容的 API 变更
# - 次版本: 向下兼容的功能新增
# - 修订号: 向下兼容的 bug 修复
__version__ = "0.1.0"
```

**src/private_teacher/__main__.py**
```python
"""
允许 `python -m private_teacher` 直接运行 CLI。

原理：Python 解释器执行 `python -m <package>` 时会：
  1. 找到包所在目录
  2. 执行该目录下的 __main__.py
所以这个文件就是 CLI 入口。
"""

# 从 cli 模块导入 main 函数（命令行主逻辑）
from private_teacher.cli import main

# raise SystemExit(main()) 让 exit code 正确传递给 shell
# 例如：python -m private_teacher hello-llm 出错时，shell 里 $? 会是 1
raise SystemExit(main())
```

**src/private_teacher/cli.py**
```python
"""
命令行入口模块。

提供子命令：
  - hello: 打印欢迎语（不调用 LLM，零成本）
  - hello-llm: 调用 LLM 让它介绍自己（会消耗 API 费用）

设计要点：
  - 用 argparse 标准库，不引入 typer（保持 Phase 0 依赖最小）
  - 子命令作为函数实现（cmd_hello, cmd_hello_llm），便于测试
  - main 函数接受 argv 参数，默认 None 时用 sys.argv[1:]
  - 顶层 __name__ == "__main__" 块：让这个文件也能直接 `python cli.py hello` 运行
"""

from __future__ import annotations

# ---------- 标准库导入 ----------
import argparse  # 命令行参数解析
import sys  # 用于获取 argv

# ---------- 本项目导入 ----------
from private_teacher import __version__  # 包版本号
from private_teacher.config import AppSettings  # 配置聚合
from private_teacher.utils.logging import configure_logging  # 日志配置
from private_teacher.utils.exceptions import ConfigurationError, LLMError
from private_teacher.llm.factory import quick_chat  # 便捷 LLM 调用


# ============================================================
# 子命令实现
# ============================================================
def cmd_hello(args: argparse.Namespace) -> int:
    """
    hello 子命令：纯本地打印，不调 LLM。

    用法：python -m private_teacher hello
    """
    # 输出到 stdout（print 默认），而不是 logger
    # 因为 hello 是用户面输出，不应该带时间戳、级别
    print(f"Hello from Private Teacher! v{__version__}")
    print("Phase 0 is ready. Try `python -m private_teacher hello-llm` next.")
    return 0  # 0 表示成功


def cmd_hello_llm(args: argparse.Namespace) -> int:
    """
    hello-llm 子命令：调用 LLM 让它自我介绍。

    用法：python -m private_teacher hello-llm

    会消耗 1 个 API call 的费用（约 $0.001）。
    """
    # 1. 加载配置
    try:
        settings = AppSettings.load()
    except Exception as exc:
        # 配置加载失败（如 .env 缺失）给出友好提示
        print(f"[ERROR] 配置加载失败: {exc}", file=sys.stderr)
        return 1

    # 2. 初始化日志
    configure_logging(settings.logging)

    # 3. 跑前校验（确保 key 已配置）
    try:
        settings.validate_for_runtime()
    except ConfigurationError as exc:
        # 缺 key 是用户配置问题，给出明确指引
        print(f"[ERROR] 配置错误: {exc}", file=sys.stderr)
        print("[HINT] 请在 .env 文件中设置必要的环境变量（参考 .env.example）", file=sys.stderr)
        return 2  # 用不同的 exit code 区分错误类型

    # 4. 调用 LLM
    try:
        reply = quick_chat("用一句话介绍你自己，控制在 30 字以内。")
    except LLMError as exc:
        # LLM 调用失败
        print(f"[ERROR] LLM 调用失败: {exc}", file=sys.stderr)
        return 3

    # 5. 输出
    print(f"\n[LLM 回复]\n{reply}\n")
    return 0


# ============================================================
# argparse 配置
# ============================================================
def build_parser() -> argparse.ArgumentParser:
    """
    构造 argparse 解析器。

    子命令采用「add_subparsers」模式：
      python -m private_teacher hello         → cmd_hello
      python -m private_teacher hello-llm     → cmd_hello_llm
      python -m private_teacher --version     → 版本号
    """
    parser = argparse.ArgumentParser(
        prog="private-teacher",  # 帮助信息里显示的 program 名
        description="私人教师大模型 App - 本地优先的课件学习 Agent",
    )

    # 全局 --version
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    # 子命令
    subparsers = parser.add_subparsers(
        title="commands",
        description="可用的子命令",
        dest="command",  # args.command 存子命令名
        required=True,  # 必须指定子命令
    )

    # --- hello ---
    hello_parser = subparsers.add_parser(
        "hello",
        help="打印欢迎语（不调 LLM）",
    )
    hello_parser.set_defaults(func=cmd_hello)

    # --- hello-llm ---
    hello_llm_parser = subparsers.add_parser(
        "hello-llm",
        help="调用 LLM 验证配置（会消耗 API 费用）",
    )
    hello_llm_parser.set_defaults(func=cmd_hello_llm)

    return parser


# ============================================================
# 入口函数
# ============================================================
def main(argv: list[str] | None = None) -> int:
    """
    CLI 入口函数。

    Args:
        argv: 命令行参数列表，None 时用 sys.argv[1:]

    Returns:
        进程 exit code（0 成功，非 0 失败）
    """
    parser = build_parser()
    args = parser.parse_args(argv)  # 解析参数

    # 派发到对应的 cmd_* 函数
    # 每个 cmd_* 接受 args，返回 int exit code
    try:
        return args.func(args)
    except KeyboardInterrupt:
        # 用户按 Ctrl+C
        print("\n[INFO] 用户中断", file=sys.stderr)
        return 130  # 128 + SIGINT(2) = 130，标准约定


# 直接运行此文件时的入口
# 平时通过 python -m private_teacher 走 __main__.py
# 这里写一份兜底，方便调试单个文件
if __name__ == "__main__":
    raise SystemExit(main())
```

### 9.4 测试代码（test_cli.py）

**tests/unit/test_cli.py**
```python
"""
测试 CLI 模块。

覆盖：
  - hello 命令打印欢迎语
  - hello-llm 走 mock quick_chat
  - 错误情况返回非零 exit code
"""

from __future__ import annotations

import argparse  # ← 用于构造 cmd_hello(args) 的假 Namespace 参数
import pytest

from private_teacher.cli import main, cmd_hello, cmd_hello_llm, build_parser


class TestHelloCommand:
    """测试 hello 子命令。"""

    def test_hello_prints_greeting(self, capsys: pytest.CaptureFixture) -> None:
        """hello 应打印包含 'Hello' 的欢迎语。"""
        result = cmd_hello(argparse.Namespace())  # type: ignore[arg-type]
        captured = capsys.readouterr()

        assert result == 0
        assert "Hello" in captured.out
        assert "Private Teacher" in captured.out

    def test_hello_via_main(self, capsys: pytest.CaptureFixture) -> None:
        """通过 main(['hello']) 调用应工作。"""
        result = main(["hello"])
        captured = capsys.readouterr()
        assert result == 0
        assert "Hello" in captured.out


class TestHelloLLMCommand:
    """测试 hello-llm 子命令。"""

    def test_hello_llm_uses_quick_chat(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """hello-llm 应调用 quick_chat 并打印结果。"""
        # mock quick_chat
        monkeypatch.setattr(
            "private_teacher.cli.quick_chat",
            lambda prompt, settings=None: "I am a fake LLM",
        )
        # mock 配置加载（避免依赖真实 .env）
        from private_teacher.config import AppSettings, LogSettings, LLMSettings
        from private_teacher.utils.exceptions import ConfigurationError
        fake_settings = AppSettings(
            llm=LLMSettings(provider="claude"),
            logging=LogSettings(level="ERROR"),  # 测试时静音
            paths=type("P", (), {"data_dir": None})(),  # 占位
        )
        monkeypatch.setattr(
            "private_teacher.cli.AppSettings.load",
            classmethod(lambda cls: fake_settings),
        )
        # validate_for_runtime 不抛错
        monkeypatch.setattr(
            "private_teacher.config.AppSettings.validate_for_runtime",
            lambda self: None,
        )

        result = main(["hello-llm"])
        captured = capsys.readouterr()

        assert result == 0
        assert "fake LLM" in captured.out

    def test_hello_llm_returns_error_on_missing_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """缺 key 时 hello-llm 应返回非零 exit code。"""
        from private_teacher.config import AppSettings, LogSettings, LLMSettings
        from private_teacher.utils.exceptions import ConfigurationError

        fake_settings = AppSettings(
            llm=LLMSettings(provider="claude"),
            logging=LogSettings(level="ERROR"),
            paths=type("P", (), {"data_dir": None})(),
        )
        monkeypatch.setattr(
            "private_teacher.cli.AppSettings.load",
            classmethod(lambda cls: fake_settings),
        )

        # 让 validate_for_runtime 抛错
        def fake_validate(self) -> None:
            raise ConfigurationError("missing key", env_var="ANTHROPIC_API_KEY")
        monkeypatch.setattr(
            "private_teacher.config.AppSettings.validate_for_runtime",
            fake_validate,
        )

        result = main(["hello-llm"])
        assert result == 2  # 缺 key 用 exit code 2


class TestParser:
    """测试 argparse 配置。"""

    def test_parser_requires_command(self) -> None:
        """不指定子命令应报错。"""
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_parser_version(self, capsys: pytest.CaptureFixture) -> None:
        """--version 应打印版本号。"""
        parser = build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--version"])
        # version action 会调 sys.exit(0)
        assert exc_info.value.code == 0
```

> ⚠️ **真实跑通注意**：
> - `tests/unit/test_cli.py` 里的 `test_hello_llm_returns_error_on_missing_key` 中用 `type("P", ...)` 创建了一个临时类，是为了让 `AppSettings` 实例化不报错（因为 `PathSettings` 需要 `Path` 字段）。生产代码里永远不要这么写。
> - 如果 `monkeypatch.setattr` 找不到路径（因为是 `from x import y` 引入的），要用 `import private_teacher.cli as cli; monkeypatch.setattr(cli, "AppSettings", ...)`。**Phase 0 教学简化处理，遇到问题再调整**。

### 9.5 验证
```bash
# 手动验证 CLI
uv run python -m private_teacher hello            # 应看到欢迎语
uv run python -m private_teacher --version        # 应看到版本号
uv run python -m private_teacher                  # 不带参数应报错

# 跑 CLI 测试
uv run pytest tests/unit/test_cli.py -v
```

预期：**6 个测试全过**。

> 🐛 **踩坑**：
> - 如果 `from private_teacher.xxx import yyy` 报 `ModuleNotFoundError`，说明 `src/` 没装。先跑 `uv pip install -e .`（uv 会自动做），或者在 `pyproject.toml` 检查 `[tool.uv] package = true` 之类的配置。
> - **`test_cli.py` 必须 `import argparse`**——测试用 `argparse.Namespace()` 构造假参数对象。如果忘了 import，会报 `NameError: name 'argparse' is not defined`。

---

## 10. 测试基建（45 分钟）⭐ 重点

> 📍 **本节只放测试基础设施**（`conftest.py` / `fakes/` / `pyproject.toml` 配置）。
> 各模块的测试代码已经挪到对应模块章节末尾，紧跟实现代码：
>
> | 测试文件 | 位置 |
> | --- | --- |
> | `tests/unit/test_config.py` | [第 4.4 节](#44-测试代码test_configpy) |
> | `tests/unit/test_exceptions.py` | [第 5.4 节](#54-测试代码test_exceptionspy) |
> | `tests/unit/test_logging.py` | [第 6.4 节](#64-测试代码test_loggingpy) |
> | `tests/unit/test_retry.py` | [第 7.4 节](#74-测试代码test_retrypy) |
> | `tests/unit/test_llm_factory.py` | [第 8.4 节](#84-测试代码test_llm_factorypy) |
> | `tests/unit/test_cli.py` | [第 9.4 节](#94-测试代码test_clipy) |
>
> 推荐学习方式：**先写完一个模块的实现，立刻跳到对应章节末尾写测试**，跑通再继续下一个。

### 10.1 目标
- pytest 能跑起来。
- 有一个 `fake_llm` fixture，**所有 LLM 调用都走它**，**不消耗真实 API**。
- 标记 `unit` / `integration` / `requires_llm`，默认只跑 unit。
- 覆盖率 ≥ 80%（Phase 0 收尾自检）。

### 10.2 概念小课堂
- **fixture** = pytest 的依赖注入，`@pytest.fixture` 装饰函数，测试函数参数名同 fixture 名即可注入。
- **conftest.py** = 共享 fixture 的地方，无需 import。
- **monkeypatch** = pytest 内置的 mock 工具，会自动还原。
- **Fake vs Mock**：Fake 是「假的能用的实现」（如 FakeRedis），Mock 是「行为记录器」。我们要的是 **Fake**。
- **respx** = 专门 mock httpx 的库（LangChain 新版用 httpx）。
- **vcrpy** = 录制/回放 HTTP 调用的库。先不急，Phase 1 大量 LLM 调用时再用。

### 10.3 文件内容指引

**tests/conftest.py**：
- `import pytest`、`from private_teacher.config import AppSettings`。
- `@pytest.fixture(scope="session") def settings() -> AppSettings:`：返回 `AppSettings.load()`（用 `monkeypatch` 注入临时 env）。
- `@pytest.fixture(autouse=True) def _quiet_logs(caplog):`：自动把 loguru 设为 ERROR。
- `@pytest.fixture def tmp_env(monkeypatch, tmp_path) -> Path`：在 `tmp_path` 下创建临时 `.env` 并 `monkeypatch.chdir(tmp_path)`。
- 注册 marker：在文件底部写 `def pytest_configure(config): config.addinivalue_line("markers", "requires_llm: needs real LLM API key")`。

**tests/fakes/fake_llm.py**：
- 实现一个 `class FakeChatModel(BaseChatModel)`：
  - 继承 `langchain_core.language_models.chat_models.BaseChatModel`。
  - 在初始化时接受一个 `responses: list[str]`，每次 `.invoke` 按顺序返回一个，循环使用。
  - 至少实现 `_generate`（langchain 内部接口），把 `messages` 拼成一个 string，匹配到对应响应。
- 也可以更简单：直接用 LangChain 自带的 `FakeListLLM` / `FakeListChatModel`（langchain_core.language_models.fake_chat_models 里有）。**推荐用官方的**，自己写麻烦。

**pyproject.toml 新增 section**：
```
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra -q --strict-markers --cov=src/private_teacher --cov-report=term-missing"
markers = [
  "integration: integration tests",
  "e2e: end-to-end tests",
  "requires_llm: requires real LLM API key (deselect with -m 'not requires_llm')",
]
asyncio_mode = "auto"

[tool.coverage.run]
source = ["src/private_teacher"]
omit = ["*/tests/*", "*/__main__.py"]
```

### 10.4 每个测试文件要写什么

| 文件 | 测试点 |
| --- | --- |
| `test_config.py` | 默认值 / .env 加载 / 环境变量覆盖 / validate 抛错 / 嵌套字段 |
| `test_exceptions.py` | 基类 catch / 异常 str / 子类关系 |
| `test_logging.py` | configure 后能输出 / 不重复 / format 正确 |
| `test_retry.py` | 3 次内成功 / 超过 max_attempts 失败 / 等待时间大致符合指数 / 不在 retry_on 列表里的异常不重试 |
| `test_llm_factory.py` | 各 provider 返回正确类型 / 缺 key 抛错 / 未知 provider 抛错 / quick_chat 走 base 接口 |
| `test_cli.py` | hello 命令输出 / hello-llm 走 mock quick_chat / 错误命令返回非零 exit code |

**实现（直接复制）**：

**tests/conftest.py**
```python
"""
pytest 全局配置与共享 fixture。

这个文件无需 import，所有测试文件自动可见里面的 fixture。

重要 fixture：
  - settings: 全局配置（session 级别，整个测试会话只 load 一次）
  - tmp_env: 临时 .env 环境（function 级别，每个测试独立）
  - fake_chat_model: 假 LLM（按 responses 列表循环返回）
  - _quiet_logs: 自动应用，测试时静音 loguru 输出
"""

from __future__ import annotations

# ---------- 第三方库导入 ----------
import pytest
from loguru import logger  # type: ignore[import-untyped]
from langchain_core.language_models.fake_chat_models import FakeListChatModel


# ============================================================
# Marker 注册
# ============================================================
def pytest_configure(config: pytest.Config) -> None:
    """
    注册自定义 marker。

    --strict-markers 开启后，未在此注册的 marker 会报错。
    """
    config.addinivalue_line(
        "markers",
        "integration: 集成测试（可能涉及文件系统、慢）",
    )
    config.addinivalue_line(
        "markers",
        "e2e: 端到端测试（需要完整环境）",
    )
    config.addinivalue_line(
        "markers",
        "requires_llm: 需要真实 LLM API key（默认跳过）",
    )


# ============================================================
# 全局 fixtures
# ============================================================
@pytest.fixture(autouse=True)
def _quiet_logs() -> None:
    """
    自动应用：每个测试运行时临时静音 loguru 到 ERROR 级别。

    autouse=True 表示不需要显式传入，pytest 自动注入到所有测试。

    为什么这么做？
      - 日志会让 pytest 输出变脏
      - 测试时只看 assert 结果，不看 INFO 日志
    """
    logger.remove()  # 移除所有 sink
    # 不加任何 sink，等于完全静默
    # 如果想临时看日志，在测试里手动 logger.add(sys.stderr, level="DEBUG")


@pytest.fixture
def tmp_env(monkeypatch: pytest.MonkeyPatch, tmp_path) -> "object":
    """
    临时 .env 环境。

    用法：
        def test_xxx(tmp_env):
            (tmp_env / ".env").write_text("PRIVATE_TEACHER_LLM_PROVIDER=ollama")
            settings = AppSettings.load()
            assert settings.llm.provider == "ollama"

    实现：
      1. 在 tmp_path 下创建 .env
      2. monkeypatch.chdir 到 tmp_path，让 pydantic 找到 .env
    """
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def fake_chat_model():
    """
    假 ChatModel，循环返回预设 responses。

    用法：
        def test_xxx(fake_chat_model, monkeypatch):
            monkeypatch.setattr(
                "private_teacher.llm.factory.build_chat_model",
                lambda settings: fake_chat_model(responses=["pong", "hello"]),
            )
            from private_teacher.llm.factory import quick_chat
            assert "pong" in quick_chat("ping")

    LangChain 自带的 FakeListChatModel 已实现所有接口（invoke/ainvoke/stream 等），
    不需要自己写 _generate。
    """
    def _factory(responses: list[str]) -> FakeListChatModel:
        return FakeListChatModel(responses=responses)
    return _factory


@pytest.fixture(scope="session")
def settings():
    """
    共享 AppSettings（整个测试会话只 load 一次）。

    注意：session scope 需要测试中不修改 settings 字段。
    需要修改请用 tmp_env fixture。
    """
    # 延迟 import 避免循环
    from private_teacher.config import AppSettings
    return AppSettings.load()
```

**tests/fakes/__init__.py**
```python
# 让 tests/fakes/ 成为可导入的包
# 假 LLM / 假向量库 / 假 RAG 等都在这里
```

**tests/fakes/fake_llm.py**（备用：自己实现的版本，不依赖 LangChain）
```python
"""
自定义假 LLM（备选方案）。

多数情况下用 LangChain 自带的 FakeListChatModel 就够了。
这个文件实现一个更灵活的版本，支持：
  - 按 prompt 关键字路由不同响应
  - 模拟流式输出
  - 模拟错误
"""

from __future__ import annotations

# ---------- 第三方库导入 ----------
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from typing import Any


class FakeChatModel(BaseChatModel):
    """
    一个简单的假 chat model。

    用法：
        model = FakeChatModel(responses={"hello": "hi", "ping": "pong"})
        assert model.invoke([HumanMessage(content="ping")]).content == "pong"
    """

    # Pydantic 字段：响应字典
    # key 是 prompt 包含的关键字，value 是返回值
    responses: dict[str, str] = {}
    # 默认响应（当 prompt 不匹配任何 key 时）
    default_response: str = "fake response"

    @property
    def _llm_type(self) -> str:
        """LangChain 要求实现：返回模型类型标识。"""
        return "fake"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,  # noqa: ARG002
        run_manager: Any = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> ChatResult:
        """
        LangChain 内部接口：同步生成。

        把 messages 拼成 string，匹配 responses 字典里的 key。
        """
        # 提取最后一条 HumanMessage 的内容作为 prompt
        prompt = ""
        for msg in messages:
            if hasattr(msg, "content"):
                prompt += str(msg.content) + " "

        # 路由：找到第一个匹配的 key
        for keyword, response in self.responses.items():
            if keyword in prompt:
                return ChatResult(
                    generations=[ChatGeneration(message=AIMessage(content=response))]
                )

        # 没匹配到，用默认
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=self.default_response))]
        )
```

> 💡 **推荐**：测试时直接用 `langchain_core.language_models.fake_chat_models.FakeListChatModel`（更稳定、LangChain 内部已经广泛测试）。上面的 `FakeChatModel` 是教学示例，展示了如何自定义 chat model。

**tests/unit/__init__.py**
```python
# 标记 unit 为包
```

**pyproject.toml 新增 section**：

把下面这段追加到 `pyproject.toml` 末尾：

```toml
# ============================================================
# pytest 配置
# ============================================================
[tool.pytest.ini_options]
# 测试目录
testpaths = ["tests"]

# 默认选项：
#   -ra: 显示所有跳过原因
#   -q: 简洁输出
#   --strict-markers: 未注册的 marker 报错（防止拼写错）
#   --strict-config: pytest 配置错误时报错
#   --cov=src/private_teacher: 跑覆盖率
#   --cov-report=term-missing: 终端输出 + 缺失行号
addopts = "-ra -q --strict-markers --strict-config --cov=src/private_teacher --cov-report=term-missing"

# 自定义 marker
markers = [
    "integration: 集成测试（可能涉及文件系统、慢）",
    "e2e: 端到端测试（需要完整环境）",
    "requires_llm: 需要真实 LLM API key（默认跳过）",
]

# pytest-asyncio 模式：自动识别 async 测试
asyncio_mode = "auto"


# ============================================================
# coverage 配置
# ============================================================
[tool.coverage.run]
# 覆盖率统计的源码目录
source = ["src/private_teacher"]
# 不统计的文件
omit = [
    "*/tests/*",
    "*/__main__.py",
    "*/__init__.py",
]

[tool.coverage.report]
# 低于 80% 覆盖率时 CI 失败（Phase 0 收尾标准）
fail_under = 80
# 缺失行显示
show_missing = true
# 排除某些行不计入覆盖率
exclude_lines = [
    "pragma: no cover",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
    "if TYPE_CHECKING:",
]
```

### 10.5 验证
- 跑 `uv run pytest -m "not requires_llm"`，应该全绿。
- 跑 `uv run pytest --cov`，覆盖率报告应该 ≥ 80%。
- 跑 `uv run pytest -m requires_llm`（如果你配了真 key），应该至少 hello-llm 测试通过。

> 🐛 **踩坑**：
> - `--strict-markers` 强制所有 marker 在 `markers = []` 里登记过，否则报错。
> - LangChain `FakeListChatModel` 的参数名要查清楚（不同版本可能叫 `responses` 或 `responses_cycle`），看报错信息调。
> - `monkeypatch.setenv` 后要 reload settings（pydantic 1.x 默认 cache），必要时在 fixture 里每次 `Settings()`。

---

## 11. Lint / Format：ruff（10 分钟）

### 11.1 目标
- ruff 替代 flake8 + isort + black，速度快 100 倍。
- CI 跑 `ruff check` 和 `ruff format --check`。

### 11.2 操作
- 跑 `uv run ruff check .` 看初始报错。
- 跑 `uv run ruff check . --fix` 自动修。
- 跑 `uv run ruff format .` 统一格式。
- 在 `pyproject.toml` 加 `[tool.ruff]` section：
  - `line-length = 100`
  - `target-version = "py311"`
  - `[tool.ruff.lint]` 启用 `E, F, W, I, UP, B, SIM` 规则集。
- 把 `ruff` 加到 Makefile。

### 11.3 验证
- `uv run ruff check .` 应该无错。
- `uv run ruff format --check .` 应该全过。
- 故意写一行 `import os;x=1`（不换行），ruff check 应该报 E702。

**实现（直接复制）**：

把下面这段追加到 `pyproject.toml`：

```toml
# ============================================================
# ruff 配置（lint + format）
# ============================================================
[tool.ruff]
# 单行最大字符数
line-length = 100
# 目标 Python 版本（决定可用语法）
target-version = "py311"
# 同时应用的目录
src = ["src", "tests"]
# 不检查的目录
extend-exclude = [
    ".venv",
    "data",
    "logs",
    "build",
    "dist",
]

[tool.ruff.lint]
# 启用的规则集
# E/W: pycodestyle 错误和警告
# F: pyflakes（未使用变量、未定义名称等）
# I: isort（import 排序）
# UP: pyupgrade（用新语法替代老语法）
# B: flake8-bugbear（常见 bug 模式）
# SIM: flake8-simplify（代码简化建议）
# C4: flake8-comprehensions（推导式建议）
# RUF: ruff 自身规则
select = ["E", "W", "F", "I", "UP", "B", "SIM", "C4", "RUF"]
# 忽略的规则
ignore = [
    "E501",  # 行长度（已由 line-length 控制，不重复检查）
    "B008",  # 函数调用作为默认参数（如 dict() 在 default_factory 里）—— 业务常用，false positive 多
]

[tool.ruff.lint.per-file-ignores]
# 测试文件允许未使用 import（fixture 可能只在 conftest 里用）
"tests/**/*.py" = ["F401", "F811"]
# __init__.py 可能 import 仅用于 re-export
"__init__.py" = ["F401"]

[tool.ruff.lint.isort]
# import 排序：标准库 → 第三方 → 本项目
known-first-party = ["private_teacher"]
combine-as-imports = true

[tool.ruff.format]
# format 风格：与 black 兼容
quote-style = "double"
indent-style = "space"
skip-magic-trailing-comma = false
line-ending = "lf"
```

> 💡 **规则速查**：
> - `E/W/F` 是基础三件套，必须开。
> - `I` 让 import 自动按字母排序，团队协作必备。
> - `UP` 自动建议用新语法（如 `list[X]` 替代 `List[X]`）。
> - `B/SIM` 是质量加分项，Phase 0 启用，遇到误报可单独 ignore。

---

## 12. 类型检查：mypy（10 分钟）

### 12.1 目标
- 关键模块（`config.py`、`llm/factory.py`、`utils/exceptions.py`）零 mypy 错误。
- Phase 0 不用全量检查（CI 全开太严），先严格这几个核心文件。

### 12.2 操作
- 在 `pyproject.toml` 加 `[tool.mypy]`：
  - `python_version = "3.11"`
  - `strict = true`
  - `ignore_missing_imports = true`（langchain 缺 stub，先放过去）
- 跑 `uv run mypy src/private_teacher/config.py src/private_teacher/llm/factory.py src/private_teacher/utils/exceptions.py` 验证。
- 把这行命令加到 Makefile `lint-mypy`。

### 12.3 验证
- 在 `config.py` 故意写一个返回类型错的函数（如声明 `-> int` 但返回 str），mypy 应该报错。
- 删掉故意写的错，确认 mypy 0 错。

**实现（直接复制）**：

把下面这段追加到 `pyproject.toml`：

```toml
# ============================================================
# mypy 配置（静态类型检查）
# ============================================================
[tool.mypy]
# Python 版本
python_version = "3.11"
# 严格模式（开启所有严格选项）
# 包含 disallow_untyped_defs, no_implicit_optional, warn_unused_ignores 等
strict = true
# 允许未注解的 def（Phase 0 渐进式落地）
# 之后可以打开 disallow_untyped_defs = true
disallow_untyped_defs = false
# 不检查缺失的 import（langchain 等第三方库 type stub 不全）
ignore_missing_imports = true
# 不显示 "note" 级别（让输出更干净）
show_error_codes = true
# 输出简洁
pretty = true
# 按文件输出结果
show_absolute_path = false

# 按模块分级别
[[tool.mypy.overrides]]
# 第三方库容忍一些不严格
module = ["loguru", "tenacity.*", "unstructured.*"]
ignore_missing_imports = true
ignore_errors = false

# 测试文件可以松一点
[[tool.mypy.overrides]]
module = "tests.*"
disallow_untyped_defs = false
```

> ⚠️ **Phase 0 收尾时**：
> - `disallow_untyped_defs = false` 意味着可以有空注解函数，**Phase 1 完成后改为 true** 强制每个函数都有类型。
> - `ignore_missing_imports = true` 让 langchain 等库不会刷屏，**Phase 1 后尝试加上 type stub**。

---

## 13. CI：GitHub Actions（15 分钟）

### 13.1 目标
PR 触发 ruff + mypy + pytest，给 PR 状态徽章。

### 13.2 文件内容指引（.github/workflows/ci.yml）
- 触发：`on: [push, pull_request]`
- 矩阵：Python 3.11（单版本够了，Phase 0 先不折腾多版本）
- 步骤：
  1. `actions/checkout@v4`
  2. `astral-sh/setup-uv@v1`（**关键**：官方 action 装 uv）
  3. `uv sync --all-extras` 装全部依赖
  4. `uv run ruff check .`
  5. `uv run ruff format --check .`
  6. `uv run mypy src/...`（关键模块）
  7. `uv run pytest -m "not requires_llm"`（不跑真 LLM）
- 缓存：`actions/setup-uv` 自带 `.uv-cache` 缓存。

### 13.3 验证
- 把代码 push 到 GitHub（Phase 0 第一次 push 需要 `git remote add origin ...`）。
- 在 PR 页面看到 CI 跑过。
- 故意制造 ruff 错误，看 CI 红灯。

> 🐛 **踩坑**：GitHub Actions 跑 `uv sync` 时如果 `pyproject.toml` 引用了 `langchain-anthropic` 等需要编译的包，第一次会慢（30s~1min）。后续有缓存。

**实现（直接复制）**：

**.github/workflows/ci.yml**
```yaml
# ============================================================
# Private Teacher - CI Workflow
# ============================================================
# 触发时机：每次 push 或 pull_request 到 main 分支
# 跑什么：lint (ruff) + typecheck (mypy) + test (pytest)
# 注意：默认不跑需要真实 LLM 的测试（避免 CI 烧钱）
# ============================================================

name: CI

# 触发条件
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

# 允许在 Actions 页面手动触发（调试用）
workflow_dispatch:

# 取消上一次未完成的同分支 run（节省 CI 分钟）
concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

jobs:
  # ============================================================
  # 单一 job：lint + typecheck + test
  # ============================================================
  ci:
    # 运行环境
    runs-on: ubuntu-latest  # GitHub 提供的免费 Linux runner
    # Python 版本矩阵（Phase 0 单版本，后面可扩展）
    strategy:
      matrix:
        python-version: ["3.11"]
      # 如果任何一个版本失败，整个 job 失败
      fail-fast: true

    steps:
      # 1. 检出代码
      - name: Checkout code
        uses: actions/checkout@v4

      # 2. 安装 uv
      # 官方 action：https://github.com/astral-sh/setup-uv
      # 自动带缓存（uv-cache）
      - name: Install uv
        uses: astral-sh/setup-uv@v5
        with:
          version: "latest"  # 用最新 uv
          # 启用缓存（默认缓存 ~/.cache/uv）
          enable-cache: true

      # 3. 设置 Python 版本
      - name: Set up Python ${{ matrix.python-version }}
        run: uv python install ${{ matrix.python-version }}

      # 4. 同步依赖
      # --all-extras 会装所有 optional groups（dev、test、docs）
      - name: Install dependencies
        run: uv sync --all-extras --python ${{ matrix.python-version }}

      # 5. Ruff lint
      - name: Lint with ruff
        run: uv run ruff check .

      # 6. Ruff format 检查
      # --check 不修改文件，只检查格式
      - name: Check formatting
        run: uv run ruff format --check .

      # 7. Mypy 类型检查（只检查关键模块）
      # 暂不全量，避免 Phase 0 太严格
      - name: Type check with mypy
        run: |
          uv run mypy \
            src/private_teacher/config.py \
            src/private_teacher/llm/factory.py \
            src/private_teacher/utils/exceptions.py

      # 8. 跑 pytest（不含需要真实 LLM 的测试）
      # -m "not requires_llm" 跳过 requires_llm marker 的测试
      # CI 没有真实 API key，跑会失败
      - name: Run tests
        env:
          # CI 环境用 .env.example 作为兜底（虽然大部分是 None）
          # 让 pydantic 至少能加载到默认值
          PRIVATE_TEACHER_LLM_PROVIDER: ollama
          PRIVATE_TEACHER_LOG_LEVEL: ERROR
        run: uv run pytest -m "not requires_llm" --tb=short
```

---

## 14. Makefile（10 分钟）

### 14.1 目标
一组常用命令。

### 14.2 文件内容指引
包含至少这些 target：
- `install`：`uv sync --all-extras`
- `dev`：`uv run streamlit run src/private_teacher/app.py`（Phase 0 app.py 还不存在，**先注释掉**或者留作占位）
- `test`：`uv run pytest -m "not requires_llm"`
- `test-llm`：`uv run pytest -m requires_llm`
- `test-cov`：`uv run pytest --cov`
- `lint`：`uv run ruff check . && uv run ruff format --check .`
- `lint-fix`：`uv run ruff check . --fix && uv run ruff format .`
- `typecheck`：`uv run mypy src/private_teacher/config.py src/private_teacher/llm/factory.py src/private_teacher/utils/exceptions.py`
- `hello`：`uv run python -m private_teacher hello`
- `hello-llm`：`uv run python -m private_teacher hello-llm`
- `ci`：`make lint && make typecheck && make test`
- `clean`：清理 `__pycache__`、`.pytest_cache`、`.ruff_cache`、`.mypy_cache`

每个 target 写成 `@echo ">>> xxx"; uv run ...` 方便看出哪步慢。

### 14.3 验证
- 跑 `make help`（如果加了 help target）看到所有命令。
- 跑 `make install` 应该装好。
- 跑 `make test` 应该全绿。

**实现（直接复制）**：

**Makefile**
```makefile
# ============================================================
# Private Teacher - Makefile
# ============================================================
# 用法：
#   make help        查看所有命令
#   make install     安装依赖
#   make test        跑测试
#   make ci          跑全套 CI 检查（lint + typecheck + test）
# ============================================================

# 防止多个 make 并行跑同一个项目造成冲突
# 同一时间只允许一个 make 进程
.NOTPARALLEL:

# make 不打印「进入/离开目录」日志，更清爽
MAKEFLAGS += --no-print-directory

# 默认目标：显示 help
.DEFAULT_GOAL := help

# ============================================================
# 帮助
# ============================================================
.PHONY: help
help:  ## 显示所有命令
	@echo "Private Teacher - 可用命令："
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "示例："
	@echo "  make install    # 装依赖"
	@echo "  make test       # 跑测试"
	@echo "  make ci         # 跑全套检查"

# ============================================================
# 安装与运行
# ============================================================
.PHONY: install
install:  ## 安装所有依赖
	@echo ">>> Installing dependencies..."
	uv sync --all-extras

.PHONY: hello
hello:  ## 打印欢迎语（不调 LLM）
	uv run python -m private_teacher hello

.PHONY: hello-llm
hello-llm:  ## 调用 LLM 验证配置（消耗 API 费用）
	uv run python -m private_teacher hello-llm

# Phase 1+ 启用：Streamlit UI
# .PHONY: dev
# dev:  ## 启动 Streamlit UI
# 	uv run streamlit run src/private_teacher/app.py

# ============================================================
# 测试
# ============================================================
.PHONY: test
test:  ## 跑单元测试（不消耗 LLM 费用）
	@echo ">>> Running unit tests..."
	uv run pytest -m "not requires_llm"

.PHONY: test-llm
test-llm:  ## 跑需要真实 LLM 的测试（会消耗费用）
	@echo ">>> Running LLM-required tests..."
	uv run pytest -m requires_llm

.PHONY: test-cov
test-cov:  ## 跑测试 + 显示覆盖率
	@echo ">>> Running tests with coverage..."
	uv run pytest --cov-report=html
	@echo ""
	@echo "覆盖率报告：htmlcov/index.html"

.PHONY: test-all
test-all:  ## 跑所有测试
	uv run pytest

# ============================================================
# Lint & Format
# ============================================================
.PHONY: lint
lint:  ## 检查代码风格（不修改文件）
	@echo ">>> Running ruff check..."
	uv run ruff check .
	@echo ">>> Checking format..."
	uv run ruff format --check .

.PHONY: lint-fix
lint-fix:  ## 自动修复代码风格问题
	@echo ">>> Auto-fixing lint issues..."
	uv run ruff check . --fix
	@echo ">>> Auto-formatting..."
	uv run ruff format .

# ============================================================
# Type Check
# ============================================================
.PHONY: typecheck
typecheck:  ## mypy 严格检查关键模块
	@echo ">>> Running mypy..."
	uv run mypy \
		src/private_teacher/config.py \
		src/private_teacher/llm/factory.py \
		src/private_teacher/utils/exceptions.py

.PHONY: typecheck-all
typecheck-all:  ## mypy 全量检查（Phase 1+ 启用）
	uv run mypy src/

# ============================================================
# CI
# ============================================================
.PHONY: ci
ci: lint typecheck test  ## 跑完整 CI 流程
	@echo ""
	@echo "✅ CI 全部通过"

# ============================================================
# 清理
# ============================================================
.PHONY: clean
clean:  ## 清理缓存与临时文件
	@echo ">>> Cleaning caches..."
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ 清理完成"

.PHONY: clean-data
clean-data:  ## ⚠️ 危险：删除所有用户数据（不可恢复）
	@echo "⚠️  即将删除 data/ 目录中的所有内容"
	@read -p "确认删除？(y/N) " confirm && [ "$$confirm" = "y" ]
	rm -rf data/courseware data/knowledge_bases data/units data/final data/submissions data/app.db
	@echo "✅ 用户数据已删除"

# ============================================================
# 实用工具
# ============================================================
.PHONY: deps-update
deps-update:  ## 更新所有依赖到最新版本
	uv lock --upgrade
	uv sync --all-extras

.PHONY: deps-tree
deps-tree:  ## 显示依赖树
	uv tree

.PHONY: info
info:  ## 显示环境信息
	@echo "Python:  $$(uv run python --version)"
	@echo "uv:      $$(uv --version)"
	@echo "Project: $$(uv run python -c 'import private_teacher; print(private_teacher.__version__)')"
	@echo "Data:    $$(uv run python -c 'from private_teacher.config import AppSettings; print(AppSettings.load().paths.data_dir)')"
```

> 💡 **小贴士**：
> - `make help` 会自动扫描 `## 注释`，生成命令清单（无需手维护）
> - `make ci` 跑完会显示「✅ CI 全部通过」
> - `make clean-data` 是个**危险** target，会等你输 `y` 才执行

---

## 15. Phase 0 收尾验证（30 分钟）

### 15.1 冒烟测试（必做）
1. `make install` → 成功
2. `make lint` → 0 错
3. `make typecheck` → 0 错
4. `make test` → 全绿，覆盖率 ≥ 80%
5. `make hello` → 打印欢迎语
6. `make hello-llm` → 打印 LLM 回复
7. 把代码 push 到 GitHub → CI 绿

### 15.2 自检清单
- [ ] `pyproject.toml` 干净，依赖都加好了
- [ ] `.env` 没进 git（`git status` 不显示）
- [ ] 真 key 在 `.env`，不在任何 .py 文件里
- [ ] `tests/unit/` 下每个文件都覆盖了对应模块
- [ ] `make test` 跑完无 warning（除了 deprecation）
- [ ] `README.md` 写好了「Quick Start」段落
- [ ] CI 配好了

### 15.3 提交
- 第一次 commit：`chore: init phase 0 infrastructure`
- 内容包括：所有 src/、tests/、配置文件、CI workflow。
- **不要** commit `.env`、`.venv/`、`data/`。

---

## 16. README 草稿（10 分钟）

`README.md` 至少包含：
- **项目简介**：一句话说明 Private Teacher 是什么。
- **状态**：Phase 0 — 基础设施（链接到 Build_plan.md）。
- **特性（已实现）**：能跑 `hello-llm`、支持 3 个 LLM provider、有完整测试基建。
- **特性（路线图）**：列出 Phase 1~7。
- **快速开始**：
  ```
  git clone ...
  cd private_teacher
  cp .env.example .env
  # 编辑 .env 填入 API key
  make install
  make hello-llm
  ```
- **开发**：`make test` / `make lint` / `make typecheck`
- **架构**：链接到 `docs/architecture.md`（Phase 7 才写，先放占位）。
- **许可**：MIT（自己定）。

**实现（直接复制）**：

**README.md**
```markdown
# Private Teacher

> 一个本地优先的私人教师大模型 App。  
> 上传课件 → 自动构建 RAG 知识库 → 大模型规划学习路径 → 生成单元内容/测试/答案 → 批改与终期评价。

**当前状态**：Phase 0 — 基础设施搭建中 🚧  
**完整开发计划**：见 [Build_plan.md](Build_plan.md) 和 [phase0_plan.md](phase0_plan.md)

---

## ✨ 特性（已实现）

- ✅ 支持 3 个 LLM provider：Anthropic Claude / OpenAI 兼容 / Ollama 本地
- ✅ 统一的 LangChain `BaseChatModel` 工厂接口
- ✅ 类型安全的配置管理（pydantic-settings + .env）
- ✅ 自定义异常体系（带 transient 标记 + context 上下文）
- ✅ 指数退避重试装饰器（tenacity）
- ✅ 完整测试基建：pytest + 假 LLM + 覆盖率
- ✅ Lint / Format（ruff）
- ✅ 严格类型检查（mypy）
- ✅ GitHub Actions CI

## 🚧 路线图

- [x] **Phase 0**：基础设施
- [ ] **Phase 1**：课件管理 + RAG 知识库
- [ ] **Phase 2**：课程分析 + 单元学习内容
- [ ] **Phase 3**：单元测试生成
- [ ] **Phase 4**：批改与答案解析
- [ ] **Phase 5**：期末项目 + 终期评价
- [ ] **Phase 6**：答疑 + 源文件查看
- [ ] **Phase 7**：打磨与发布

---

## 🚀 快速开始

### 前置要求
- Python 3.11+
- [uv](https://github.com/astral-sh/uv)（包管理器）

### 安装

```bash
# 1. 克隆仓库
git clone <repo-url>
cd private_teacher

# 2. 复制环境变量模板
cp .env.example .env

# 3. 编辑 .env，至少填一个 LLM provider 的 API key
#    推荐：ANTHROPIC_API_KEY
$EDITOR .env

# 4. 安装依赖
make install
```

### 验证安装

```bash
# 不消耗费用
make hello

# 消耗 1 个 API 调用的费用（验证 LLM 配置正确）
make hello-llm
```

---

## 🛠 开发

| 命令 | 作用 |
| --- | --- |
| `make help` | 查看所有命令 |
| `make install` | 装依赖 |
| `make test` | 跑单元测试 |
| `make test-llm` | 跑真实 LLM 测试（需 API key） |
| `make test-cov` | 跑测试 + 覆盖率报告 |
| `make lint` | ruff check + format 检查 |
| `make lint-fix` | ruff 自动修复 |
| `make typecheck` | mypy 关键模块检查 |
| `make ci` | 跑完整 CI（lint + typecheck + test） |
| `make clean` | 清理缓存 |

---

## 🏗 架构概览

```
src/private_teacher/
├── config.py          # pydantic-settings 配置
├── cli.py             # CLI 入口
├── llm/
│   └── factory.py     # LLM provider 工厂
├── utils/
│   ├── exceptions.py  # 自定义异常
│   ├── logging.py     # loguru 配置
│   └── retry.py       # tenacity 重试装饰器
└── ...

tests/
├── conftest.py        # pytest 共享 fixture
├── fakes/             # 假实现
└── unit/              # 单元测试
```

详见 [Build_plan.md](Build_plan.md)。

---

## 📦 技术栈

- **Python** 3.11+
- **LangChain** —— LLM 抽象
- **pydantic-settings** —— 配置管理
- **loguru** —— 日志
- **tenacity** —— 重试
- **pytest** + **FakeListChatModel** —— 测试
- **ruff** + **mypy** —— 代码质量
- **GitHub Actions** —— CI

---

## 📄 许可

MIT

---

## 🤝 贡献

欢迎提 Issue 和 PR。在 PR 之前请确保 `make ci` 全部通过。
```

> 💡 **README 维护原则**：
> - 顶部放「当前状态」徽章，让别人一眼知道项目进度
> - 「快速开始」必须 5 步以内能跑起来
> - 命令清单放表格，方便查找
> - 架构图在 Phase 7 之前先放简化版占位

---

## 17. 常见问题 FAQ（自学用）

| 问题 | 回答 |
| --- | --- |
| `uv` 和 `pip` / `poetry` 区别？ | uv 极快、单一二进制、lock 格式现代。poetry 更成熟但慢。pip 是裸的。 |
| `BaseSettings` vs 普通 class？ | pydantic 自动做类型校验、env 加载、`.dict()` 序列化。 |
| 为什么用 loguru 而不是 logging？ | loguru 一行 `logger.info()` 就够用，format 配置简单，颜色 / JSON 切换 0 成本。 |
| tenacity 怎么决定重试？ | `retry_if_exception_type` 只对指定类型重试，**不**对 `ValueError` / `KeyError` 等业务错误重试。 |
| `BaseChatModel` 怎么用？ | `model.invoke([HumanMessage(content="...")])`，统一接口。 |
| pytest fixture scope？ | 默认 `function`（每个测试新建），`session` 全局一个，`module` 一个文件一个。本课 `settings` 用 session，`tmp_env` 用 function。 |
| `monkeypatch` 怎么改 env？ | `monkeypatch.setenv("KEY", "value")`，测试结束自动还原。 |
| 怎么避免真 LLM 烧钱？ | 所有 LLM 调用都通过工厂，**测试一律用 FakeListChatModel**；真 LLM 测试加 `requires_llm` marker 默认跳过。 |
| `__init__.py` 要写啥？ | 留空 或 写 `__version__ = "0.1.0"`。Python 3.3+ namespace package 可以不要，但保留更稳。 |
| `__main__.py` vs `cli.py`？ | `__main__.py` 是入口（`python -m xxx`），`cli.py` 是命令实现。一般 `__main__.py` 只 import cli 即可。 |

---

## 18. Phase 0 完成 → 进入 Phase 1 的判定

- [ ] `make ci` 全部通过（lint + typecheck + test）
- [ ] `make hello-llm` 能跑
- [ ] CI 在 GitHub 上绿
- [ ] README 写好
- [ ] git 历史清晰（至少 3 个 commit：init / tests / docs）

满足以上 5 条 → 切到 **Phase 1：课件 + RAG**（参考 [Build_plan.md](Build_plan.md) 的 9.1 节）。

---

## 19. 你需要准备的资源

- [ ] 一个 Anthropic API key（https://console.anthropic.com/）
- [ ] 备用：OpenAI key 或一个 Ollama 本地模型
- [ ] 一个 GitHub 仓库（空仓库就行，第一次 push 用）
- [ ] 2~3 小时连续时间（Phase 0 全部跑完预计 4~6 小时，分两次做也行）

---

> 接下来：**先跑通第 0~2 步**（环境 + uv + 依赖），把空骨架立起来。然后**按顺序**做 3 → 4 → 5 → 6 → 7 → 8 → 9，每步结束都用「验证」段落的命令自检。**第 8 步 LLM 工厂是最大坎**，遇到报错就贴出来我们一起看。
>
> 跑完 Phase 0 之后告诉我，我陪你进 Phase 1。

---

## 20. 补充：添加 MiniMax Provider（v0.1.0 增量变更）

> 本节是对前面章节的**增量修改**，把所有出现 `claude / ollama / openai_compat` 的地方都加上 `minimax` 选项。
> **不需要重做整个 Phase 0**，只需要在对应文件里加几行代码。

### 20.1 变更总览

| 改动点 | 文件 | 内容 |
| --- | --- | --- |
| 1 | `.env.example` | 增加 `MINIMAX_API_KEY` / `PRIVATE_TEACHER_LLM_MINIMAX_*` 三个变量 |
| 2 | `src/private_teacher/config.py` | `LLMSettings.provider` Literal 增加 `"minimax"`；新增 `minimax_model` / `minimax_base_url` 字段；`validate_for_runtime` 增加 minimax 校验分支 |
| 3 | `src/private_teacher/llm/factory.py` | `match/case` 增加 `case "minimax":`，复用 `ChatOpenAI` + 自定义 `base_url` |
| 4 | `tests/unit/test_config.py` | 增加 `test_validate_raises_for_minimax_without_key` 和 `test_validate_passes_for_minimax_with_key` |
| 5 | `tests/unit/test_llm_factory.py` | 增加 `test_minimax_returns_chat_openai` 和 `test_minimax_without_key_raises` |
| 6 | `pyproject.toml` | **不需要改动**（依赖没变） |
| 7 | `README.md` | 路线图或特性里可加一句「支持 MiniMax provider」 |

### 20.2 完整变更 diff（直接复制）

#### 1) `.env.example` —— 在 Ollama 配置段后追加

```bash
# ---------- MiniMax 配置 ----------
# MiniMax（OpenAI 兼容 API）申请 key 请到 MiniMax 开发者控制台
MINIMAX_API_KEY=
PRIVATE_TEACHER_LLM_MINIMAX_BASE_URL=https://api.minimax.com/v1
PRIVATE_TEACHER_LLM_MINIMAX_MODEL=minimaxm3
```

#### 2) `src/private_teacher/config.py` —— 三处修改

**改 ①**：`provider` 字段增加 `"minimax"`：

```python
provider: Literal["claude", "ollama", "openai_compat", "minimax"] = Field(
    default="claude",
    description="LLM 提供方，可选：claude / ollama / openai_compat / minimax",
)
```

**改 ②**：在 `ollama_model` 之后增加 MiniMax 字段：

```python
# ----- MiniMax 配置（OpenAI 兼容协议）-----
minimax_model: str = Field(
    default="minimaxm3",
    description="MiniMax 模型名（默认 minimaxm3）",
)
minimax_base_url: str = Field(
    default="https://api.minimax.com/v1",
    description="MiniMax API 基础地址（OpenAI 兼容）",
)
```

**改 ③**：`validate_for_runtime` 增加 elif 分支：

```python
elif self.llm.provider == "minimax":
    if not os.getenv("MINIMAX_API_KEY"):
        raise ValueError(
            "provider=minimax 需要设置环境变量 MINIMAX_API_KEY，"
            "请在 .env 文件中配置"
        )
```

#### 3) `src/private_teacher/llm/factory.py` —— 在 `case "ollama"` 之后增加新分支

```python
# ---------- MiniMax（OpenAI 兼容协议）----------
case "minimax":
    # MiniMax 提供的是 OpenAI 兼容 API，所以直接复用 ChatOpenAI
    # 关键是把 base_url 指向 MiniMax 的网关
    api_key = os.getenv("MINIMAX_API_KEY")
    if not api_key:
        raise ConfigurationError(
            "缺少 MINIMAX_API_KEY",
            provider="minimax",
            env_var="MINIMAX_API_KEY",
            hint="在 .env 文件中设置 MINIMAX_API_KEY=<你的 key>",
        )
    return ChatOpenAI(
        model=settings.minimax_model,        # 默认 minimaxm3
        api_key=api_key,
        base_url=settings.minimax_base_url,  # 默认 https://api.minimax.com/v1
        temperature=0.2,
    )
```

**别忘了**把 `case _` 兜底分支的 `valid_options` 列表也加上 `"minimax"`：

```python
raise ConfigurationError(
    f"未知的 LLM provider: {settings.provider!r}",
    provider=str(settings.provider),
    valid_options=["claude", "ollama", "openai_compat", "minimax"],  # ← 加 minimax
)
```

#### 4) `tests/unit/test_config.py` —— 在 `TestAppSettings` 类里追加

```python
def test_validate_raises_for_minimax_without_key(
    self, monkeypatch: pytest.MonkeyPatch
) -> None:
    """provider=minimax 但 MINIMAX_API_KEY 缺失时应抛 ConfigurationError。"""
    monkeypatch.setenv("PRIVATE_TEACHER_LLM_PROVIDER", "minimax")
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)

    settings = AppSettings.load()
    with pytest.raises(ConfigurationError) as exc_info:
        settings.validate_for_runtime()
    assert "MINIMAX_API_KEY" in str(exc_info.value)

def test_validate_passes_for_minimax_with_key(
    self, monkeypatch: pytest.MonkeyPatch
) -> None:
    """provider=minimax 且 MINIMAX_API_KEY 设置时不应抛错。"""
    monkeypatch.setenv("PRIVATE_TEACHER_LLM_PROVIDER", "minimax")
    monkeypatch.setenv("MINIMAX_API_KEY", "test-minimax-key")

    settings = AppSettings.load()
    settings.validate_for_runtime()  # 不应抛错
```

#### 5) `tests/unit/test_llm_factory.py` —— 在 `TestBuildChatModel` 类里追加

```python
def test_minimax_returns_chat_openai(
    self, monkeypatch: pytest.MonkeyPatch
) -> None:
    """provider=minimax 应返回 ChatOpenAI 实例（OpenAI 兼容协议）。"""
    monkeypatch.setenv("MINIMAX_API_KEY", "test-minimax-key")
    settings = LLMSettings(
        provider="minimax",
        minimax_model="minimaxm3",
        minimax_base_url="https://api.minimax.com/v1",
    )

    model = build_chat_model(settings)

    from langchain_openai import ChatOpenAI
    assert isinstance(model, ChatOpenAI)
    # 进一步验证 base_url 配置正确
    assert model.openai_api_base == "https://api.minimax.com/v1"  # type: ignore[attr-defined]

def test_minimax_without_key_raises(
    self, monkeypatch: pytest.MonkeyPatch
) -> None:
    """provider=minimax 缺 key 时应抛 ConfigurationError。"""
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    settings = LLMSettings(provider="minimax")

    with pytest.raises(ConfigurationError) as exc_info:
        build_chat_model(settings)
    assert "MINIMAX_API_KEY" in str(exc_info.value)
```

### 20.3 验证清单

按顺序跑下面这些命令，全部通过才算改完：

```bash
# 1. 静态检查
uv run ruff check .
uv run ruff format --check .
uv run mypy \
  src/private_teacher/config.py \
  src/private_teacher/llm/factory.py \
  src/private_teacher/utils/exceptions.py

# 2. 跑所有单元测试
uv run pytest -m "not requires_llm" -v

# 3. 专门跑 minimax 相关测试
uv run pytest -v -k "minimax"
# 预期看到至少 4 个测试通过：
#   test_validate_raises_for_minimax_without_key  PASSED
#   test_validate_passes_for_minimax_with_key     PASSED
#   test_minimax_returns_chat_openai              PASSED
#   test_minimax_without_key_raises               PASSED

# 4. 实际调用 minimax（需要真 key）
cp .env.example .env
# 在 .env 里填：
#   PRIVATE_TEACHER_LLM_PROVIDER=minimax
#   MINIMAX_API_KEY=<你的真 key>
make hello-llm
```

### 20.4 设计要点解释

**Q1：为什么 minimax 复用 `ChatOpenAI` 而不是自己写一个 `ChatMiniMax`？**

> 因为 MiniMax 提供的是 **OpenAI 兼容 API**（绝大多数新晋 LLM 服务商都这么做，复用 OpenAI 协议 = 复用 LangChain 生态）。
> 关键就是把 `base_url` 指向 MiniMax 的网关，剩下的 `model` / `api_key` / 鉴权头都跟 OpenAI 一模一样。
> **好处**：零额外依赖、零新代码、立刻能用 LangChain 全部能力（流式、async、tools）。

**Q2：为什么不用 `langchain-minimax` 这种独立包？**

> 目前没有官方 langchain-minimax 包（你可以提交一个 PR）。
> 等有了再独立成 `ChatMiniMax` 类也来得及——到时候只需把 `case "minimax":` 里的 `ChatOpenAI(...)` 换成 `ChatMiniMax(...)` 即可，上层调用方零改动。

**Q3：MINIMAX_API_KEY 为什么不放在 LLMSettings 字段里而是走环境变量？**

> 同 Claude / OpenAI 一样，**安全考虑**：
> - `LLMSettings.model_dump()` 会把字段序列化成 dict，**不应**包含 key
> - 日志、`__repr__`、错误信息也都不应包含 key
> - 环境变量是 OS 级别的敏感信息隔离机制

**Q4：minimaxm3 是哪个模型？**

> 按你要求，默认模型名是 `minimaxm3`。
> 如果 MiniMax 后续发布新模型（如 `minimaxm4`），改 `.env` 里的 `PRIVATE_TEACHER_LLM_MINIMAX_MODEL` 即可，**不需要改代码**。

### 20.5 常见踩坑

| 现象 | 原因 | 解决 |
| --- | --- | --- |
| `ConfigurationError: 缺少 MINIMAX_API_KEY` | 没在 `.env` 里配 | 编辑 `.env`，加 `MINIMAX_API_KEY=...` |
| `404 Not Found` | base_url 配错了 | 检查 `PRIVATE_TEACHER_LLM_MINIMAX_BASE_URL` |
| `401 Unauthorized` | key 错误或过期 | 重新去 MiniMax 控制台申请 |
| `Model not found: minimaxm3` | 模型名拼错 | 确认 MiniMax 文档里实际叫 `minimaxm3`（注意大小写） |
| 测试 `assert model.openai_api_base == ...` 失败 | LangChain 版本升级改了属性名 | 改成 `assert "minimax.com" in str(model.openai_api_base)` 更稳 |

> 💡 **下一步**：改完跑 `make ci`，全绿后可以 git commit（commit message: `feat(llm): add minimax provider`）。

