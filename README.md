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
# 1. 克隆仓库 {#1-克隆仓库  data-source-line="3561"}
git clone <repo-url>
cd private_teacher

# 2. 复制环境变量模板 {#2-复制环境变量模板  data-source-line="3565"}
cp .env.example .env

# 3. 编辑 .env，至少填一个 LLM provider 的 API key {#3-编辑-env至少填一个-llm-provider-的-api-key  data-source-line="3568"}
# 推荐：ANTHROPIC_API_KEY {#推荐anthropic\_api\_key  data-source-line="3569"}
$EDITOR .env

# 4. 安装依赖 {#4-安装依赖  data-source-line="3572"}
make install
``` {data-source-line="3574"}

### 验证安装

```bash
# 不消耗费用 {#不消耗费用  data-source-line="3579"}
make hello

# 消耗 1 个 API 调用的费用（验证 LLM 配置正确） {#消耗-1-个-api-调用的费用验证-llm-配置正确  data-source-line="3582"}
make hello-llm
``` {data-source-line="3584"}

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
src/private_teacher/
├── config.py # pydantic-settings 配置
├── cli.py # CLI 入口
├── llm/
│ └── factory.py # LLM provider 工厂
├── utils/
│ ├── exceptions.py # 自定义异常
│ ├── logging.py # loguru 配置
│ └── retry.py # tenacity 重试装饰器
└── ...

tests/
├── conftest.py # pytest 共享 fixture
├── fakes/ # 假实现
└── unit/ # 单元测试

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