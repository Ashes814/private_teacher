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