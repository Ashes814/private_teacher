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
from private_teacher.llm.factory import quick_chat  # 便捷 LLM 调用
from private_teacher.utils.exceptions import ConfigurationError, LLMError
from private_teacher.utils.logging import configure_logging  # 日志配置


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
    print(f"Hello I am your Private Teacher! v{__version__}")
    print("Phase 0 is ready. Try `python -m private_teacher hello-llm` next.")
    return 0  # 0 表示成功


def cmd_hello_llm(args: argparse.Namespace) -> int:
    """hello-llm 子命令：调用 LLM 让它自我介绍。

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
    构造argparse解析器

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
