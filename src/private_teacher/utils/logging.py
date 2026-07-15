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


# 日志配置函数
def configure_logging(settings: LogSettings) -> None:
    """
    根据配置初始化全局 logger。

    调用时机：CLI 入口、Streamlit 应用启动、测试启动
    调用次数：通常只调一次，重复调用会先 remove 再 add（不会重复输出）

    Args:
        settings: 日志配置
    """

    # 1. 先清掉loguru默认的stderr handler
    # 为什么要 remove？避免与其他 handler 重复输出
    logger.remove()

    # 2. 根据json_logs决定格式
    if settings.json_logs:
        # JSON格式：生产环境用，方便日志聚合系统（Loki / ELK）解析
        # 这里用loguru内置的serialize = True 让loguru自动序列化为JSON
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
        colorize=not settings.json_logs,
        backtrace=True,
        diagnose=False,
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
    logger.debug(f"日志初始化wjig：level={settings.level}, json_logs={settings.json_logs}")


# ============================================================
# 模块级便利函数（可选）
# ============================================================
def get_logger() -> logger:
    """
    获取全局 logger 实例（推荐直接 from loguru import logger 即可，这里只是备用）。

    用法：
        from private_teacher.utils.logging import get_logger
        log = get_logger()
        log.info("hi")
    """
    return logger
