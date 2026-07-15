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

from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from private_teacher.utils.exceptions import LLMError
from private_teacher.utils.logging import logger


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
                isinstance(exc, retry_on) and not (isinstance(exc, LLMError) and not exc.transient)
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
    initial_wait=2.0,  # LLM API 通常 rate limit 较严，初始等待稍长
    max_wait=60.0,
    retry_on=(LLMError, TimeoutError, ConnectionError),
    use_llm_transient_check=True,  # 关键：transient=False 的 LLMError 不重试
)
