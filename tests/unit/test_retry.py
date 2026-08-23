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
from itertools import pairwise

import pytest

from private_teacher.utils.exceptions import LLMError
from private_teacher.utils.retry import make_retry_decorator


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

    def test_wait_time_grows_exponentially(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """等待时间应大致呈指数增长（带抖动）。"""
        timestamps: list[float] = []
        sleeps: list[float] = []  # 记录每次 sleep 的"请求时长"

        @make_retry_decorator(
            max_attempts=4,
            initial_wait=0.05,
            max_wait=1.0,
            retry_on=(ValueError,),
        )
        def always_fail() -> None:
            timestamps.append(time.monotonic())
            raise ValueError("fail")

        # 把 time.sleep 替换成"只记录、不真睡"
        # tenacity 内部最终也是调 time.sleep，所以这一行覆盖所有 retry 路径
        # 这样测试只关心退避逻辑本身，不受系统负载 / CI 抖动影响
        monkeypatch.setattr(
            "time.sleep",
            lambda s: sleeps.append(s),
        )

        with pytest.raises(ValueError):
            always_fail()

        # 退避序列：理论上 0.05, 0.10, 0.20（带 jitter）
        assert len(sleeps) == 3  # 4 次尝试之间有 3 次等待
        # jitter 可能让相邻几次看起来"乱"（jitter 范围 0~max_wait），
        # 但整体趋势一定递增 —— 最后一个比第一个明显长
        assert sleeps[-1] > sleeps[0], (
            f"期望指数增长，实际 sleeps={sleeps}（{sleeps[-1]} <= {sleeps[0]}）"
        )


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
