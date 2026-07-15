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
    ConfigurationError,
    DocumentLoadError,
    GradingError,
    LLMError,
    NotebookError,
    PrivateTeacherError,
    RAGError,
)


class TestPrivateTeacherError:
    """测试根异常"""

    def test_can_be_caught_by_base_class(self) -> None:
        """子类应能被根异常捕获"""
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
    """测试异常继承关系"""

    @pytest.mark.parametrize(
        "exc_cls",
        [
            ConfigurationError,
            LLMError,
            DocumentLoadError,
            RAGError,
            NotebookError,
            GradingError,
        ],
    )
    def test_all_inherit_from_root(self, exc_cls: type) -> None:
        """所有自定义异常都应继承 PrivateTeacherError。"""
        assert issubclass(exc_cls, PrivateTeacherError)
