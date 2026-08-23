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


# 根异常
class PrivateTeacherError(Exception):
    """项目根异常。

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
            **context: 任意附加上下文，会在__str__中打印
        """
        # 调用父类Exception的__init__, 让traceback正常显示
        super().__init__(message)
        self.message = message
        self.context: dict[str, object] = context

    def __str__(self) -> str:
        """
        自定义字符串表示：把 context 拼接到 message 后面。

        例子：
          message="API 调用失败", context={"status": 500, "url": "..."}
          输出: "API 调用失败 | status=500, url=..."
        """

        if not self.context:
            return self.message

        ctx_str = ", ".join(f"{k}={v!r}" for k, v in self.context.items())
        return f"{self.message} | {ctx_str}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.message!r}, **{self.context!r})"


# 配置相关


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


# LLM相关
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


# 文档加载相关（Phase 1 使用，先建好避免循环 import）
class DocumentLoadError(PrivateTeacherError):
    """
    文档加载错误。

    使用场景：
      - PDF 解析失败
      - 文件不存在
      - 文件格式不支持
    """

    pass


# RAG 相关（Phase 1 使用）
class RAGError(PrivateTeacherError):
    """
    RAG 流程错误。

    使用场景：
      - Embedding 失败
      - 向量库写入失败
      - 检索异常
    """

    pass


# Notebook 相关（Phase 3 使用）
class NotebookError(PrivateTeacherError):
    """
    Jupyter Notebook 处理错误。

    使用场景：
      - ipynb 格式损坏
      - 用户代码执行超时
      - 沙箱逃逸尝试
    """

    pass


# 批改相关（Phase 4 使用）
class GradingError(PrivateTeacherError):
    """
    批改流程错误。

    使用场景：
      - 题目无法解析
      - 评分异常
    """

    pass


class DuplicateDocumentError(PrivateTeacherError, FileExistsError):
    """重复上传同一个文件（sha256 已存在）。

    ⚠️ 注意这里的**多重继承**：
      同时继承 PrivateTeacherError 和内置的 FileExistsError。

    为什么这么做？
      - 继承 PrivateTeacherError：上层可以用 `except PrivateTeacherError` 统一兜住
      - 继承 FileExistsError：语义天然吻合，且调用方用
        `except FileExistsError` 也能捕获（对 UI 代码更直观）
      两个基类都直接/间接继承自 Exception，MRO 能正常解析，不会冲突。
    """

    pass


class ServiceError(PrivateTeacherError):
    """业务编排层的错误（参数非法、状态冲突等）。"""

    pass