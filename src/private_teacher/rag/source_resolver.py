"""
源文件溯源
==========

输入：一个检索到的 chunk
输出：它来自哪个文件、第几页、第几行，以及能不能真的打开

【三种精度】
  行级（代码/txt/md）：line_start ~ line_end，最精确，可以高亮具体几行
  页级（PDF/PPTX）  ：page，能跳到某一页
  文件级            ：只能定位到文件（兜底）

【必须处理的现实问题】
  文件可能已经被用户移动、改名、删除。
  这时不能抛异常让整个页面崩掉 —— 要优雅降级成"文件已移除"提示，
  并且仍然把 chunk 的内容展示出来（内容存在向量库里，不依赖源文件）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document
from loguru import logger

from private_teacher.config import PathSettings


@dataclass(slots=True)
class SourceLocation:
    """一个 chunk 在源文件中的位置。"""

    file_path: Path
    source_name: str
    file_type: str
    page: int | None
    total_pages: int | None
    line_start: int | None
    line_end: int | None
    snippet: str  # chunk 内容预览
    exists: bool  # 源文件当前是否还在磁盘上

    # ---------- 展示辅助 ----------
    @property
    def has_lines(self) -> bool:
        return self.line_start is not None

    @property
    def has_pages(self) -> bool:
        return self.page is not None and (self.total_pages or 1) > 1

    def label(self) -> str:
        """一行式位置描述，UI 的标题栏用。

        例："机器学习.pdf · 第 12 页" / "train.py · 第 30-55 行"
        """
        parts = [self.source_name]

        if self.has_lines:
            if self.line_end and self.line_end != self.line_start:
                parts.append(f"第 {self.line_start}-{self.line_end} 行")
            else:
                parts.append(f"第 {self.line_start} 行")

        elif self.has_pages:
            parts.append(f"第 {self.page}/{self.total_pages} 页")

        if not self.exists:
            parts.append("⚠️ 文件已移除")

        return " · ".join(parts)

    def uri(self) -> str:
        """生成可点击的定位链接。

        - PDF：file:///path/to/a.pdf#page=12
          （Chrome / Edge / Preview 都支持 #page= 锚点直接跳页）
        - 其他：file:///path/to/a.py
          （编辑器链接如 vscode://file/path:line 依赖用户装了 VSCode，
            不作为默认行为，留给 UI 层自己决定）
        """
        if not self.exists:
            return ""

        # as_uri() 会正确处理空格、中文、Windows 盘符等，
        # 手写 "file://" + str(path) 在这些情况下都会出错
        base = self.file_path.as_uri()

        if self.file_type == "pdf" and self.page:
            return f"{base}#page={self.page}"
        return base


class SourceResolver:
    """把 chunk 映射回源文件位置。"""

    def __init__(self, course_id: str, paths: PathSettings) -> None:
        self.course_id = course_id
        self.paths = paths

    # ============================================================
    # 解析
    # ============================================================
    def resolve(self, chunk: Document, snippet_chars: int = 200) -> SourceLocation:
        """从 chunk 的 metadata 还原位置信息。

        Args:
            chunk: 检索结果
            snippet_chars: 内容预览截断长度

        Returns:
            SourceLocation（即使文件已丢失也会返回，用 exists 标记）
        """
        meta = chunk.metadata

        raw_path = str(meta.get("source_path", ""))
        file_path = Path(raw_path) if raw_path else Path()

        # ---------- 容错查找 ----------
        # 文件可能因为项目目录整体搬家而找不到，
        # 试着在当前课件目录下按文件名再找一次
        exists = bool(raw_path) and file_path.exists()
        if not exists and raw_path:
            recovered = self._try_recover(file_path)
            if recovered is not None:
                file_path = recovered
                exists = True

        content = chunk.page_content
        snippet = (
            content if len(content) <= snippet_chars else content[:snippet_chars].rstrip() + "..."
        )

        return SourceLocation(
            file_path=file_path,
            source_name=str(meta.get("source_name") or file_path.name or "未知文件"),
            file_type=str(meta.get("file_type", "")),
            # _as_int：metadata 从 Chroma 取回来可能是 str，统一转 int
            page=_as_int(meta.get("page")),
            total_pages=_as_int(meta.get("total_pages")),
            line_start=_as_int(meta.get("line_start")),
            line_end=_as_int(meta.get("line_end")),
            snippet=snippet,
            exists=exists,
        )

    def resolve_many(self, chunks: list[Document]) -> list[SourceLocation]:
        """批量解析。"""
        return [self.resolve(c) for c in chunks]

    def _try_recover(self, missing: Path) -> Path | None:
        """按文件名在本课程的课件目录里重新找一遍。

        场景：用户把整个项目文件夹从 D:\\ 挪到了 E:\\，
        绝对路径全失效，但文件其实都还在。
        """
        course_dir = self.paths.course_dir(self.course_id)
        if not course_dir.exists() or not missing.name:
            return None

        # next(iterator, None)：取第一个匹配，没有就返回 None
        found = next(course_dir.rglob(missing.name), None)
        if found is not None:
            logger.debug(f"源文件路径已失效，按文件名找回：{found}")
        return found

    # ============================================================
    # 读取源文件片段
    # ============================================================
    def read_snippet(
        self,
        location: SourceLocation,
        context_lines: int = 3,
    ) -> str:
        """读源文件对应位置的内容（含上下文）。

        Args:
            location: resolve() 的结果
            context_lines: 上下各多读几行

        Returns:
            源文件真实内容；文件不可读时退回 location.snippet

        为什么要"重新读源文件"，而不是直接用 chunk 内容？
          - chunk 是切分后的片段，边界可能在句子中间
          - 从源文件读能给出完整上下文，用户体验好得多
          - 还能验证"向量库里的内容和源文件是否还一致"
        """

        # 没有行号信息（PDF/PPTX）→ 只能用 chunk 内容
        if not location.exists or not location.has_lines:
            return location.snippet

        try:
            from private_teacher.loaders.text_loader import read_text_smart

            lines = read_text_smart(location.file_path).splitlines()
        except Exception as exc:
            logger.warning(f"读取源文件失败 {location.file_path}: {exc}")
            return location.snippet

        # ---------- 行号 → 下标 ----------
        # 行号从 1 开始，列表下标从 0 开始，所以要 -1
        start = max(0, (location.line_start or 1) - 1 - context_lines)
        end = min(len(lines), (location.line_end or location.line_start or 1) + context_lines)

        return "\n".join(lines[start:end])

    def read_numbered_snippet(
        self,
        location: SourceLocation,
        context_lines: int = 3,
    ) -> str:
        """带行号的片段，命中行用 ▶ 标出。

        输出示例：
             28 | def train():
             29 |     model = build()
          ▶  30 |     model.fit(X, y)
          ▶  31 |     return model
             32 |
        """

        if not location.exists or not location.has_lines:
            return location.snippet

        try:
            from private_teacher.loaders.text_loader import read_text_smart

            lines = read_text_smart(location.file_path).splitlines()
        except Exception:
            return location.snippet

        start = max(0, (location.line_start or 1) - 1 - context_lines)
        end = min(
            len(lines),
            (location.line_end or location.line_start or 1) + context_lines,
        )

        out: list[str] = []
        for idx in range(start, end):
            line_no = idx + 1
            hit = location.line_start is not None and location.line_start <= line_no <= (
                location.line_end or location.line_start
            )
            marker = "▶" if hit else " "
            # >4 是右对齐到 4 位宽，让行号列整齐
            out.append(f"{marker} {line_no:>4} | {lines[idx]}")

        return "\n".join(out)

    # ============================================================
    # 分组
    # ============================================================
    def group_by_file(
        self,
        chunks: list[Document],
    ) -> dict[str, list[SourceLocation]]:
        """按源文件分组。

        用途：检索结果按文件折叠展示，
        "机器学习.pdf（3 处匹配）" 比平铺 3 条更容易读。
        """
        grouped: dict[str, list[SourceLocation]] = {}
        for loc in self.resolve_many(chunks):
            grouped.setdefault(loc.source_name, []).append(loc)

        # 每组内部按页码/行号排序，符合阅读顺序
        for locations in grouped.values():
            locations.sort(key=lambda loc: (loc.page or 0, loc.line_start or 0))

        return grouped


def _as_int(value: object) -> int | None:
    """安全地把 metadata 里的值转成 int。

    为什么需要这个？
      Chroma 存取过程中，数字类型可能变成 str（尤其是从旧版本库读出来时）。
      直接 int(value) 遇到 None 或 "" 会抛异常，
      而 metadata 缺字段是完全正常的情况（PDF 就没有行号）。
    """
    if value is None or value == "":
        return None
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
