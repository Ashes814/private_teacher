"""持久化层。

这里做 re-export，让外部可以写：
    from private_teacher.storage import Database, Course, Document
而不必记住每个符号在哪个子模块里。

⚠️ 注意：course_repo / document_repo 故意**不**在这里 re-export 函数，
   因为两边都有 create / delete / get_by_id，扁平化后必然撞名。
   用的时候请显式导入模块：
       from private_teacher.storage import course_repo, document_repo
       course_repo.create(conn, name="...")
"""

from private_teacher.storage import course_repo, document_repo
from private_teacher.storage.db import Database, get_connection, init_db
from private_teacher.storage.models import Course, Document, now_utc

__all__ = [
    "Course",
    "Database",
    "Document",
    "course_repo",
    "document_repo",
    "get_connection",
    "init_db",
    "now_utc",
]