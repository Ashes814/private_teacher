"""业务编排层。

分层原则回顾（从下到上）：
    storage / loaders / rag   基础能力，互不知道对方的业务含义
    services                  编排它们，保证跨模块状态一致
    ui                        只调 services，不碰下面任何一层

判断代码该放哪一层的简单方法：
  - 只操作数据库          → repo
  - 只处理文件/向量        → loaders / rag
  - 需要同时协调两个以上   → services
"""

from private_teacher.services.course_service import CourseService, CourseStats
from private_teacher.services.kb_service import KBService, SearchHit

__all__ = ["CourseService", "CourseStats", "KBService", "SearchHit"]
