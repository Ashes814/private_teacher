-- ============================================================
-- Private Teacher 数据库 Schema
-- ============================================================
-- 约定：
--   1. 所有建表语句都用 IF NOT EXISTS —— 重复执行不报错，方便测试反复 init
--   2. 时间统一存 ISO8601 字符串（如 "2026-07-29T10:30:00+00:00"）
--      不用 sqlite 的 TIMESTAMP 自动转换，因为 Python 3.12 已废弃默认的
--      datetime adapter，会打 DeprecationWarning。我们在 Python 侧手动转。
--   3. 主键用 TEXT（8 位 uuid 短 id），不用自增整数
--      —— 短 id 可以直接当目录名，且不暴露"你一共建了几门课"
-- ============================================================

-- ---------- 课程表 ----------
CREATE TABLE IF NOT EXISTS courses (
    id TEXT PRIMARY KEY, -- uuid4 前 8 位，如 "a1b2c3d4"
    name TEXT NOT NULL, -- 课程名，如 "机器学习导论"
    description TEXT NOT NULL DEFAULT '', -- 课程描述，允许空串但不允许 NULL
    created_at TEXT NOT NULL, -- ISO8601 UTC
    updated_at TEXT NOT NULL -- ISO8601 UTC
);


-- ---------- 文档表 ----------
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    course_id TEXT NOT NULL 
        REFERENCES courses(id) ON DELETE CASCADE, -- 删课程时自动删文档
    path TEXT NOT NULL, -- 文件落盘后的绝对/相对路径（str 形式）
    category TEXT NOT NULL
        CHECK (category IN ('main', 'auxiliary')), -- 只允许这两个值，DB 层兜底
    sha256 TEXT NOT NULL, -- 文件内容哈希，用于去重 + 增量索引
    size INTEGER NOT NULL, -- 文件字节数
    mtime TEXT NOT NULL, -- 文件修改时间（ISO8601）
    indexed_at TEXT, -- 建索引时间；NULL = 还没进向量库

    -- 关键约束：同一门课里，同样内容的文件只能存一份
    -- 注意是 (course_id, sha256) 组合唯一，不同课程可以各存一份同样的 PDF
    UNIQUE (course_id, sha256)
); 

-- ---------- 索引 ----------
-- 场景：几乎每次查询都是 "把某门课的文档列出来"，course_id 上建索引避免全表扫
CREATE INDEX IF NOT EXISTS idx_documents_course_id
    ON documents (course_id);

-- 场景：增量索引时要快速找出 "还没建索引的文档"
CREATE INDEX IF NOT EXISTS idx_documents_indexed_at
    ON documents (course_id, indexed_at);