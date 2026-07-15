"""
允许 `python -m private_teacher` 直接运行 CLI。

原理：Python 解释器执行 `python -m <package>` 时会：
  1. 找到包所在目录
  2. 执行该目录下的 __main__.py
所以这个文件就是 CLI 入口。
"""

# 从 cli 模块导入 main 函数（命令行主逻辑）
from private_teacher.cli import main

# raise SystemExit(main()) 让 exit code 正确传递给 shell
# 例如：python -m private_teacher hello-llm 出错时，shell 里 $? 会是 1
raise SystemExit(main())
