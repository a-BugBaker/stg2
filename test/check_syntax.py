"""快速语法检查脚本"""
import sys
import py_compile
from pathlib import Path

files = [
    "dag_core.py",
    "dag_storage.py", 
    "dag_manager.py",
    "dag_event_generator.py",
    "closure_retrieval.py",
    "config.py",
    "immediate_update.py",
    "buffer_update.py",
    "memory_manager.py",
    "__init__.py",
]

stg_dir = Path(__file__).parent

errors = []
for f in files:
    path = stg_dir / f
    try:
        py_compile.compile(str(path), doraise=True)
        print(f"✓ {f}")
    except py_compile.PyCompileError as e:
        print(f"✗ {f}: {e}")
        errors.append(f)

if errors:
    print(f"\n{len(errors)} 个文件有语法错误")
    sys.exit(1)
else:
    print(f"\n所有 {len(files)} 个文件语法检查通过")
    sys.exit(0)
