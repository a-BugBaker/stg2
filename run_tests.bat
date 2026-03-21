@echo off
echo ============================================
echo STG DAG系统 - 端到端测试
echo ============================================

cd /d "D:\after_CSU\实验室\Kan\stg_reference_v2"

REM 激活 Z-py-3-10 环境
call conda activate Z-py-3-10

echo.
echo 1. 检查Python语法...
python -m py_compile stg/dag_core.py stg/dag_storage.py stg/dag_manager.py stg/dag_event_generator.py stg/closure_retrieval.py stg/config.py
if errorlevel 1 (
    echo 语法检查失败!
    pause
    exit /b 1
)
echo 语法检查通过!

echo.
echo 2. 运行端到端测试...
python stg/test_e2e.py

echo.
echo 测试完成
pause
