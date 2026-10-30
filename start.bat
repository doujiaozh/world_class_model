@echo off
chcp 65001 > nul
cd /d "%~dp0"

echo ===============================================================
echo   world_class_model v2.0 - 混合超级引擎
echo   本地极速路由 + 本地微模型 + 云端超级大脑(DeepSeek API)
echo ===============================================================
echo.

where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] 系统未找到 python，请先安装 Python 3.10+ 并加入 PATH
    pause
    exit /b 1
)

python -c "import numpy" >nul 2>&1
if %errorlevel% neq 0 (
    echo [*] 正在安装依赖 numpy ...
    pip install -r requirements.txt
)

if not exist "artifacts\wcm_micro_int8.npz" (
    echo [!] 本地微模型权重缺失，将通过云端大脑直接回答
    echo     (或运行 python train.py --config micro 从零训练)
    echo.
)

if "%DEEPSEEK_API_KEY%"=="" (
    echo [i] 未检测到 DEEPSEEK_API_KEY，将使用本地引擎兜底
    echo     配置方法: set DEEPSEEK_API_KEY=你的Key
    echo.
)

if "%1"=="--demo" (
    python chat.py --demo
) else (
    python chat.py %*
)
pause
