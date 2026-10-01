@echo off
REM ====================================
REM  image2 出图工具 Web 界面启动脚本 (Windows)
REM  复用 gen.py 的出图逻辑，浏览器访问 http://127.0.0.1:5001
REM ====================================
setlocal

REM 切换到脚本所在目录，保证 web.py 里的 `import gen` 与相对路径正常
cd /d "%~dp0"

REM 端口：默认 5001，可用第一个参数覆盖，例如 start.bat 6001
set "PORT=%~1"
if "%PORT%"=="" set "PORT=5001"

echo ====================================
echo   image2 Web 界面 (GPT Image 2 / 2.5)
echo ====================================
echo.
echo 启动后请用浏览器访问:  http://127.0.0.1:%PORT%
echo 按 Ctrl+C 可以停止服务
echo.

REM 优先使用 uv（项目用 pyproject.toml / uv.lock 管理依赖），否则回退到 python
where uv >nul 2>nul
if %errorlevel%==0 (
    echo [运行方式] uv run python web.py --port %PORT%
    echo.
    uv run python web.py --port %PORT%
) else (
    echo [运行方式] 未检测到 uv，回退到 python web.py --port %PORT%
    echo.
    python web.py --port %PORT%
)

echo.
echo 服务已退出。
pause
endlocal
