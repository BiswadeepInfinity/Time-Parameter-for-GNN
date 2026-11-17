@echo off
setlocal enabledelayedexpansion

REM ============================================================================
REM RT-GNN Test Suite & Reviewer Audit Runner
REM ============================================================================

set "SCRIPT_DIR=%~dp0"
if exist "%SCRIPT_DIR%..\rtgnn_data.py" (
    cd /d "%SCRIPT_DIR%.."
) else (
    cd /d "%SCRIPT_DIR%"
)

REM Find working Python executable
set "PYTHON_EXE="

if exist "C:\Users\madhu\AppData\Local\Python\pythoncore-3.14-64\python.exe" (
    set "PYTHON_EXE=C:\Users\madhu\AppData\Local\Python\pythoncore-3.14-64\python.exe"
) else (
    python -V >nul 2>nul
    if !ERRORLEVEL! equ 0 (
        set "PYTHON_EXE=python"
    ) else if exist "C:\Python312\python.exe" (
        set "PYTHON_EXE=C:\Python312\python.exe"
    ) else if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" (
        set "PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    ) else (
        echo [ERROR] Could not find a functional Python interpreter.
        echo Please ensure Python is installed and accessible in your PATH.
        goto :END
    )
)

if "%~1"=="" (
    "%PYTHON_EXE%" tests\run_and_report.py
) else (
    echo [RUNNING] Executing specific target: %*
    "%PYTHON_EXE%" -m unittest %*
)

set "TEST_EXIT_CODE=%ERRORLEVEL%"

:END
REM Keep window open if launched by double-clicking from Windows Explorer
echo %cmdcmdline% | findstr /i /c:"%~f0" >nul
if %ERRORLEVEL% equ 0 (
    echo.
    echo Press any key to exit...
    pause >nul
) else (
    echo %cmdcmdline% | findstr /i /c:"/c" >nul
    if %ERRORLEVEL% equ 0 (
        echo.
        echo Press any key to exit...
        pause >nul
    )
)

exit /b %TEST_EXIT_CODE%
