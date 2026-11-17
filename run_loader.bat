@echo off
setlocal enabledelayedexpansion

REM Determine project root
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

REM Find working Python executable
set "PYTHON_EXE="

python -V >nul 2>nul
if %ERRORLEVEL% equ 0 (
    set "PYTHON_EXE=python"
) else (
    if exist "C:\Python312\python.exe" (
        set "PYTHON_EXE=C:\Python312\python.exe"
    ) else if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" (
        set "PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    ) else (
        echo [ERROR] Could not find Python executable.
        pause
        exit /b 1
    )
)

echo Executing data_loader.py with %PYTHON_EXE%...
"%PYTHON_EXE%" data_loader.py %*

if %ERRORLEVEL% neq 0 (
    echo.
    echo Execution finished with error code %ERRORLEVEL%.
)

echo %cmdcmdline% | findstr /i /c:"/c" >nul
if %ERRORLEVEL% equ 0 (
    echo.
    echo Press any key to exit...
    pause >nul
)
