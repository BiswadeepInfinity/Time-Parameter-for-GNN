@echo off
setlocal enabledelayedexpansion

REM ============================================================================
REM RT-GNN Full Empirical Benchmark Runner (5-Seed TGB Training)
REM ============================================================================

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

set "PYTHON_EXE="
if exist "C:\Users\madhu\AppData\Local\Python\pythoncore-3.14-64\python.exe" (
    set "PYTHON_EXE=C:\Users\madhu\AppData\Local\Python\pythoncore-3.14-64\python.exe"
) else (
    set "PYTHON_EXE=python"
)

echo ============================================================================
echo   Starting RT-GNN Full Benchmark on tgbl-wiki (5 Seeds, 5 Epochs each)
echo   This will train both Stateless and Stateful models and measure live metrics.
echo   High CPU/GPU load and fan activity is expected during execution.
echo ============================================================================
echo.

"%PYTHON_EXE%" run_benchmark.py --seeds 0 1 2 3 4 --epochs 5 --models stateless stateful %*

echo.
echo ============================================================================
echo   Benchmark Complete! Updating Test Suite and Report...
echo ============================================================================
"%PYTHON_EXE%" tests\run_and_report.py

echo.
echo Press any key to exit...
pause >nul
