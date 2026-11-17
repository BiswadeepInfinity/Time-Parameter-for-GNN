@echo off
REM Delegate to tests\run_tests.bat
call "%~dp0tests\run_tests.bat" %*
exit /b %ERRORLEVEL%
