@echo off
setlocal
pushd "%~dp0"
python run_web_ui.py %*
set EXITCODE=%ERRORLEVEL%
if "%NOPAUSE%"=="" (
  echo.
  echo Press any key to close this window...
  pause >nul
)
popd
exit /b %EXITCODE%
