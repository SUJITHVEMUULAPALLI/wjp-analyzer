@echo off
setlocal
pushd "%~dp0"

rem Prefer the project's virtualenv Python and Streamlit if available
set "VENV_SCRIPTS=%~dp0.venv\Scripts"
set "PY_CMD=%VENV_SCRIPTS%\python.exe"
set "ST_CMD=%VENV_SCRIPTS%\streamlit.exe"

if exist "%PY_CMD%" (
  if exist "%ST_CMD%" (
    set "STREAMLIT_CMD=%ST_CMD%"
  ) else (
    set "STREAMLIT_CMD=%PY_CMD% -m streamlit"
  )
  goto :run
)

rem Fallback: try to find streamlit first, then python
for /f "delims=" %%S in ('where streamlit.exe 2^>nul') do (
  set "STREAMLIT_CMD=%%S"
  goto :run
)

for /f "delims=" %%P in ('where python 2^>nul') do (
  set "PY_CMD=%%P"
  goto :have_py
)
for /f "delims=" %%P in ('where py 2^>nul') do (
  set "PY_CMD=%%P"
  goto :have_py
)

echo Could not locate Python. Install Python 3.11+ or create .venv.
if "%NOPAUSE%"=="" (
  echo.
  echo Press any key to close this window...
  pause >nul
)
popd
exit /b 1

:have_py
rem Guard against MSYS/Git Bash pseudo paths like /usr/bin\python.exe
if "%PY_CMD:~0,1%"=="/" goto bad_py
if not exist "%PY_CMD%" (
  echo Resolved Python interpreter is invalid: %PY_CMD%
  echo Ensure you run this from Command Prompt (cmd.exe) or PowerShell, not Git Bash.
  if "%NOPAUSE%"=="" pause
  popd
  exit /b 1
)
set "STREAMLIT_CMD=%PY_CMD% -m streamlit"

:run
rem Environment hygiene for consistent behavior
set "STREAMLIT_BROWSER_GATHER_USAGE_STATS=0"
if not defined USE_STUB_IMAGES set "USE_STUB_IMAGES=1"
set "PYTHONUTF8=1"

rem Default host/port (can be overridden via %* args)
set "HOST=127.0.0.1"
set "PORT=8501"

rem Prefer new multipage UI; fallback to legacy path if missing
rem Use the legacy Streamlit UI integrated in src/
set "UI_SCRIPT=src\wjp_analyser\web\streamlit_app.py"

echo Using interpreter/streamlit: %STREAMLIT_CMD%
echo UI script: %UI_SCRIPT%
 %STREAMLIT_CMD% run "%UI_SCRIPT%" --server.address %HOST% --server.port %PORT% %*
set EXITCODE=%ERRORLEVEL%
if "%NOPAUSE%"=="" (
  echo.
  echo Press any key to close this window...
  pause >nul
)
popd
exit /b %EXITCODE%

:bad_py
echo Resolved Python interpreter is invalid: %PY_CMD%
echo Detected POSIX-style path. Please run this script from Command Prompt or PowerShell.
if "%NOPAUSE%"=="" pause
popd
exit /b 1
