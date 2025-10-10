@echo off
REM WJP Analyzer - Guided Individual Interface Launcher
REM ==================================================

echo 🎯 **WJP ANALYZER - GUIDED INDIVIDUAL INTERFACE**
echo ================================================
echo.
echo Starting Guided Individual Interface on port 8504...
echo 🌐 URL: http://localhost:8504
echo.
echo Features:
echo ✅ Step-by-step guidance for individual projects
echo ✅ From prompt to PDF report
echo ✅ Intelligent tips and warnings
echo.
echo Press Ctrl+C to stop the server
echo.

python run_web_ui.py --guided

pause
