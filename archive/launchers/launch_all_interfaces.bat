@echo off
REM WJP Analyzer - All Interfaces Launcher
REM ======================================

echo 🚀 **WJP ANALYZER - ALL INTERFACES**
echo ====================================
echo.
echo Starting All Interfaces...
echo.
echo 🌐 Main Interface: http://localhost:8501
echo 🎯 Guided Individual: http://localhost:8504
echo 📦 Guided Batch: http://localhost:8505
echo.
echo Features:
echo ✅ Complete WJP analysis tools
echo ✅ Step-by-step guidance for individual projects
echo ✅ Intelligent batch processing
echo ✅ Smart optimization suggestions
echo.
echo Press Ctrl+C to stop all servers
echo.

python run_web_ui.py --all-interfaces

pause
