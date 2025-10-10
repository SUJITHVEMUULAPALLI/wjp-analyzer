@echo off
REM WJP Analyzer - Guided Batch Interface Launcher
REM ==============================================

echo 📦 **WJP ANALYZER - GUIDED BATCH INTERFACE**
echo ============================================
echo.
echo Starting Guided Batch Interface on port 8505...
echo 🌐 URL: http://localhost:8505
echo.
echo Features:
echo ✅ Intelligent batch processing
echo ✅ Smart optimization suggestions
echo ✅ Comprehensive analysis and reporting
echo.
echo Press Ctrl+C to stop the server
echo.

python run_web_ui.py --batch-guided

pause
