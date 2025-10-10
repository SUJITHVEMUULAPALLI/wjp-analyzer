@echo off
REM OpenAI Evaluation Runner for Waterjet Analyzer
REM Windows batch script for easy evaluation execution

echo OpenAI API Evaluation for Waterjet Analyzer
echo ==========================================

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Check if API key is set
if "%OPENAI_API_KEY%"=="" (
    echo Warning: OPENAI_API_KEY environment variable not set
    echo Please set it with: set OPENAI_API_KEY=your-api-key-here
    echo.
)

REM Create evaluation directory if it doesn't exist
if not exist "evaluation_results" mkdir evaluation_results

echo.
echo Available evaluation options:
echo 1. Generate test files
echo 2. Run simple evaluation
echo 3. Run comprehensive evaluation
echo 4. Run all evaluations
echo.

set /p choice="Enter your choice (1-4): "

if "%choice%"=="1" (
    echo Generating test DXF files...
    python evaluation/test_generator.py
    echo Test files generated successfully!
) else if "%choice%"=="2" (
    echo Running simple evaluation...
    python evaluation/simple_evaluation.py
) else if "%choice%"=="3" (
    echo Running comprehensive evaluation...
    python evaluation/run_evaluation.py
) else if "%choice%"=="4" (
    echo Running complete evaluation suite...
    echo.
    echo Step 1: Generating test files...
    python evaluation/test_generator.py
    echo.
    echo Step 2: Running simple evaluation...
    python evaluation/simple_evaluation.py
    echo.
    echo Step 3: Running comprehensive evaluation...
    python evaluation/run_evaluation.py
    echo.
    echo All evaluations completed!
) else (
    echo Invalid choice. Please run the script again.
)

echo.
echo Evaluation results saved to: evaluation_results/
echo.
pause

