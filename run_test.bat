@echo off
REM AUTO-ROTO Test Runner - Windows Batch Wrapper
REM
REM Usage:
REM   run_test.bat                    - Run with defaults (SAM + refinement, skip combine)
REM   run_test.bat --sam-only         - Run SAM2 only (fastest)
REM   run_test.bat --full             - Run full pipeline
REM   run_test.bat --help             - Show all options

setlocal

REM Set environment variable to disable torch.compile (Windows compatibility)
set TORCHDYNAMO_DISABLE=1

REM Activate conda and run the test script
call conda activate autoroto
if errorlevel 1 (
    echo ERROR: Failed to activate conda environment 'autoroto'
    echo Make sure you have created the environment with: conda create -n autoroto python=3.10
    pause
    exit /b 1
)

python "%~dp0run_test.py" %*

if errorlevel 1 (
    echo.
    echo Pipeline finished with errors.
) else (
    echo.
    echo Pipeline completed successfully!
)

endlocal
