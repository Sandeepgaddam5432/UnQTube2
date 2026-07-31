@echo off
set CURRENT_DIR=%CD%
echo ***** Current directory: %CURRENT_DIR% *****
set PYTHONPATH=%CURRENT_DIR%

rem set HF_ENDPOINT=https://hf-mirror.com
rem webui\App.py runs webui\Main.py with the Apple glassmorphism theme applied.
streamlit run .\webui\App.py --browser.gatherUsageStats=False --server.enableCORS=True
