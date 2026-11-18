@echo off
title TruthGuard - Fake News Detection Server
echo ====================================================================
echo Starting TruthGuard Fake News Detection Server on Port 8000...
echo Open in any browser: http://localhost:8000
echo ====================================================================
cd /d "%~dp0"
python app.py
pause
