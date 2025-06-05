@echo off
REM — arranca Angular —
cd frontend
start "" cmd /k "ng serve --open"

REM — arranca Uvicorn —
cd ..
cd Redes
start "" cmd /k "uvicorn api:app --reload --port 8000"

exit
