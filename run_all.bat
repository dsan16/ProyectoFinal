@echo off
REM — Cambiar al directorio del frontend y arrancar Angular —
cd /d E:\Dani\ProyectoFinal\frontend
start "" cmd /k "ng serve --open"

REM — Cambiar al directorio del backend Python y arrancar Uvicorn —
cd /d E:\Dani\ProyectoFinal\Redes
start "" cmd /k "uvicorn api:app --reload --port 8000"

exit
