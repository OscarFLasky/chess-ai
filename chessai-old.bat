@echo off
REM Ancien reseau (18 plans, tete tanh), donnees 1900+.
REM Se place dans le dossier du script : CHESSAI_CKPT est relatif.
cd /d "%~dp0"

set CHESSAI_CKPT=parameters/ckpt_32000.pt
set CHESSAI_SIMS=400

REM Interpreteur : le venv du projet s il existe, sinon le python du PATH.
REM Pour un venv situe ailleurs, definir CHESSAI_PYTHON avant d appeler ce script.
if not defined CHESSAI_PYTHON (
    if exist ".venv\Scripts\python.exe" (
        set CHESSAI_PYTHON=.venv\Scripts\python.exe
    ) else (
        set CHESSAI_PYTHON=python
    )
)

"%CHESSAI_PYTHON%" -m chessai.adapters.uci %*
