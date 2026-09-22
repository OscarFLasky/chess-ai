@echo off
REM Ancien reseau (18 plans, tete tanh), donnees 1900+.
REM cd sur le dossier du .bat : CHESSAI_CKPT est relatif.
cd /d "%~dp0"
set CHESSAI_CKPT=parameters/ckpt_32000.pt
set CHESSAI_SIMS=400
"C:\Users\Faure\chess-ai\.venv\Scripts\python.exe" -m chessai.adapters.uci %*
