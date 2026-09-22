@echo off
REM Moteur UCI pour lichess-bot et cutechess.
REM cd sur le dossier du .bat : CHESSAI_CKPT est relatif.
cd /d "%~dp0"
set CHESSAI_CKPT=parameters/ckpt_60000.pt
set CHESSAI_SIMS=400
"C:\Users\Faure\chess-ai\.venv\Scripts\python.exe" -m chessai.adapters.uci %*
