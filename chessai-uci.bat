@echo off
REM Moteur UCI pour lichess-bot et cutechess.
REM Lance l'adaptateur avec le venv du projet, celui qui contient torch.
set CHESSAI_CKPT=parameters/ckpt_60000.pt
set CHESSAI_SIMS=400
"C:\Users\Faure\info\chess-ai\.venv\Scripts\python.exe" -m chessai.adapters.uci
