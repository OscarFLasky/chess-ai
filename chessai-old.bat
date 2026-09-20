@echo off
REM Ancien reseau (18 plans, tete tanh)
set CHESSAI_CKPT=runs/ckpt_32000.pt
set CHESSAI_SIMS=400
"C:\Users\Faure\info\chess-ai\.venv\Scripts\python.exe" -m chessai.adapters.uci
