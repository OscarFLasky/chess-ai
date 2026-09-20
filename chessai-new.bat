@echo off
REM Reseau entraine sur le jeu >2200 Elo (17 plans, tete logit)
set CHESSAI_CKPT=parameters/ckpt_60000.pt
set CHESSAI_SIMS=400
"C:\Users\Faure\info\chess-ai\.venv\Scripts\python.exe" -m chessai.adapters.uci
