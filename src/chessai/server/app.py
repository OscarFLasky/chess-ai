from fastapi import FastAPI, Depends, HTTPException
from .schemas import AnalyseRequest, AnalyseResponse, MoveScore
from contextlib import asynccontextmanager
from ..engine.policy import PolicyEngine
from ..engine.base import Engine, Analysis
from .deps import get_engine
from fastapi.staticfiles import StaticFiles
import time
import chess

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.engine = PolicyEngine()
    except Exception as e:
        print(f"engine not loaded: {e}")
        app.state.engine = None
    yield
    app.state.engine = None



app = FastAPI(title="ChessEngine", version="0.1.0", lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": app.state.engine is not None}

@app.post("/moves/analyse", response_model=AnalyseResponse)
def analyse(req: AnalyseRequest, engine: Engine = Depends(get_engine)):
    board = chess.Board(req.fen)
    try:
        result = engine.analyse(board)
    except ValueError:
        raise HTTPException(status_code=409, detail="game over")
    bestmove = result.best_move.uci()
    bestsan = board.san(chess.Move.from_uci(bestmove))
    moves = [
    MoveScore(uci=m.uci(), san=board.san(m), prob=p)
    for m, p in result.list_moves[:req.top_k]
    ]
    value = result.value
    nodes = result.nbNodesVisited
    elapsed= result.elapsed
    return(AnalyseResponse(best_move = bestmove, best_san = bestsan,
                            moves = moves,value =  value, nodes = nodes, elapsed_ms = elapsed * 1000, engine = engine.name ))



app.mount("/", StaticFiles(directory="src/chessai/server/static", html=True), name="static")






    
