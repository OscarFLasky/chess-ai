from fastapi import FastAPI, Depends, HTTPException, Request
from .schemas import AnalyseRequest, AnalyseResponse, MoveScore
from contextlib import asynccontextmanager
from ..engine.policy import PolicyEngine
from ..engine.mcts import MCTSEngine
from ..engine.base import Engine, Analysis, Limit
from .deps import get_engine
from .config import MAX_BODY_BYTES, MAX_CONCURRENT, MAX_NODES, MAX_TIME_S, RATE_LIMIT
from .security import BodySizeLimit, limiter
from fastapi.staticfiles import StaticFiles
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
import threading
import time
import chess

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.engine = MCTSEngine(PolicyEngine())
    except Exception as e:
        print(f"engine not loaded: {e}")
        app.state.engine = None
    yield
    app.state.engine = None



app = FastAPI(title="ChessEngine", version="0.1.0", lifespan=lifespan)

# Limitation de débit par IP : 429 au-delà de RATE_LIMIT.
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Corps de requête plafonnés avant d'être lus en entier.
app.add_middleware(BodySizeLimit, max_bytes=MAX_BODY_BYTES)

# Au plus MAX_CONCURRENT analyses en même temps sur le CPU/GPU.
analysis_slots = threading.BoundedSemaphore(MAX_CONCURRENT)


def build_limit(req: AnalyseRequest) -> Limit:
    # Le client peut demander moins que le plafond, jamais plus.
    nodes = MAX_NODES if req.nodes is None else min(req.nodes, MAX_NODES)
    time_s = MAX_TIME_S if req.time_ms is None else min(req.time_ms / 1000, MAX_TIME_S)
    return Limit(time=time_s, nbNodesMax=nodes)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": app.state.engine is not None}

@app.post("/moves/analyse", response_model=AnalyseResponse)
@limiter.limit(RATE_LIMIT)
def analyse(request: Request, req: AnalyseRequest, engine: Engine = Depends(get_engine)):
    board = chess.Board(req.fen)
    limit = build_limit(req)
    if not analysis_slots.acquire(timeout=MAX_TIME_S):
        raise HTTPException(status_code=503, detail="server busy")
    try:
        result = engine.analyse(board, limit)
    except ValueError:
        raise HTTPException(status_code=409, detail="game over")
    finally:
        analysis_slots.release()
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
