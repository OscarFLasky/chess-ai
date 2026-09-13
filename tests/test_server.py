import chess
import pytest
from fastapi.testclient import TestClient

from chessai.engine.base import Analysis, Engine
from chessai.server.app import app
from chessai.server.deps import get_engine

START = chess.STARTING_FEN
MATE = "7k/5QK1/8/8/8/8/8/8 b - - 0 1"


class FirstLegalEngine(Engine):

    name = "first-legal"

    def analyse(self, board, limit=None):
        moves = list(board.legal_moves)
        if not moves:
            raise ValueError("no legal move in this position")
        n = len(moves)
        return Analysis(
            best_move=moves[0],
            list_moves=[(m, 1.0 / n) for m in moves],
            value=0.0,
            nbNodesVisited=1,
            elapsed=0.0,
        )


@pytest.fixture
def client():
    app.dependency_overrides[get_engine] = lambda: FirstLegalEngine()
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_analyse_startpos(client):
    r = client.post("/moves/analyse", json={"fen": START, "top_k": 3})
    assert r.status_code == 200
    body = r.json()
    assert len(body["moves"]) == 3
    assert body["best_move"] == body["moves"][0]["uci"]


def test_returned_move_is_legal(client):
    r = client.post("/moves/analyse", json={"fen": START})
    move = chess.Move.from_uci(r.json()["best_move"])
    assert move in chess.Board(START).legal_moves


def test_san_matches_uci(client):
    r = client.post("/moves/analyse", json={"fen": START})
    body = r.json()
    board = chess.Board(START)
    assert body["best_san"] == board.san(chess.Move.from_uci(body["best_move"]))


def test_invalid_fen(client):
    r = client.post("/moves/analyse", json={"fen": "pas un fen"})
    assert r.status_code == 422


def test_top_k_out_of_bounds(client):
    r = client.post("/moves/analyse", json={"fen": START, "top_k": 99})
    assert r.status_code == 422


def test_terminal_position(client):
    r = client.post("/moves/analyse", json={"fen": MATE})
    assert r.status_code == 409