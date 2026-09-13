from fastapi import FastAPI
from typing import Annotated
from fastapi import Path
from pydantic import BaseModel, Field, field_validator
import chess
import random



app = FastAPI(title = "test API", version = "0.1.0")

@app.get("/")
def racine():
    return {"status":"ok"}

@app.get("/version")
def version():
    return {"version": app.version}

class DemandeCoup(BaseModel):
    fen: str = Field(min_length=10, description="Position au format FEN")
    profondeur : int = Field(default = 1, ge = 1, le = 20)
    temperature: float = Field(default = 0.0, ge = 0.0, le = 2.0)
    top_k : int = Field(default = 1 , ge = 1, le = 10)

    @field_validator("fen")
    @classmethod
    def fen_valide(cls, v: str) -> str:
        v = v.strip()
        try:
            board = chess.Board(v)
        except:
            raise ValueError("fen invalide")
        if board.is_game_over():
            raise ValueError("Partie terminée")

        return v
    


@app.post("/coup")
def meilleur_coup(demande: DemandeCoup):
    board = chess.Board(demande.fen)
    coup = random.choice(list(board.legal_moves))
    return { "coup": coup.uci(), "san": board.san(coup)}

DEPART = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
APRES_E4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
FINALE = "8/8/8/4k3/8/8/4P3/4K3 w - - 0 1"
MAT_DU_BERGER = "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4"