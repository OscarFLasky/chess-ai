from pydantic import BaseModel, Field, field_validator
import chess

class AnalyseRequest(BaseModel):
    fen : str = Field(max_length = 100)
    top_k : int = Field(default = 5, ge = 1, le = 10)
    temperature : float = Field(default = 0.0, ge=0.0, le = 2.0)

    @field_validator("fen")
    @classmethod
    def fen_valide(cls, v:str) -> bool:
        try:
            b = chess.Board(v)
        except:
            raise ValueError("fen invalide")
        return v


class MoveScore(BaseModel):
    uci: str
    san: str
    prob: float


class AnalyseResponse(BaseModel):
    best_move: str
    best_san: str
    moves: list[MoveScore]
    value: float
    nodes: int
    elapsed_ms: float
    engine: str