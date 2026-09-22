from pydantic import BaseModel, Field, field_validator
import chess

class AnalyseRequest(BaseModel):
    fen : str = Field(max_length = 100)
    top_k : int = Field(default = 5, ge = 1, le = 10)
    # Budget souhaité par le client. La route le borne par les plafonds de config.py.
    nodes : int | None = Field(default = None, ge = 1)
    time_ms : float | None = Field(default = None, gt = 0, allow_inf_nan = False)

    @field_validator("fen")
    @classmethod
    def fen_valide(cls, v:str) -> str:
        try:
            b = chess.Board(v)
        except ValueError:
            raise ValueError("fen invalide")
        if not b.is_valid():
            raise ValueError("position impossible")
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
