from dataclasses import dataclass, field
import chess

@dataclass
class Analysis:
    best_move: chess.Move
    list_moves: list[tuple[chess.Move, float]]
    value: float
    nbNodesVisited: int
    elapsed: float
    
@dataclass
class Limit:
    time: float | None = None
    nbNodesMax: int | None = None
    maxDepth: int | None = None

class Engine:

    name = "Engine"

    def analyse(self, board, limit = None):
        raise NotImplementedError

    def play(self, board, limit = None):
        return self.analyse(board,limit).best_move


