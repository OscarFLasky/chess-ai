from dataclasses import dataclass
import chess

@dataclass
class Limit:
    time : float = None
    nbNodesMax: int = None
    maxDepth: int = None

@dataclass
class Analysis:
    best_move : chess.Move
    list_moves : list[tuple[chess.Move, float]]
    value : float
    nbNodesVisited: int
    elapsed: float




class Engine:

    name = "engine"

    def analyse(self, board, limit=None):
        raise NotImplementedError

    def play(self, board, limit=None):
        return self.analyse(board,limit).best_move