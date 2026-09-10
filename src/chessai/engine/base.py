from dataclasses import dataclass, field
import chess

@dataclass
class Analysis:
    board: chess.Board
    Limit: float

    


class Engine:

    def analyse(self, board, limit = None):
        raise NotImplementedError

    def play(self, board, limit = None):
        return self.analyse(board,limit).best_move