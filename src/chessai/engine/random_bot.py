from .base import Engine, Analysis
import time
import random as rd

class RandomEngine(Engine):

    name = "random"

    def __init__(self, seed):
        self.rng = rd.Random(seed)

    def analyse(self, board, limit = None):
        t0 = time.perf_counter()
        moves = list(board.legal_moves)
        if not moves:
            raise ValueError("no legal move available in this position")
        move = self.rng.choice(moves)
        return Analysis(move, [move], value = None, nbNodesVisited=1, elapsed= time.perf_counter()-t0 )
    