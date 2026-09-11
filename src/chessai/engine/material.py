import chess
import random as rd
import time
from .base import Engine, Analysis

pvalues: dict[chess.PieceType, int] = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}


def ismate(board, move):
    board.push(move)
    r = board.is_checkmate()
    board.pop()
    return r

def move_gain(board, move):
    gain = 0
    if ismate(board,move):
        gain += 100
    if board.is_en_passant(move):
        gain += pvalues[chess.PAWN]

    else:
        taken = board.piece_at(move.to_square)
        if taken is not None:
            gain += pvalues[taken.piece_type]
        if move.promotion is not None:
            gain+= pvalues[move.promotion] - pvalues[chess.PAWN]
    return gain

class MaterialEngine(Engine):

    name = "material"

    def __init__(self, seed):
        self.rng= rd.Random(seed)

    def analyse(self, board, limit = None):
        t0 = time.perf_counter()
        moves = list(board.legal_moves) 
        if not moves:
            raise ValueError("no legal move available in this position")
        best = max(move_gain(board,m) for m in moves)
        best_moves = [m for m in moves if move_gain(board, m)==best]
        return Analysis(self.rng.choice(best_moves), best_moves, value = None, nbNodesVisited= 1, elapsed= time.perf_counter()-t0)

