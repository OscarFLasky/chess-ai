import math
from dataclasses import dataclass

import chess

from .search.base import Bot

DRAW = 0.5

class MatchResult:
    def __init__(self, bot_a, bot_b, games, wins_a, draws, wins_b):
        self.bot_a = bot_a
        self.bot_b = bot_b
        self.games = games
        self.wins_a = wins_a
        self.draws = draws
        self.wins_b = wins_b
    @property
    def scoreA(self):
        score = self.wins_a + self.draws*DRAW
        return score
    @property
    def scoreB(self):
        score = self.wins_b + self.draws*DRAW
        return score

    def __str__(self):
        return( f"{self.bot_a} vs {self.bot_b}: {self.scoreA} - {self.scoreB}")


def play_game(white, black, max_moves=400):
    b = chess.Board()
    moves= 0
    while moves < max_moves and b.outcome(claim_draw=True) is None:
        engine = white if b.turn == chess.WHITE else black
        moves +=1
        m = engine.play(b)
        if m not in b.legal_moves:
            raise ValueError(f"{engine.name} returned an illegal move: {m}")
        b.push(m)
    return b

def game_score(board, botiswhite):
    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        return DRAW
    else:
        return 1.0 if outcome.winner==botiswhite else 0.0


def play_match(botA, botB, nbmatchs, maxmoves=400):
    wins_A = 0
    wins_B= 0
    draws = 0
    bots = [botA,botB]
    for i in range(nbmatchs):
        AIsWhite = i%2 ==0
        white, black = (botA, botB) if AIsWhite else (botB, botA)
        b = play_game(white,black, maxmoves)
        score = game_score(b, AIsWhite)
        if score == 0.5:
            draws+= 1
        elif score ==1:
            wins_A+=1
        else:
            wins_B+=1

    return(MatchResult(botA.name, botB.name, nbmatchs, wins_A, draws, wins_B))
