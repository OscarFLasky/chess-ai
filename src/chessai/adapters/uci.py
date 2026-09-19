

import math
import os
import sys

import chess

from ..engine.base import Limit
from ..engine.mcts import MCTSEngine
from ..engine.policy import PolicyEngine

NAME = "ChessAI"
AUTHOR = "laskyroin"

DEFAULT_CKPT = os.environ.get("CHESSAI_CKPT", "parameters/ckpt_32000.pt")
DEFAULT_SIMS = int(os.environ.get("CHESSAI_SIMS", "200"))
DEFAULT_KIND = os.environ.get("CHESSAI_ENGINE", "mcts")

MAX_SIMS = 100_000


def out(line):
    print(line, flush=True)


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def build_engine():
    policy = PolicyEngine(ckpt_path=DEFAULT_CKPT)
    if DEFAULT_KIND == "policy":
        return policy
    return MCTSEngine(policy, nbSims=DEFAULT_SIMS)


def value_to_cp(value):
    value = max(-0.999, min(0.999, value))
    return int(111.7 * math.tan(1.5620688 * value))


def parse_position(board, tokens):
    if not tokens:
        return board

    if "moves" in tokens:
        split = tokens.index("moves")
        head, moves = tokens[:split], tokens[split + 1:]
    else:
        head, moves = tokens, []

    if head and head[0] == "startpos":
        board = chess.Board()
    elif head and head[0] == "fen":
        board = chess.Board(" ".join(head[1:]))
    else:
        board = chess.Board()

    for uci in moves:
        try:
            board.push_uci(uci)
        except ValueError:
            log(f"coup ignoré, illégal ou mal formé : {uci}")
    return board


def parse_go(tokens, board):
    args = {}
    i = 0
    while i < len(tokens):
        key = tokens[i]
        if key in ("infinite", "ponder"):
            args[key] = True
            i += 1
        elif i + 1 < len(tokens):
            try:
                args[key] = int(tokens[i + 1])
            except ValueError:
                pass
            i += 2
        else:
            i += 1

    if "movetime" in args:
        return Limit(time=args["movetime"] / 1000.0, nbNodesMax=MAX_SIMS)

    if "nodes" in args:
        return Limit(nbNodesMax=args["nodes"])

    if "infinite" in args:
        return Limit(nbNodesMax=MAX_SIMS, time=10.0)

    remaining = args.get("wtime" if board.turn == chess.WHITE else "btime")
    if remaining is not None:
        inc = args.get("winc" if board.turn == chess.WHITE else "binc", 0)
        budget = remaining / 30.0 + inc * 0.8
        budget = max(0.05, budget / 1000.0)
        return Limit(time=budget, nbNodesMax=MAX_SIMS)

    return Limit()


def handle_go(engine, board, tokens):
    limit = parse_go(tokens, board)

    try:
        result = engine.analyse(board, limit)
    except ValueError:
        out("bestmove 0000")
        return

    out(
        f"info depth 1 nodes {result.nbNodesVisited} "
        f"time {int(result.elapsed * 1000)} "
        f"score cp {value_to_cp(result.value)} "
        f"pv {result.best_move.uci()}"
    )
    out(f"bestmove {result.best_move.uci()}")


def main():
    engine = None
    board = chess.Board()

    for raw in sys.stdin:
        tokens = raw.split()
        if not tokens:
            continue
        cmd, rest = tokens[0], tokens[1:]

        if cmd == "uci":
            out(f"id name {NAME}")
            out(f"id author {AUTHOR}")
            out("uciok")

        elif cmd == "isready":
            if engine is None:
                engine = build_engine()
                log(f"moteur chargé : {engine.name}")
            out("readyok")

        elif cmd == "ucinewgame":
            board = chess.Board()

        elif cmd == "position":
            board = parse_position(board, rest)

        elif cmd == "go":
            if engine is None:
                engine = build_engine()
            handle_go(engine, board, rest)

        elif cmd == "stop":
            pass

        elif cmd == "quit":
            break

       


if __name__ == "__main__":
    main()