from .base import Engine, Analysis
from .policy import PolicyEngine
import time
import math
import chess
import numpy as np

class Node:

    def __init__(self):
        self.tested = False
        self.pending = False
        self.children = {}
        self.N = {}
        self.W = {}
        self.P = {}

    def bestmove(self):
        if self.P == {}:
            raise ValueError("pas de coups possibles")
        else:
            return max(self.P, key = self.P.get)

    def update(self, move, value):
        self.N[move] +=1
        self.W[move] += value

    def add_virtual_loss(self, move):
        self.N[move] += 1
        self.W[move] -= 0.3

    def revert_virtual_loss(self, move):
        self.N[move] -= 1
        self.W[move] += 0.3

    def q(self, move):
        n = self.N[move]
        return self.W[move]/n if n>0 else 0.0

    def selectFunction(self, c_puct, move):
        if move not in self.P:
            raise ValueError
        else:
            return self.q(move) + c_puct *  self.P[move] * math.sqrt(sum(self.N.values()))/(1+self.N[move])

    def select(self, c_puct):
        if self.P == {}:
            raise ValueError
        else:
            dic = {move: self.selectFunction(c_puct, move) for move in self.P.keys()}
            movetoexpl = max(dic, key = dic.get)
            return movetoexpl



class MCTSEngine(Engine):

    name = "MCTSEngine"

    def __init__(self, policy, nbSims = 400, c_puct = 1.5, eps = 0.25, alpha = 0.3, noise = False, batch_size = 32):
        self.policy = policy
        self.nbSims = nbSims
        self.c_puct = c_puct
        self.eps = eps
        self.alpha = alpha
        self.noise = noise
        self.batch_size = batch_size

    def expand(self, node, moves, probs):
        node.P = dict(zip(moves, probs))
        node.N = {m: 0 for m in node.P}
        node.W = {m: 0.0 for m in node.P}
        node.tested = True

    def backup(self, path, value):
        for node, move in reversed(path):
            value = -value
            node.revert_virtual_loss(move)
            node.update(move, value)

    def analyse(self, board, limit=None):

        if not any(board.legal_moves):
            raise ValueError
        #time
        t0 = time.perf_counter()
        board = board.copy()
        #initializing node
        root = Node()
        #expanding node + Dirichlet noise
        moves, probs, _ = self.policy.evaluate([board])[0]
        self.expand(root, moves, probs)
        if self.noise:
            moves = list(root.P)
            eta = np.random.dirichlet([self.alpha] * len(moves))
            for m, e in zip(moves, eta):
                root.P[m] = (1 - self.eps) * root.P[m] + self.eps * e
        #limit calcul
        budget = self.nbSims
        if limit is not None and limit.nbNodesMax is not None:
            budget = limit.nbNodesMax
        deadline = None
        if limit is not None and limit.time is not None:
            deadline = t0 + limit.time

        sims = 0
        while sims < budget:
            if deadline is not None and time.perf_counter() > deadline:
                break
            pending = []
            for i in range(min(self.batch_size, budget - sims)):
                actualNode = root
                path = []
                while actualNode.tested and not board.is_game_over():
                    move = actualNode.select(self.c_puct)
                    path.append((actualNode, move))
                    actualNode.add_virtual_loss(move) # applies penalty to force variation in moves
                    board.push(move)
                    if move not in actualNode.children:
                        actualNode.children[move]= Node()
                    actualNode = actualNode.children[move]

                if board.is_game_over():
                    outcome= board.outcome()
                    value = 0.0 if outcome.winner is None else -1.0
                    self.backup(path, value)
                    sims += 1
                elif actualNode.pending:
                    # feuille déjà en attente dans ce batch : on annule cette descente
                    for node, move in path:
                        node.revert_virtual_loss(move)
                else:
                    actualNode.pending = True
                    pending.append((actualNode, path, board.copy(stack=False)))

                for _ in path:
                    board.pop()

            if pending:
                results = self.policy.evaluate([b for _, _, b in pending])
                for (node, path, b), (moves, probs, value) in zip(pending, results):
                    self.expand(node, moves, probs)
                    node.pending = False
                    self.backup(path, value if b.turn == chess.WHITE else -value)
                sims += len(pending)

        total = sum(root.N.values())
        list_moves = sorted(((m, n / total) for m, n in root.N.items()),key=lambda kv: kv[1], reverse=True,)
        value = root.q(list_moves[0][0])
        best = list_moves[0][0]
        elapsed = time.perf_counter() - t0
        return Analysis(best_move = best, list_moves= list_moves, nbNodesVisited= total, value = value, elapsed= elapsed)
