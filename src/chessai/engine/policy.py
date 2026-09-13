import torch
from ..encoding import board_to_tensor, move_to_index, index_to_move
from ..model import ChessNet
from .base import Engine, Analysis
import time


class PolicyEngine(Engine):

    name = "Policy"


    def __init__(self, ckpt_path="runs/ckpt_32000.pt", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ChessNet().to(self.device)
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

    def evaluate(self, boards):
        encoded = torch.stack([board_to_tensor(b) for b in boards]).float().to(self.device)
        with torch.no_grad():
            y, v = self.model(encoded)

        legal = [list(b.legal_moves) for b in boards]
        rows = torch.tensor([i for i, moves in enumerate(legal) for _ in moves],
                            dtype=torch.long, device=y.device)
        cols = torch.tensor([move_to_index(m) for moves in legal for m in moves],
                            dtype=torch.long, device=y.device)
        
        masked = torch.full_like(y, float("-inf"))
        masked[rows, cols] = y[rows, cols]
        # un seul transfert GPU -> CPU par tenseur, au lieu d'un .item() par coup
        probs = torch.softmax(masked, dim=-1)[rows, cols].tolist()
        values = v.view(-1).tolist()

        out = []
        start = 0
        for moves, value in zip(legal, values):
            out.append((moves, probs[start:start + len(moves)], value))
            start += len(moves)
        return out

    def analyse(self, board, limit = None):
        t0 = time.perf_counter()
        moves, probs, value = self.evaluate([board])[0]
        if not moves:
            raise ValueError("no legal move in this position")
        listmoves = sorted(zip(moves, probs), key=lambda kv: kv[1], reverse=True)
        return Analysis(listmoves[0][0], listmoves, value, 1, time.perf_counter()-t0  )
