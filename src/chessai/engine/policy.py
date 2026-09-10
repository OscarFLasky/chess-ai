import torch
from .base import Bot
from ..encoding import board_to_tensor, move_to_index, index_to_move
from ..model import ChessNet
from .base import Engine, Analysis
import time


class PolicyEngine(Engine):

    name = "Policy"


    def __init__(self, ckpt_path="runs/ckpt.pt", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ChessNet().to(self.device)
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

    def analyse(self, board, limit = None):
        t0 = time.perf_counter()
        encoded = board_to_tensor(board).unsqueeze(0)
        encoded = encoded.float().to(self.device)
        with torch.no_grad():
            y, v = self.model(encoded)
        moves = list(board.legal_moves)
        idx = torch.tensor([move_to_index(m) for m in moves], device = y.device)
        probs = torch.softmax(y.view(-1)[idx], dim = -1)
        indexing = torch.argsort(probs, descending=True)
        scores = probs[indexing].tolist()
        listmoves = [(moves[indexing[i].item()], scores[i] ) for i in range(len(probs))]
        return Analysis(listmoves[0][0], listmoves, v.item(), 1, time.perf_counter()-t0  )
        
    






    
