

import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from chessai.model import ChessNet

DATA = Path(os.environ.get("CHESSAI_DATA", "C:/Users/Faure/chess-dataset/full"))
PACKED = Path(os.environ.get("CHESSAI_PACKED", DATA / "Xbits17.npy"))
RUNS = Path(os.environ.get("CHESSAI_RUNS", "runs/hard"))

N_PLANES = 17                          
BYTES_PER_POS = N_PLANES * 64 // 8     

BLOCKS = 1000                          
VAL_BLOCKS = 20                        
TEST_BLOCKS = 20                       

STEPS = int(os.environ.get("CHESSAI_STEPS", 60_000))
BATCH = int(os.environ.get("CHESSAI_BATCH", 1024))
LR = float(os.environ.get("CHESSAI_LR", 1e-3))
N_BLOCKS_NET = int(os.environ.get("CHESSAI_NBLOCKS", 6))
N_HIDDEN = int(os.environ.get("CHESSAI_NHIDDEN", 128))
WARMUP = 500
EVAL_EVERY = 500
EVAL_BATCH = 4096
CKPT_EVERY = 2000
SEED = 44


# X.npy (N, 18, 8, 8) -> (N, 136) without 18th layer
if not PACKED.exists():
    src = np.load(DATA / "X.npy", mmap_mode="r")
    print(f"compression de {DATA/'X.npy'} : {len(src)} positions, {src.nbytes/2**30:.1f} Go", flush=True)
    packed = np.lib.format.open_memmap(PACKED, mode="w+", dtype=np.uint8,
                                       shape=(len(src), BYTES_PER_POS))
    chunk = 50_000
    t0 = time.perf_counter()
    for start in range(0, len(src), chunk):
        stop = min(start + chunk, len(src))
        a = np.asarray(src[start:stop, :N_PLANES])
        packed[start:stop] = np.packbits(a.reshape(stop - start, -1), axis=1)
        if (start // chunk) % 40 == 0:
            print(f"  {stop/len(src)*100:5.1f} %  ({time.perf_counter()-t0:.0f} s)", flush=True)
    packed.flush()
    print(f"  termine en {time.perf_counter()-t0:.0f} s -> {PACKED} "
          f"({packed.nbytes/2**30:.1f} Go)", flush=True)
    del src, packed




if os.environ.get("CHESSAI_MMAP"):
    Xb = np.load(PACKED, mmap_mode="r")
else:
    print(f"chargement de {PACKED.name} en RAM ({PACKED.stat().st_size/2**30:.1f} Go)...", flush=True)
    t_load = time.perf_counter()
    Xb = np.load(PACKED)
    print(f"  charge en {time.perf_counter()-t_load:.0f} s", flush=True)

Y = np.load(DATA / "policy.npy")     
V = np.load(DATA / "value.npy")      
N = len(Xb)
assert len(Y) == N and len(V) == N, "les trois fichiers n'ont pas la meme longueur"



bounds = np.linspace(0, N, BLOCKS + 1).astype(np.int64)
rng = np.random.default_rng(SEED)
order = rng.permutation(BLOCKS)
val_b = order[:VAL_BLOCKS]
test_b = order[VAL_BLOCKS:VAL_BLOCKS + TEST_BLOCKS]
train_b = order[VAL_BLOCKS + TEST_BLOCKS:]


device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    print("ATTENTION : pas de CUDA, l'entrainement sera environ 100x plus lent.")


def get_batch(blocks, size):
    b = rng.choice(blocks, size)
    lo, hi = bounds[b], bounds[b + 1]
    ix = (lo + rng.random(size) * (hi - lo)).astype(np.int64)
    ix.sort()
    bits = np.unpackbits(Xb[ix], axis=1)         
    Xbatch = torch.from_numpy(bits).view(size, N_PLANES, 8, 8).float()
    Ybatch = torch.from_numpy(Y[ix].astype(np.int64))
    Vbatch = torch.from_numpy(V[ix].astype(np.float32))
    return Xbatch.to(device), Ybatch.to(device), Vbatch.to(device)


model = ChessNet(n_blocks=N_BLOCKS_NET, n_hidden=N_HIDDEN, n_in=N_PLANES,
                 value_head="logit").to(device)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
params = sum(p.numel() for p in model.parameters())
print(f"{N} positions | {len(train_b)} blocs train, {len(val_b)} val, {len(test_b)} test")
print(f"reseau {N_BLOCKS_NET} blocs x {N_HIDDEN} canaux, {params/1e6:.2f} M parametres, "
      f"device {device}, batch {BATCH}, {STEPS} pas")

RUNS.mkdir(parents=True, exist_ok=True)
model.train()
lossi = []
t0 = time.perf_counter()

for i in range(STEPS):

    
    
    lr = float(LR * (i + 1) / WARMUP if i < WARMUP else
               LR * 0.5 * (1 + np.cos(np.pi * (i - WARMUP) / max(1, STEPS - WARMUP))))
    for pg in opt.param_groups:
        pg["lr"] = lr

    # batching
    Xbatch, Ybatch, Vbatch = get_batch(train_b, BATCH)

    # forward pass
    logits, value = model(Xbatch)
    loss_p = F.cross_entropy(logits, Ybatch)
    # cible du value : 1 gain blanc, 0.5 nulle, 0 gain noir
    loss_v = F.binary_cross_entropy_with_logits(value.squeeze(-1), (Vbatch + 1) / 2)
    loss = loss_p + loss_v

    # backward pass
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # update
    opt.step()

    # tracking stats
    lossi.append(loss.item())
    if i % EVAL_EVERY == 0:
        model.eval()
        with torch.no_grad():
            Xval, Yval, Vval = get_batch(val_b, EVAL_BATCH)
            lv, vv = model(Xval)
            ce = F.cross_entropy(lv, Yval).item()
            bce = F.binary_cross_entropy_with_logits(vv.squeeze(-1), (Vval + 1) / 2).item()
        model.train()
        speed = BATCH * (i + 1) / (time.perf_counter() - t0)
        print(f"{i:7d}  train {loss.item():.3f} (p {loss_p.item():.3f} v {loss_v.item():.3f})"
              f"  | val ce {ce:.3f} bce {bce:.3f}  | {speed:.0f} pos/s", flush=True)

    if i % CKPT_EVERY == 0 and i > 0:
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": i,
                    "n_in": N_PLANES, "n_blocks": N_BLOCKS_NET, "n_hidden": N_HIDDEN,
                    "value_head": "logit", "seed": SEED, "blocks": BLOCKS,
                    "val_blocks": val_b.tolist()}, RUNS / f"ckpt_{i}.pt")

torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": STEPS,
            "n_in": N_PLANES, "n_blocks": N_BLOCKS_NET, "n_hidden": N_HIDDEN,
            "value_head": "logit", "seed": SEED, "blocks": BLOCKS,
            "val_blocks": val_b.tolist()}, RUNS / f"ckpt_{STEPS}.pt")
