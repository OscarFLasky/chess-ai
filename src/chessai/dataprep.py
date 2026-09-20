"""
dataprep.py - version lib+CLI de data.py, parametree pour la data prep
distribuee sur les postes des salles info.

Utilisation CLI :
    python -m chessai.dataprep INPUT OUT_DIR --min-elo 2200 --min-base-sec 600

Utilisation lib :
    from chessai.dataprep import build_dataset
    build_dataset("lichess.pgn.zst", "out/", min_elo=2200, min_base_sec=600)

Design :
- Aucun side-effect a l'import (contrairement a data.py qui lance un run au
  chargement du module). Ce fichier est importable sans risque.
- Encodage canonique : board_to_tensor et move_to_index viennent de
  chessai.encoding. Toute evolution de l'encodage est automatiquement repercutee.
- Filtre elo et cadence en parametres, pas de globales a monkey-patcher.

Perf :
- Lecture bas niveau : on tokenise le PGN en (headers dict, texte brut du game)
  sans passer par chess.pgn.read_game. Le filter (elo, cadence, termination,
  result) est applique sur les headers seuls.
- chess.pgn.read_game (cher) n'est appele que sur les games qui passent le
  filter. Sur Lichess 2024+ 2200/600, c'est ~0.1% des games -> parser saute
  ~99.9% du travail vs approche naive.
"""
import argparse
import io
import time
from itertools import islice
from pathlib import Path

import numpy as np
import chess
import chess.pgn
import zstandard

from chessai.encoding import board_to_tensor, move_to_index


RESULT_TO_VALUE = {"1-0": 1, "0-1": -1, "1/2-1/2": 0}


def _open_pgn_stream(path):
    """Ouvre un .pgn ou .pgn.zst comme flux texte."""
    path = Path(path)
    if str(path).endswith(".zst"):
        fh = open(path, "rb")
        reader = zstandard.ZstdDecompressor(max_window_size=2**31).stream_reader(fh)
        return io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
    return open(path, encoding="utf-8", errors="replace")


def _extract_header(line, headers):
    """Parse une ligne '[Key "Value"]' et stocke dans headers dict."""
    if not line.startswith("["):
        return
    end = line.find("]")
    if end == -1:
        return
    body = line[1:end]
    q1 = body.find('"')
    q2 = body.rfind('"')
    if q1 <= 0 or q2 <= q1:
        return
    key = body[:q1].strip()
    if key:
        headers[key] = body[q1 + 1:q2]


def iter_game_texts(path):
    """
    Yield (headers: dict, game_text: str) pour chaque game d'un PGN.
    Ne fait AUCUN parse de coups : seulement segmentation texte + headers.
    Beaucoup plus rapide que chess.pgn.read_game quand on filtre au niveau headers.
    """
    stream = _open_pgn_stream(path)
    with stream:
        buffer_lines = []
        headers = {}
        # states : "start" (avant tout header), "headers", "moves"
        state = "start"
        for line in stream:
            if state == "start":
                if line.startswith("["):
                    state = "headers"
                    _extract_header(line, headers)
                    buffer_lines.append(line)
                # sinon : preamble, ignore
            elif state == "headers":
                if line.startswith("["):
                    _extract_header(line, headers)
                    buffer_lines.append(line)
                elif line.strip() == "":
                    state = "moves"
                    buffer_lines.append(line)
                else:
                    # pas de ligne blanche entre headers et moves : tolerer
                    state = "moves"
                    buffer_lines.append(line)
            else:  # state == "moves"
                if line.strip() == "":
                    buffer_lines.append(line)
                    yield headers, "".join(buffer_lines)
                    buffer_lines = []
                    headers = {}
                    state = "start"
                elif line.startswith("["):
                    # pas de blank line entre games : emettre courant, demarrer nouveau
                    yield headers, "".join(buffer_lines)
                    buffer_lines = [line]
                    headers = {}
                    _extract_header(line, headers)
                    state = "headers"
                else:
                    buffer_lines.append(line)
        # dernier game (pas de trailing blank line)
        if buffer_lines and headers:
            yield headers, "".join(buffer_lines)


def iter_games(path):
    """
    Yield chess.pgn.Game pour chaque game. Fait un full-parse de tous les games.
    Conserve pour compatibilite / usage lib externe. Pour la data prep de masse,
    prefere iter_game_texts + keep_headers + read_game selectif.
    """
    stream = _open_pgn_stream(path)
    with stream:
        while True:
            g = chess.pgn.read_game(stream)
            if g is None:
                return
            yield g


def get_elo(headers, key):
    v = headers.get(key, "")
    return int(v) if v.isdigit() else None


def get_base_seconds(headers):
    tc = headers.get("TimeControl", "")
    if tc in ("", "-", "?"):
        return None
    base = tc.split("+")[0]
    return int(base) if base.isdigit() else None


def keep_headers(headers, min_elo, min_base_sec):
    """
    Filter sur headers dict (pas de Game object requis).
    Meme semantique que keep_game(game.headers, ...).
    """
    we = get_elo(headers, "WhiteElo")
    be = get_elo(headers, "BlackElo")
    if we is None or be is None or min(we, be) < min_elo:
        return False
    base = get_base_seconds(headers)
    if base is None or base < min_base_sec:
        return False
    if headers.get("Termination") != "Normal":
        return False
    return headers.get("Result") in RESULT_TO_VALUE


def keep_game(game, min_elo, min_base_sec):
    """Filter sur un Game (backward compat)."""
    return keep_headers(game.headers, min_elo, min_base_sec)


def extract_positions(game, rng, skip_opening=10, samples_per_game=25):
    value = RESULT_TO_VALUE.get(game.headers.get("Result"))
    if value is None:
        return
    moves = list(game.mainline_moves())
    n = len(moves)
    if n <= skip_opening:
        return
    candidates = np.arange(skip_opening, n)
    if len(candidates) > samples_per_game:
        candidates = rng.choice(candidates, samples_per_game, replace=False)
    wanted = {int(i) for i in candidates}
    board = game.board()
    for i, mv in enumerate(moves):
        if i in wanted:
            tens = board_to_tensor(board).numpy().astype(np.uint8)
            yield tens, move_to_index(mv), value
        board.push(mv)


def build_dataset(
    input_path,
    out_dir,
    min_elo=2200,
    min_base_sec=600,
    max_games=None,
    seed=0,
    skip_opening=10,
    samples_per_game=25,
    progress_every=10000,
):
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    xs, ps, vs = [], [], []
    kept = 0
    total = 0
    t0 = time.time()

    game_texts = iter_game_texts(input_path)
    if max_games is not None:
        game_texts = islice(game_texts, max_games)

    for total, (headers, text) in enumerate(game_texts, start=1):
        if keep_headers(headers, min_elo, min_base_sec):
            # full-parse uniquement pour les kept
            game = chess.pgn.read_game(io.StringIO(text))
            if game is not None:
                kept += 1
                for tens, idx, v in extract_positions(game, rng, skip_opening, samples_per_game):
                    xs.append(tens)
                    ps.append(idx)
                    vs.append(v)
        if total % progress_every == 0:
            dt = time.time() - t0
            print(
                f"[{total} lues, {kept} gardees, {len(xs)} positions] "
                f"{total/dt:.0f} parties/s",
                flush=True,
            )

    dt = time.time() - t0
    print(
        f"FIN : {total} lues, {kept} gardees, {len(xs)} positions en "
        f"{dt:.1f}s ({total/max(dt, 1e-9):.0f} parties/s)",
        flush=True,
    )

    X = np.stack(xs) if xs else np.zeros((0, 18, 8, 8), np.uint8)
    np.save(out_dir / "X.npy", X)
    np.save(out_dir / "policy.npy", np.array(ps, dtype=np.int16))
    np.save(out_dir / "value.npy", np.array(vs, dtype=np.int8))
    print(f"Ecrit dans {out_dir}/", flush=True)


def main():
    ap = argparse.ArgumentParser(
        description="Generation d'un shard de dataset chess-ai depuis un PGN Lichess."
    )
    ap.add_argument("input", help="Fichier PGN (.pgn ou .pgn.zst)")
    ap.add_argument("out_dir", help="Repertoire de sortie (X.npy, policy.npy, value.npy)")
    ap.add_argument("--min-elo", type=int, default=2200)
    ap.add_argument("--min-base-sec", type=int, default=600,
                    help="Rapid+classical (>= 10 min), exclut le blitz")
    ap.add_argument("--max-games", type=int, default=None,
                    help="Arret apres N parties LUES (avant filtrage), pour tests")
    ap.add_argument("--skip-opening", type=int, default=10)
    ap.add_argument("--samples-per-game", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--progress-every", type=int, default=10000)
    args = ap.parse_args()

    build_dataset(
        args.input, args.out_dir,
        min_elo=args.min_elo, min_base_sec=args.min_base_sec,
        max_games=args.max_games, seed=args.seed,
        skip_opening=args.skip_opening, samples_per_game=args.samples_per_game,
        progress_every=args.progress_every,
    )


if __name__ == "__main__":
    main()
