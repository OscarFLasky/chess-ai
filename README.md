# Chess-AI

A chess engine built from scratch: a residual policy/value network trained on human
Lichess games, with no engine distillation at any point.

**Status: work in progress.** This repository currently contains a *policy-only* bot,
meaning it plays the move its network finds most likely without searching the game tree
at all. The next milestone is an MCTS agent (PUCT, AlphaZero-style) built on top of the
same network. The numbers below are a baseline, not a final result.

## Guiding principle: no distillation

Most open-source chess networks are trained on Stockfish evaluations. This project
deliberately avoids that. The network never sees an engine output, an engine evaluation,
or an engine-labelled position. It learns from two signals only:

- which move a human actually played in a given position (policy head)
- how the game eventually ended (value head)

Everything the engine knows, it inferred from human play.

## Results

Measured with `chessai.arena`, colours alternating to cancel the first-move advantage.

| Matchup | Games | Score |
|---|---|---|
| PolicyBot (step 32000) vs MaterialBot | 100 | 98.5% |
| PolicyBot (step 32000) vs RandomBot | 100 | 98.5% |
| PolicyBot (step 14000) vs MaterialBot | 100 | 84.5% |
| MaterialBot vs RandomBot | 100 | 92.5% |

Validation cross-entropy on the policy head: about **1.8** on held-out positions, down
from `ln(4672) = 8.45` at initialisation. That corresponds to a perplexity near 6, so the
network narrows a position down to roughly six plausible moves on average.

The jump from 84.5% to 98.5% came from 4000 extra steps at a reduced learning rate, for
a validation loss gain of less than 0.1. Cross-entropy and playing strength are only
loosely related: what costs games is the occasional blunder, not average accuracy.

**These percentages are not Elo.** They are scores against two deliberately weak
baselines. Calibration against a rating scale requires matches versus a limited-strength
reference engine, which is planned but not done.

## The three bots

**RandomBot** picks uniformly among legal moves. It is the performance floor and exists
mainly to validate the game loop.

**MaterialBot** plays mate in one when available, otherwise the capture with the highest
victim value (pawn 1, knight and bishop 3, rook 5, queen 9), breaking ties at random, and
falls back to a random move when nothing can be captured. It never looks one ply ahead,
so it hangs pieces constantly, but it punishes every undefended piece its opponent leaves.
That combination makes it a surprisingly informative sparring partner for a searchless
network.

**PolicyBot** encodes the position, runs a single forward pass, restricts the 4672 output
logits to the indices of the legal moves, and plays the argmax. One network evaluation
per move, no lookahead, roughly one millisecond on GPU.

## How PolicyBot actually plays, and why it loses

It plays by pattern, the way a human moves instantly in a blitz game without calculating.
Openings are sound, piece placement is reasonable, plans are coherent. Then it drops a
piece.

The typical failure is a capture into a defended square: rook takes bishop, bishop takes
a protected pawn. The reason is structural rather than a bug. In games between 1900-rated
players, a capture is almost always good, because the human calculated before playing it.
The network learns that "rook takes bishop" is a frequent, high-value pattern. It never
learns to check whether the square is defended, because that verification happens in the
player's head and leaves no trace in the move list.

A searchless policy cannot evaluate a position. It cannot ask "if I play this, what
happens next". It only imitates the move distribution of a ~1900 Lichess player as closely
as it can, and inherits both their strengths and a share of their mistakes without their
tactical check.

One informal game against the 1300-rated chess.com bot illustrates the pattern well. The
network built a winning position and went two pieces up, then dropped a knight to an
understandable tactical shot, then hung its queen outright to a piece attacking it from
across the board. The first mistake is the kind a club player makes. The second is not:
no human two pieces up leaves their queen en prise to a long-range attacker. It is the
signature of an agent that recognises good positions but never checks a single line.

This is exactly the gap MCTS closes. A few hundred simulations expand the offending
branch, play the recapture inside the tree, let the value head score the resulting
position as bad, and discard the move.

## Data

Source: the public Lichess database, `lichess_db_standard_rated_2016-02`, streamed
directly from the compressed archive so the multi-gigabyte file is never fully expanded
in memory.

Filters applied per game:

- both players rated at least **1900**
- base time control of at least **180 seconds**, which excludes bullet
- `Termination` equal to `Normal`, which excludes losses on time where the final position
  contradicts the result and would poison the value target
- a defined result

About **5.2%** of games survive these filters.

From each retained game, up to 25 positions are sampled at random, skipping the first 10
plies. Sampling rather than keeping every position covers far more distinct games for the
same dataset size, which matters because consecutive positions from one game are nearly
identical and produce heavily correlated gradients.

Final dataset: **3,593,754 positions**, stored as three aligned arrays (`uint8` board
tensors, `int16` move indices, `int8` results), split 80/10/10 by random permutation.

### Why 1900 and not 2400

A quality-versus-quantity trade-off. Raising the threshold improves the average move but
collapses the number of games available: the Lichess rating distribution is centred well
below 2000, so a 2400 filter would leave a dataset too small to train 1.8M parameters
without immediate overfitting.

The time control filter turned out to matter as much as the rating one. A 1900-rated
player in bullet plays considerably worse than a 1900-rated player in rapid, so cutting
short time controls raises data quality at a much lower cost in volume than raising the
Elo threshold would.

The target is not to imitate the strongest possible player anyway. Supervised learning
caps out around the level of the data; strength beyond that has to come from search.

## Board and move encoding

**Position**: an `(18, 8, 8)` tensor.

| Channels | Content |
|---|---|
| 0 to 5 | white pieces, one plane per type |
| 6 to 11 | black pieces |
| 12 | side to move |
| 13 to 16 | castling rights |
| 17 | halfmove clock (see known limitations) |

One-hot planes rather than a single plane of piece codes, so that the convolution is not
handed a meaningless ordering between piece types.

**Moves**: the AlphaZero scheme, `4672 = 64 x 73`. A move is indexed by its origin square
crossed with a move type: 56 queen-like moves (8 directions by 7 distances), 8 knight
jumps, and 9 underpromotions (3 pieces by 3 file offsets). Queen promotions share the
index of the corresponding pawn push, which is why decoding an index needs the board.

This layout is not arbitrary. It folds exactly onto the 8x8 grid, which lets the policy
head emit 73 channels and flatten them into the 4672 logits with no dense layer at all.

## Architecture

```
input (B, 18, 8, 8)
  conv 3x3 18 -> 128, batch norm, ReLU
  6 x residual block:
      conv 3x3 128 -> 128, batch norm, ReLU
      conv 3x3 128 -> 128, batch norm
      add input, ReLU
  policy head: conv 1x1 128 -> 73, permute, flatten -> (B, 4672) logits
  value head:  conv 1x1 128 -> 1, flatten -> 64 -> 256 -> 1, tanh -> (B, 1)
```

**1,819,907 parameters.** Almost all of them sit in the trunk, because the policy head
avoids the usual 9.6M-parameter dense projection by mapping channels directly onto move
types.

Training: AdamW, batch size 512, 29000 steps at `lr = 1e-3` followed by 4000 at `1e-4`.
Loss is `cross_entropy(policy) + mse(value)`, equally weighted. About 1100 positions per
second on an RTX 5060 Ti, roughly 55 minutes per epoch.

## What is handwritten and what comes from PyTorch

The point of the project was to build the network, not to assemble it.

**Written from scratch**, as `nn.Module` subclasses:

- `Linear`, which is really a convolution: it unfolds the input into 3x3 neighbourhoods
  and applies a single matrix product
- `unfoldX`, the im2col transform itself, implemented as nine shifted slices of a
  zero-padded board
- `BatchNorm2d`, including the running mean and variance buffers and the train/eval
  switch
- `ReLU`, `Tanh`, `LinearFlat`, `ResBlock`, `ChessNet`
- the training loop, the batching, the evaluation, the checkpointing
- the entire data pipeline and both encodings

**Taken from PyTorch**: autograd, `nn.Module` and `nn.Parameter` for parameter
registration, device movement and `state_dict` serialisation, `nn.ModuleList`, `AdamW`,
`F.cross_entropy`, `F.mse_loss`, `F.pad`, and `F.conv2d`.

That last one deserves a note. The handwritten `unfoldX` plus matrix product is
mathematically identical to `F.conv2d`, but it materialises a tensor nine times the size
of the activation at every layer, which the fused CUDA kernel does not. Switching the
forward pass to `F.conv2d` cut step time from 922 ms to 474 ms. The explicit
implementation is kept in the code as the reference, and the two are checked against each
other (see below).

## Correctness tests

Two of these properties fail silently if broken: a wrong encoding still trains, the loss
still goes down, and the bot is simply inexplicably bad. They are verified rather than
assumed.

**Move encoding round-trip.** Over thousands of positions reached by random play,
`index_to_move(move_to_index(m), board) == m` for every legal move, all indices fall in
`[0, 4672)`, and no two distinct legal moves in the same position ever collide on one
index.

**Flatten ordering.** A one-hot `(1, 73, 8, 8)` tensor with a single 1 at a known
(type, square) is pushed through the head's permute-and-flatten and must land at index
`73 * square + type`, matching `move_to_index` exactly.

**Convolution equivalence.** The handwritten unfold-and-matmul is checked against
`F.conv2d` on the same weights, agreeing to floating-point tolerance.

**Bot contract.** Bots return only legal moves and never mutate the board they receive.

**Overfit sanity check.** Before any long run, the network is trained to memorise 500
fixed positions. It reaches a training loss near 1e-4 while validation loss stays around
8.6, confirming both that the graph is wired correctly and that nothing leaks between
splits.

## Repository layout

```
chess-ai/
├── src/chessai/
│   ├── encoding.py        board <-> tensor, move <-> index
│   ├── data.py            PGN streaming, filtering, dataset construction
│   ├── model.py           layers, residual block, ChessNet
│   ├── train.py           training loop, evaluation, checkpointing
│   ├── arena.py           bot versus bot matches
│   └── search/
│       ├── base.py        common Bot interface
│       ├── random_bot.py
│       ├── material.py
│       └── policy.py      network argmax over legal moves
├── notebooks/             exploratory work, see note below
├── tests/
└── scripts/
```

### A note on the notebooks

The notebooks are kept on purpose. They contain earlier versions of the same components,
including the pre-`nn.Module` implementations where every layer carried its own
`parameters()` method, the handwritten convolution before it was replaced by `F.conv2d`,
and the diagnostic cells used to inspect activation and gradient statistics.

They document how the code got to its current state. They are **not** part of the working
package: nothing in `src/` imports them, and some cells contain stale duplicate class
definitions that would shadow the current ones if run.

## Installation

```bash
git clone https://github.com/laskyroin/chess-ai
cd chess-ai
pip install -e ".[dev]"
```

PyTorch is deliberately not declared as a dependency, so that a CUDA build is never
silently replaced by a CPU wheel. Install it separately, picking the right command for
your hardware from pytorch.org.

Trained weights are published under Releases rather than committed, since checkpoints are
large binaries. Download one into `runs/` and point the bot at it:

```python
from chessai.arena import play_match
from chessai.search.policy import PolicyBot
from chessai.search.material import MaterialBot

print(play_match(PolicyBot(ckpt_path="runs/ckpt_32000.pt"), MaterialBot(seed=0), 100))
```

Reproducing the dataset requires downloading a monthly archive from
`database.lichess.org` into `data/` and running the builder in `data.py`. Expect roughly
one hour of parsing.

## Known limitations

Listed because they are real and known, not because they are unavoidable.

**Train/dev leakage across games.** The split is by position, not by game. Since 25
positions come from each game, some of a game's positions land in training and others in
validation, so the validation loss is optimistic. Fixing this means storing a game id
alongside each position and splitting on that.

**The halfmove clock channel is dead.** `board_to_tensor` fills channel 17 with
`halfmove_clock / 100`, a float in `[0, 1]`. The dataset builder then casts the whole
tensor to `uint8` to keep it compact, and casting a float to an integer truncates, so
every value below 1.0 becomes exactly 0. The channel is therefore all zeros across the
entire dataset. The network sees 17 useful planes instead of 18 and has no notion of
progress toward the fifty-move rule. The fix is to store the raw counter, which fits in
`uint8` since it never exceeds 100, and divide at load time instead.

**No en passant plane and no repetition counter.** En passant is not representable in the
input at all, and the network cannot recognise a position it has already seen, so it has
no way to understand draws by repetition.

**Positions are always encoded from White's point of view.** Flipping the board so that
the side to move always plays "upward" would roughly halve what the network has to learn,
at the cost of a genuinely error-prone change to both encodings.

**Only 7% of games in the data are draws**, which is characteristic of 2016 Lichess where
short time controls dominated. The value head will therefore underestimate drawn
positions, and a future MCTS agent will tend to play for a win in objectively dead
positions.

**The value head is never evaluated.** No validation set was built for it, so its quality
is unmeasured. It does not affect PolicyBot, but it will matter a great deal for MCTS.

**PolicyBot is deterministic.** It always plays the argmax, so every game against a fixed
opponent follows the same lines and a losing line loses forever. A temperature parameter
would fix this and is also needed for MCTS.

**The dataset builder holds everything in RAM** and writes once at the end, so an
interruption loses all the work. Writing in blocks would make it resumable.

**Kernel size is read from a module-level global** inside `Linear`, while `ChessNet`
accepts it as a constructor argument. Constructing the network with a different kernel
size would build correctly sized weights and then convolve with 3x3 anyway.

**No Elo calibration.** Strength is only known relative to two weak baselines.

**No CI yet.** Tests exist but are not run automatically.

## Improvement leads

**Search is worth far more than anything else on this list.** A policy network without
lookahead plateaus somewhere around club level regardless of how well it is trained. MCTS
on top of the same weights is the single largest available gain.

**Training throughput.** The current loop runs at roughly 1100 positions per second,
which is low for a network this small. The levers, in order of expected return:
`bfloat16` autocast combined with `channels_last` memory format, which is where recent
GPUs actually use their tensor cores; replacing the handwritten `BatchNorm2d` with the
fused `nn.BatchNorm2d`, since thirteen normalisations per forward pass each cost five
separate kernels here; and `torch.compile`.

Note that hand-rolling the backward pass is *not* on that list. Autograd emits
essentially the same kernels a manual implementation would, so replacing `loss.backward()`
would cost a lot of error-prone work for no speedup. The gains are in how the forward
kernels are dispatched, not in how the gradient is computed.

**Initialisation.** Currently He scaling on every convolution plus a 0.1 factor on the
final policy layer, which puts the initial loss at the expected 8.45. With batch norm
after every convolution, initialisation matters much less than it would otherwise, so
this is a small lever. Zero-initialising the second batch norm of each residual block,
so that every block starts as the identity, is the variant most likely to help.

**Learning rate schedule.** The 84.5% to 98.5% jump came from a manual tenfold reduction
after the loss plateaued. A cosine schedule would capture that automatically and probably
do better.

**Capacity and data.** Training and validation losses stayed level with each other
throughout, so the network never overfit. A wider trunk, more months of games, or a
higher rating threshold once more data is available are all still on the table.

**Architecture.** A transformer over 64 tokens, one per square, is a competitive and
arguably better alternative to the convolutional trunk for chess. Only the trunk would
need replacing; encoding, heads, data and search all stay as they are.

## License

MIT.