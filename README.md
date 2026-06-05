# ChessAI - A bitboard-based chess engine with learned evaluation

A UCI-compatible chess engine designed to run on low-end hardware (dual-core 1.6GHz,
8GB RAM) and achieve approximately 2000+ Elo.

## Architecture

- **Chess representation**: Bitboards (`uint64_t`) with magic bitboard move generation
- **Search**: Iterative deepening alpha-beta with PVS, transposition table (256MB default),
  null move pruning, late move reductions, history heuristics and killer moves
- **Evaluation**: 391-dimension linear model trained via SGD regression from PGN game data
- **UCI protocol**: Compatible with any UCI chess GUI (Arena, CuteChess, etc.)

## Quick Start

### Prerequisites

- C++17 compiler (g++ 8+ or clang 7+)
- Python 3.8+ (for training only)

### Build

```bash
cd chess
make
```

This produces the `chessai` binary.

### Run

```bash
# UCI mode (for GUI)
./chessai

# Benchmark mode (search starting position at depth 8)
./chessai --bench

# Evaluate a single position
./chessai --eval "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"

# Custom TT size
./chessai --tt-size 128
```

### Playing via command line

The engine communicates via the UCI protocol. Use a UCI-compatible GUI such as:

- **Arena** (http://www.playwitharena.com)
- **CuteChess** (https://cutechess.com)
- **BanksiaGUI** (https://banksiagui.com)

Configure the GUI to use the `./chessai` binary as the engine.

## Training Evaluation Weights

The default evaluation weights are basic piece-square tables. To train better weights
from your own game data:

### 1. Install Python dependencies

```bash
pip install -r training/requirements.txt
```

### 2. Train from PGN

```bash
cd training
python train.py --pgn games.pgn --samples 200000 --output weights.txt
```

This will:
1. Read positions from `games.pgn` (supports .gz compressed files)
2. Extract 391 features per position
3. Train a linear regression model using SGD
4. Export weights as a C++ int16_t array to `weights.txt`

### 3. Apply the weights

Copy the generated weights into the engine:

```bash
cp training/weights.txt src/default_weights.inc
make clean && make
```

Or manually replace the contents of `src/default_weights.inc` with the output.

### Training tips

- Use high-quality PGNs (games between strong players, 2200+ Elo)
- 100,000-500,000 positions works well for training
- More positions = better, but diminishing returns after ~1 million
- Use blitz games for more positions, classical games for higher quality
- Remove engine vs engine games to avoid overfitting to engine style

## Performance

On the target hardware (1.6GHz dual-core i5, 8GB RAM):

- **NPS**: 300,000-500,000 nodes/second (single thread)
- **Search depth**: 8-10 ply in ~1 second, 12-14 ply in ~30 seconds
- **TT usage**: 256MB recommended (configure with --tt-size)

## Project Structure

```
chess/
├── src/
│   ├── board.h / board.cpp     - Bitboard representation, FEN, Zobrist hashing
│   ├── movegen.h / movegen.cpp - Magic bitboard move generation
│   ├── eval.h / eval.cpp       - Linear evaluation function (391 features)
│   ├── search.h / search.cpp   - Alpha-beta search with TT, LMR, null-move
│   ├── uci.h / uci.cpp         - UCI protocol interface
│   ├── main.cpp                - Entry point, CLI args
│   └── default_weights.inc     - Embeddable weight values
├── training/
│   ├── extract_features.py     - Feature extraction (matches C++ exactly)
│   ├── train.py                - SGD training from PGN
│   └── requirements.txt        - Python dependencies
├── Makefile                    - Build with `make`
└── README.md                   - This file
```

## Feature Design (391 dimensions)

| Index Range | Feature | Description |
|------------|---------|-------------|
| 0-63 | Pawn PSQT | Piece-square table for pawns |
| 64-127 | Knight PSQT | Piece-square table for knights |
| 128-191 | Bishop PSQT | Piece-square table for bishops |
| 192-255 | Rook PSQT | Piece-square table for rooks |
| 256-319 | Queen PSQT | Piece-square table for queens |
| 320-383 | King PSQT | Piece-square table for kings |
| 384 | Mobility | Pseudo-legal move count / 50 (capped 0-1) |
| 385 | W Isolated Pawns | White isolated pawn count |
| 386 | B Isolated Pawns | Black isolated pawn count |
| 387 | W Passed Pawns | White passed pawn count |
| 388 | B Passed Pawns | Black passed pawn count |
| 389 | W King Safety | White king pawn shield damage (0-4) |
| 390 | B King Safety | Black king pawn shield damage (0-4) |

## License

This project is provided as-is for educational purposes.
