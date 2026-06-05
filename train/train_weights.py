#!/usr/bin/env python3
"""
Chess AI evaluation weight training using Stockfish (Texel's tuning).

Usage:
  1. Install dependencies: pip install chess scikit-learn numpy
  2. Set STOCKFISH_PATH below to your Stockfish binary
  3. Generate training data:  python train_weights.py --generate 10000
  4. Train weights:          python train_weights.py --train
  5. Export to C++:          python train_weights.py --export

The training process:
  1. Generates positions from random games (or loads from PGN)
  2. Evaluates each position with Stockfish at depth 16
  3. Extracts feature vectors matching the C++ eval engine
  4. Trains a linear model using SGD to match Stockfish evaluations
  5. Outputs weights in C .inc file format
"""

import chess
import chess.pgn
import chess.engine
import numpy as np
from sklearn import linear_model
import argparse
import os
import sys
import random
import json
import time
import subprocess

# ============================================================
# Configuration - adjust these paths
# ============================================================
STOCKFISH_PATH = "/usr/local/bin/stockfish"  # Adjust as needed
NUM_FEATURES = 391
PSQT_SIZE = 384  # 6 piece types * 64 squares
ENGINE_DEPTH = 16  # Stockfish evaluation depth

# ============================================================
# Feature extraction (mirrors eval.cpp logic)
# ============================================================

# Piece encoding (matches engine::PieceType)
PT_PAWN = 0
PT_KNIGHT = 1
PT_BISHOP = 2
PT_ROOK = 3
PT_QUEEN = 4
PT_KING = 5

PIECE_TO_PT = {
    chess.PAWN: PT_PAWN,
    chess.KNIGHT: PT_KNIGHT,
    chess.BISHOP: PT_BISHOP,
    chess.ROOK: PT_ROOK,
    chess.QUEEN: PT_QUEEN,
    chess.KING: PT_KING,
}

def sq_to_index(sq, color):
    """
    Convert chess square (0-63, A1=0, H8=63) to engine square format.
    Engine format: A1=0, B1=1, ..., H1=7, A2=8, ..., H8=63
    chess square: A1=0, B1=1, ..., H1=7, A2=8, ..., H8=63  (same!).
    For black pieces, mirror vertically: sq ^ 56
    """
    if color == chess.WHITE:
        return sq
    else:
        return sq ^ 56

def extract_features(board):
    """Extract feature vector matching engine::extract_features."""
    features = np.zeros(NUM_FEATURES, dtype=np.float32)

    # PSQT features (0-383): +1 for white, -1 for black (mirrored)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            pt = PIECE_TO_PT[piece.piece_type]
            idx = pt * 64 + sq_to_index(square, piece.color)
            if piece.color == chess.WHITE:
                features[idx] += 1.0
            else:
                features[idx] -= 1.0

    # Mobility (384): number of legal moves / 50, clamped to [0, 1]
    mobility = min(board.legal_moves.count() / 50.0, 1.0)
    features[384] = mobility

    # Pawn structure
    w_pawns = board.pieces(chess.PAWN, chess.WHITE)
    b_pawns = board.pieces(chess.PAWN, chess.BLACK)

    features[385] = float(count_isolated_pawns(w_pawns, board))
    features[386] = float(count_isolated_pawns(b_pawns, board))
    features[387] = float(count_passed_pawns(w_pawns, b_pawns, chess.WHITE, board))
    features[388] = float(count_passed_pawns(b_pawns, w_pawns, chess.BLACK, board))

    # King safety (389-390)
    features[389] = float(king_safety_damage(board, chess.WHITE))
    features[390] = float(king_safety_damage(board, chess.BLACK))

    return features


def count_isolated_pawns(pawns, board):
    """Count isolated pawns (no friendly pawns on adjacent files)."""
    count = 0
    for sq in pawns:
        f = chess.square_file(sq)
        isolated = True
        if f > 0:
            mask = chess.BB_FILES[f - 1]
            if board.pieces(chess.PAWN, board.turn if board.piece_at(sq).color == chess.WHITE else not board.turn) & mask:
                # Need to check specifically friendly pawns on adjacent file
                pass
        # Simpler approach:
        file_bb = chess.BB_FILES[f]
        left_file = chess.BB_FILES[f - 1] if f > 0 else 0
        right_file = chess.BB_FILES[f + 1] if f < 7 else 0

        friendly_pawns = pawns
        if left_file and (friendly_pawns & left_file):
            continue
        if right_file and (friendly_pawns & right_file):
            continue
        count += 1
    return count


def count_passed_pawns(pawns, enemy_pawns, color, board):
    """Count passed pawns (no enemy pawns ahead on same/adjacent files)."""
    count = 0
    for sq in pawns:
        f = chess.square_file(sq)
        r = chess.square_rank(sq)

        passed = True
        for df in (-1, 0, 1):
            nf = f + df
            if nf < 0 or nf > 7:
                continue
            file_mask = chess.BB_FILES[nf]
            if color == chess.WHITE:
                # Squares ahead (higher ranks)
                ahead_mask = file_mask & ~((1 << (sq + 1)) - 1)
            else:
                # Squares ahead (lower ranks)
                ahead_mask = file_mask & ((1 << sq) - 1)
            if enemy_pawns & ahead_mask:
                passed = False
                break
        if passed:
            count += 1
    return count


def king_shield_mask(color, king_sq):
    """Get pawn shield mask for king (squares in front of king on 3 files)."""
    r = chess.square_rank(king_sq)
    f = chess.square_file(king_sq)
    mask = 0
    for df in (-1, 0, 1):
        nf = f + df
        if nf < 0 or nf > 7:
            continue
        for dr in (1, 2) if color == chess.WHITE else (-1, -2):
            nr = r + dr
            if nr < 0 or nr > 7:
                continue
            mask |= 1 << chess.square(nf, nr)
    return mask


def king_safety_damage(board, color):
    """Count missing shield pawns (0-4, higher = worse)."""
    king = board.king(color)
    if king is None:
        return 4
    shield = king_shield_mask(color, king)
    our_pawns = board.pieces(chess.PAWN, color)
    shield_pawns = shield & our_pawns
    missing = 6 - bin(shield_pawns).count('1')
    return min(missing, 4)


# ============================================================
# Stockfish evaluation
# ============================================================

def evaluate_with_stockfish(board, engine, depth=ENGINE_DEPTH):
    """Evaluate a board position using Stockfish.
    Returns score in centipawns from white's perspective (positive = white better).
    """
    result = engine.analyse(board, chess.engine.Limit(depth=depth))
    score = result["score"].relative
    if score.is_mate():
        mate_in = score.mate()
        # Convert mate score to large centipawn value
        if mate_in > 0:
            return 30000 - mate_in * 50
        else:
            return -30000 + abs(mate_in) * 50
    return score.score()  # centipawns from current side's perspective


# ============================================================
# Data generation
# ============================================================

def generate_random_position(max_moves=80):
    """Generate a random legal chess position by playing random moves."""
    board = chess.Board()
    num_moves = random.randint(10, max_moves)
    for _ in range(num_moves):
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            break
        move = random.choice(legal_moves)
        board.push(move)
        if board.is_game_over():
            break
    return board


def generate_positions_from_games(pgn_file, max_positions=10000):
    """Extract middle-game positions from a PGN file."""
    positions = []
    with open(pgn_file) as f:
        while len(positions) < max_positions:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            board = game.board()
            ply_count = 0
            for move in game.mainline_moves():
                board.push(move)
                ply_count += 1
                # Take positions from move 8 to 60 (middle game)
                if 8 <= ply_count <= 60 and ply_count % 4 == 0:
                    if not board.is_game_over():
                        positions.append(board.copy())
                        if len(positions) >= max_positions:
                            break
    return positions


def generate_training_data(num_positions, engine, output_file):
    """Generate training data: feature vectors + Stockfish evaluation."""
    print(f"Generating {num_positions} training positions...")

    X = np.zeros((num_positions, NUM_FEATURES), dtype=np.float32)
    y = np.zeros(num_positions, dtype=np.int32)

    for i in range(num_positions):
        if i % 100 == 0:
            print(f"  Position {i}/{num_positions}")

        # Generate a random position
        board = generate_random_position()

        # Extract features
        features = extract_features(board)
        X[i] = features

        # Evaluate with Stockfish
        try:
            score = evaluate_with_stockfish(board, engine)
            y[i] = score
        except Exception as e:
            print(f"  Error evaluating position {i}: {e}")
            y[i] = 0

    # Save to disk
    np.savez(output_file, X=X, y=y)
    print(f"Saved training data to {output_file}")
    return X, y


# ============================================================
# Training
# ============================================================

def train_weights(X, y, output_weights_file=None):
    """Train linear evaluation weights using SGD regression."""
    print(f"Training on {X.shape[0]} positions...")

    # Scale scores to reasonable range for SGD
    y_clipped = np.clip(y, -5000, 5000)

    # Train SGD regressor
    model = linear_model.SGDRegressor(
        max_iter=1000,
        tol=1e-4,
        penalty='l2',
        alpha=0.0001,
        learning_rate='invscaling',
        eta0=0.01,
        random_state=42,
    )
    model.fit(X, y_clipped)

    weights = model.coef_
    bias = model.intercept_[0]

    r2 = model.score(X, y_clipped)
    print(f"Training complete. R² score: {r2:.4f}")
    print(f"Bias: {bias:.2f}")

    # Analysis
    print(f"\nWeight ranges by feature group:")
    print(f"  Pawn PSQT:    [{weights[0:64].min():.1f}, {weights[0:64].max():.1f}]")
    print(f"  Knight PSQT:  [{weights[64:128].min():.1f}, {weights[64:128].max():.1f}]")
    print(f"  Bishop PSQT:  [{weights[128:192].min():.1f}, {weights[128:192].max():.1f}]")
    print(f"  Rook PSQT:    [{weights[192:256].min():.1f}, {weights[192:256].max():.1f}]")
    print(f"  Queen PSQT:   [{weights[256:320].min():.1f}, {weights[256:320].max():.1f}]")
    print(f"  King PSQT:    [{weights[320:384].min():.1f}, {weights[320:384].max():.1f}]")
    print(f"  Mobility:     {weights[384]:.1f}")
    print(f"  W-Iso Pawn:   {weights[385]:.1f}")
    print(f"  B-Iso Pawn:   {weights[386]:.1f}")
    print(f"  W-Passed Pawn:{weights[387]:.1f}")
    print(f"  B-Passed Pawn:{weights[388]:.1f}")
    print(f"  W-King Safety:{weights[389]:.1f}")
    print(f"  B-King Safety:{weights[390]:.1f}")

    if output_weights_file:
        export_weights_cpp(weights, output_weights_file)

    return weights


def export_weights_cpp(weights, output_file):
    """Export trained weights to C++ .inc file format (int16_t, scaled by 100)."""
    # Scale float weights to int16 (multiply by 100 to get centipawns)
    int_weights = np.clip(np.round(weights * 100), -32768, 32767).astype(np.int16)

    with open(output_file, 'w') as f:
        f.write("// Auto-generated evaluation weights (trained via Texel's tuning with Stockfish)\n")
        f.write(f"// Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("// Layout: PSQT (6 types x 64 squares), mobility, pawn features, king safety\n")
        f.write("// All values in centipawns from the piece's perspective (mirrored for black)\n")
        f.write("// A1=0, B1=1, ..., H1=7, A2=8, ..., H8=63\n")
        f.write("\n")

        # PSQT section header
        pt_names = ["PAWN", "KNIGHT", "BISHOP", "ROOK", "QUEEN", "KING"]
        for pt in range(6):
            f.write(f"\n// {pt_names[pt]} PSQT ({pt*64}-{pt*64+63})\n")
            for rank in range(7, -1, -1):
                if rank == 7:
                    f.write("// Row 0 = back rank\n")
                elif rank == 6:
                    f.write("// Row 1 = starting rank\n")
                else:
                    f.write(f"// Row {7-rank} = rank {rank+1}\n")
                for file in range(8):
                    sq = pt * 64 + (rank * 8 + file)
                    f.write(f"{int_weights[sq]:4d}")
                    if sq < NUM_FEATURES - 1:
                        f.write(",")
                f.write("\n")

        # Non-PSQT features
        f.write(f"\n// Mobility ({PSQT_SIZE}), Isolated W/B ({PSQT_SIZE+1}/{PSQT_SIZE+2}), "
                f"Passed W/B ({PSQT_SIZE+3}/{PSQT_SIZE+4}), "
                f"King safety W/B ({PSQT_SIZE+5}/{PSQT_SIZE+6})\n")
        for i in range(PSQT_SIZE, NUM_FEATURES):
            f.write(f"{int_weights[i]:4d}")
            if i < NUM_FEATURES - 1:
                f.write(", ")
        f.write("\n")

    print(f"Exported weights to {output_file}")
    print(f"  Weight range: [{int_weights.min()}, {int_weights.max()}]")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Train chess evaluation weights with Stockfish")
    parser.add_argument("--generate", type=int, default=0,
                        help="Generate N training positions")
    parser.add_argument("--train", action="store_true",
                        help="Train weights from saved data")
    parser.add_argument("--export", type=str, default=None,
                        help="Export trained weights to .inc file")
    parser.add_argument("--data", type=str, default="training_data.npz",
                        help="Training data file (.npz)")
    parser.add_argument("--output", type=str, default="../src/default_weights.inc",
                        help="Output weights file")
    args = parser.parse_args()

    if args.generate > 0:
        # Start Stockfish engine
        if not os.path.exists(STOCKFISH_PATH):
            print(f"Error: Stockfish not found at {STOCKFISH_PATH}")
            print("Please install Stockfish and update STOCKFISH_PATH")
            sys.exit(1)

        transport, engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

        try:
            generate_training_data(args.generate, engine, args.data)
        finally:
            engine.quit()

    if args.train or args.export:
        if not os.path.exists(args.data):
            print(f"Error: Training data file {args.data} not found.")
            print("Generate data first with: python train_weights.py --generate N")
            sys.exit(1)

        data = np.load(args.data)
        X, y = data['X'], data['y']
        print(f"Loaded {X.shape[0]} positions from {args.data}")

        weights = train_weights(X, y)

        if args.export or (args.train and args.output):
            outfile = args.export or args.output
            export_weights_cpp(weights, outfile)


if __name__ == "__main__":
    main()
