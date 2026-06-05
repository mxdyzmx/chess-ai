#!/usr/bin/env python3
"""
Train evaluation weights for the Chess AI engine.

Usage:
    # Extract features and train
    python train.py --pgn games.pgn --samples 200000 --output weights.txt

    # Load pre-extracted features
    python train.py --load-data X.npy y.npy --output weights.txt

    # Just extract features (no training)
    python train.py --pgn games.pgn --samples 200000 --save-data only

The training uses SGD regression to learn a linear evaluation function
from game outcomes. The resulting weights are exported as a C++ int16_t array
that can be directly pasted into eval_weights[] in the engine.
"""

import argparse
import gzip
import os
import sys
import time

import numpy as np
from sklearn.linear_model import SGDRegressor

from extract_features import extract_features

try:
    import chess
    import chess.pgn
except ImportError:
    print("Error: python-chess is required. Install with: pip install python-chess")
    sys.exit(1)


def sample_from_pgn(pgn_file: str, max_samples: int = 200000) -> tuple:
    """Sample positions from a PGN file and extract features.

    Args:
        pgn_file: Path to PGN file (can be .gz compressed)
        max_samples: Maximum number of positions to sample

    Returns:
        (X, y) tuple of features and labels
    """
    print(f"Reading PGN: {pgn_file}")
    print(f"Target samples: {max_samples}")

    # Open file (support .gz compression)
    if pgn_file.endswith('.gz'):
        handle = gzip.open(pgn_file, 'rt', encoding='utf-8', errors='replace')
    else:
        handle = open(pgn_file, 'r', encoding='utf-8', errors='replace')

    X_list = []
    y_list = []
    games_processed = 0
    positions_extracted = 0

    start_time = time.time()

    try:
        while positions_extracted < max_samples:
            game = chess.pgn.read_game(handle)
            if game is None:
                break  # No more games

            games_processed += 1
            if games_processed % 100 == 0:
                elapsed = time.time() - start_time
                print(f"  Games: {games_processed}, Positions: {positions_extracted}, "
                      f"Time: {elapsed:.1f}s")

            # Determine game result
            result = game.headers.get("Result", "*")
            if result == "1-0":
                label = 1.0
            elif result == "0-1":
                label = -1.0
            elif result == "1/2-1/2":
                label = 0.0
            else:
                continue  # Unknown result, skip

            # Play through the game and sample positions
            board = game.board()
            positions_in_game = []
            seen_hashes = set()

            for move in game.mainline_moves():
                # Skip the first few moves (opening book noise)
                if board.fullmove_number >= 4:
                    # Avoid duplicate positions
                    board_hash = board._transposition_key()
                    if board_hash not in seen_hashes:
                        seen_hashes.add(board_hash)
                        positions_in_game.append(board.fen())

                board.push(move)

            # Sample positions from this game (to maximize diversity)
            # Use the final label for all positions (approximation for linear model)
            # Weight: later positions are more correlated with the result,
            # but earlier positions help with general evaluation
            # For efficiency, take every Nth position
            if len(positions_in_game) > 0:
                step = max(1, len(positions_in_game) // 5)  # ~5 per game
                for fen in positions_in_game[::step]:
                    if positions_extracted >= max_samples:
                        break
                    try:
                        features = extract_features(chess.Board(fen))
                        X_list.append(features)
                        y_list.append(label)
                        positions_extracted += 1
                    except Exception as e:
                        continue

            if games_processed % 500 == 0 and positions_extracted > 0:
                # Periodically print stats
                print(f"    Feature matrix size: {len(X_list)} samples, "
                      f"{len(X_list[0]) if X_list else 0} features")

    finally:
        handle.close()

    if len(X_list) == 0:
        print("Error: No positions extracted!")
        sys.exit(1)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    elapsed = time.time() - start_time
    print(f"\nDone: {games_processed} games processed, {positions_extracted} positions extracted")
    print(f"Total time: {elapsed:.1f}s")
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"y distribution: wins={np.sum(y > 0)}, losses={np.sum(y < 0)}, draws={np.sum(y == 0)}")

    return X, y


def train_model(X: np.ndarray, y: np.ndarray) -> tuple:
    """Train a linear model using SGD regression.

    Args:
        X: Feature matrix (n_samples, n_features)
        y: Labels (+1=white win, -1=black win, 0=draw)

    Returns:
        (weights, intercept) tuple
    """
    print(f"\nTraining model on {X.shape[0]} samples with {X.shape[1]} features...")

    # Note: fitting_intercept=False because we'll handle bias separately
    # We use epsilon_insensitive loss for robustness
    model = SGDRegressor(
        loss='huber',  # Less sensitive to outliers
        penalty='l2',
        alpha=0.0001,
        max_iter=1000,
        tol=1e-4,
        learning_rate='optimal',
        fit_intercept=True,
        random_state=42,
        n_jobs=-1,
    )

    start = time.time()
    model.fit(X, y)
    elapsed = time.time() - start

    weights = model.coef_.copy()
    intercept = model.intercept_[0]

    print(f"Training completed in {elapsed:.1f}s")
    print(f"Intercept: {intercept:.6f}")

    # Compute training accuracy (how well does the model predict game outcome?)
    y_pred = model.predict(X)
    # Accuracy: % of predictions with correct sign (excluding draws)
    non_draws = y != 0
    if np.sum(non_draws) > 0:
        correct_sign = np.mean(np.sign(y_pred[non_draws]) == np.sign(y[non_draws]))
        print(f"Sign accuracy (non-draw positions): {correct_sign:.3f}")

    # Compute correlation
    corr = np.corrcoef(y, y_pred)[0, 1]
    print(f"Pearson correlation: {corr:.4f}")

    # Print weight statistics
    print(f"Weight range: [{weights.min():.4f}, {weights.max():.4f}]")
    print(f"Weight std: {weights.std():.4f}")

    return weights, intercept


def export_cpp_weights(weights: np.ndarray, intercept: float, output_file: str,
                       scale: float = 100.0) -> None:
    """Export weights as C++ int16_t array.

    Args:
        weights: Trained weight vector (n_features,)
        intercept: Bias/intercept term
        output_file: Output file path
        scale: Scale factor for converting float to int16
    """
    # Scale weights
    scaled = np.round(weights * scale).astype(np.int16)

    # Cap to int16 range
    scaled = np.clip(scaled, -32768, 32767)

    content = f"""// Auto-generated evaluation weights
// Generated by train.py
// Scale: {scale} (float * {scale} = int16)
// Intercept: {intercept:.6f} (bias not used in C++ engine)
// Feature layout:
//   0-63:   pawn PSQT
//   64-127: knight PSQT
//   128-191: bishop PSQT
//   192-255: rook PSQT
//   256-319: queen PSQT
//   320-383: king PSQT
//   384: mobility
//   385: white isolated pawn count
//   386: black isolated pawn count
//   387: white passed pawn count
//   388: black passed pawn count
//   389: white king safety damage
//   390: black king safety damage

// Replace the content of src/default_weights.inc with these values
int16_t eval_weights[{len(scaled)}] = {{"""

    # Write values in rows of 16
    for i in range(0, len(scaled), 16):
        row = scaled[i:i + 16]
        content += "\n    " + ", ".join(f"{v:5d}" for v in row) + ","

    content += f"""
}};
"""

    with open(output_file, 'w') as f:
        f.write(content)

    print(f"\nWeights exported to: {output_file}")
    print(f"Copy this file to src/default_weights.inc or update eval.cpp to include it.")


def save_data(X: np.ndarray, y: np.ndarray, prefix: str = "data") -> None:
    """Save extracted features to disk."""
    np.save(f"{prefix}_X.npy", X)
    np.save(f"{prefix}_y.npy", y)
    print(f"Saved {X.shape[0]} samples to {prefix}_X.npy and {prefix}_y.npy")


def main():
    parser = argparse.ArgumentParser(description="Train chess evaluation weights")
    parser.add_argument("--pgn", type=str, help="Path to PGN file")
    parser.add_argument("--samples", type=int, default=200000,
                        help="Maximum number of positions to sample")
    parser.add_argument("--load-data", type=str, nargs=2, metavar=("X.npy", "y.npy"),
                        help="Load pre-extracted features")
    parser.add_argument("--output", type=str, default="weights.txt",
                        help="Output file for C++ weights")
    parser.add_argument("--save-data", type=str, nargs="?", const="data",
                        help="Save extracted features to disk (optional: prefix)")
    parser.add_argument("--no-train", action="store_true",
                        help="Only extract features, don't train")

    args = parser.parse_args()

    # Load or extract data
    if args.load_data:
        X = np.load(args.load_data[0])
        y = np.load(args.load_data[1])
        print(f"Loaded {X.shape[0]} samples from {args.load_data[0]}")
    elif args.pgn:
        X, y = sample_from_pgn(args.pgn, args.samples)
        if args.save_data:
            save_data(X, y, args.save_data)
    else:
        parser.print_help()
        print("\nError: Either --pgn or --load-data is required.")
        sys.exit(1)

    if args.no_train:
        print("Skipping training (--no-train specified)")
        return

    # Train model
    weights, intercept = train_model(X, y)

    # Export weights
    export_cpp_weights(weights, intercept, args.output)


if __name__ == "__main__":
    main()
