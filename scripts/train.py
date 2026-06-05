#!/usr/bin/env python3
"""
ChessAI Self-Training Pipeline.
Plays games vs Stockfish, collects positions+evals, tunes eval weights,
generates updated default_weights.inc, and validates improvement.
"""

import subprocess, sys, os, math, select, time, re
import numpy as np
from pathlib import Path

PROJECT = os.path.dirname(os.path.dirname(__file__))
ENGINE = os.path.join(PROJECT, "chessai")
STOCKFISH = "/usr/local/bin/stockfish"
WEIGHTS = os.path.join(PROJECT, "src", "default_weights.inc")

NUM_FEATURES = 391


def popen(path):
    p = subprocess.Popen([path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True, bufsize=1)
    return p


def send(p, cmd):
    p.stdin.write(cmd + "\n")
    p.stdin.flush()


def read(p, timeout=30):
    r, _, _ = select.select([p.stdout], [], [], timeout)
    return p.stdout.readline().strip() if r else None


def read_all(p, timeout=10):
    """Read all available lines until timeout."""
    lines = []
    while True:
        r, _, _ = select.select([p.stdout], [], [], 0.5)
        if r:
            l = p.stdout.readline().strip()
            if l:
                lines.append(l)
        else:
            break
    return lines


def handshake(p, timeout=5):
    send(p, "uci")
    while True:
        l = read(p, timeout)
        if l and "uciok" in l:
            break
    send(p, "isready")
    while True:
        l = read(p, timeout)
        if l and "readyok" in l:
            break


def set_position(p, moves=None, fen=None):
    if fen:
        send(p, f"position fen {fen}")
    elif moves:
        send(p, f"position startpos moves {' '.join(moves)}")
    else:
        send(p, "position startpos")


def get_features(p, moves=None, fen=None):
    """Get feature vector from our engine for a position."""
    set_position(p, moves, fen)
    send(p, "dump_features")
    while True:
        l = read(p, 10)
        if l is None:
            return None
        # Parse space-separated floats
        try:
            vals = [float(x) for x in l.strip().split()]
            if len(vals) == NUM_FEATURES:
                return np.array(vals, dtype=np.float32)
        except ValueError:
            continue


def get_sf_eval(p, depth=14):
    """Get Stockfish evaluation in centipawns (from white's perspective)."""
    send(p, f"go depth {depth}")
    score = 0
    while True:
        l = read(p, 30)
        if l is None:
            break
        if l.startswith("info"):
            parts = l.split()
            for i, p_ in enumerate(parts):
                if p_ == "cp" and i + 1 < len(parts):
                    score = int(parts[i + 1])
                elif p_ == "mate" and i + 1 < len(parts):
                    m = int(parts[i + 1])
                    return 30000 if m > 0 else -30000
        elif l.startswith("bestmove"):
            break
    return score


def get_bestmove(p, moves=None, fen=None, movetime=10, depth=None):
    set_position(p, moves, fen)
    cmd = "go"
    if depth:
        cmd += f" depth {depth}"
    else:
        cmd += f" movetime {movetime}"
    send(p, cmd)
    while True:
        l = read(p, 30)
        if l and l.startswith("bestmove"):
            return l.split()[1]
        if l is None:
            return None


def game_result(p, moves):
    """Analyze final position and return score from white's perspective (1/0.5/0)."""
    set_position(p, moves)
    send(p, "go depth 8")
    is_mate, mate_val = False, 0
    bm = "(none)"
    while True:
        l = read(p, 15)
        if l is None:
            break
        if l.startswith("info"):
            parts = l.split()
            for i, p_ in enumerate(parts):
                if p_ == "mate" and i + 1 < len(parts):
                    is_mate, mate_val = True, int(parts[i + 1])
        elif l.startswith("bestmove"):
            bm = l.split()[1] if len(l.split()) > 1 else "(none)"
            break
    n = len(moves)
    stm = n % 2
    if bm == "(none)":
        return 0.5 if not is_mate else (1.0 if stm == 1 else 0.0)
    if is_mate:
        return 1.0 if (stm == 1) == (mate_val > 0) else 0.0
    return 0.5


def play_and_collect(eng_proc, sf_proc, num_games, movetime=50, sf_depth=8):
    """
    Play games collecting (features, sf_eval, result) for each position.
    """
    X, y_sf, y_result = [], [], []
    total_positions = 0

    for g in range(num_games):
        our_white = (g % 2 == 0)
        moves = []
        game_positions = 0

        for ply in range(200):
            our_turn = our_white == (ply % 2 == 0)

            if our_turn:
                bm = get_bestmove(eng_proc, moves, movetime=movetime)
            else:
                bm = get_bestmove(sf_proc, moves, movetime=movetime, depth=sf_depth)

            if bm is None or bm == "(none)":
                break
            moves.append(bm)

            # Collect position after each full move (both sides played)
            if ply >= 2 and ply % 2 == 1 and len(moves) >= 2:
                # Get features from our engine
                feats = get_features(eng_proc, moves)
                if feats is not None:
                    # Get Stockfish's evaluation of this position
                    set_position(sf_proc, moves)
                    sf_eval = get_sf_eval(sf_proc)

                    # Convert to our perspective
                    if not our_white:
                        sf_eval = -sf_eval

                    X.append(feats)
                    y_sf.append(sf_eval)
                    game_positions += 1

        # Determine game result
        result = game_result(sf_proc, moves)
        # Result from our perspective
        if not our_white:
            result = 1.0 - result

        # Add result to all positions from this game
        for _ in range(game_positions):
            y_result.append(result)

        total_positions += game_positions
        sys.stdout.write(f"\r  Game {g+1}/{num_games}: {game_positions} positions, result={result:.1f}   ")
        sys.stdout.flush()

    print(f"\n  Total: {total_positions} positions from {num_games} games")
    return np.array(X), np.array(y_sf), np.array(y_result)


def train_weights(X, y, lr=0.001, epochs=200, reg=0.01):
    """
    Train eval weights using gradient descent.
    Target: Stockfish centipawn evaluation.
    Loss: MSE + L2 regularization.
    """
    n, d = X.shape
    print(f"  Training: {n} samples, {d} features")

    # Initialize weights from current values (read from file)
    w = read_current_weights().astype(np.float32)

    # Normalize targets to reasonable range (cap extreme values)
    y = np.clip(y, -3000, 3000)

    # Scale features
    X_binary = np.where(X > 0, 1.0, np.where(X < 0, -1.0, 0.0))

    best_w = w.copy()
    best_loss = float('inf')

    for epoch in range(epochs):
        # Forward: predict = X_binary @ w (since features are ±1 or 0)
        pred = X_binary @ w
        error = pred - y

        # MSE loss
        loss = np.mean(error ** 2) + reg * np.mean(w ** 2)

        # Gradient
        grad = (2.0 / n) * (X_binary.T @ error) + 2 * reg * w

        # Update
        w -= lr * grad

        if loss < best_loss:
            best_loss = loss
            best_w = w.copy()

        if epoch % 20 == 0:
            corr = np.corrcoef(pred, y)[0, 1]
            print(f"    Epoch {epoch:4d}: loss={loss:.1f}, corr={corr:.3f}")

    # Clip weights to reasonable range
    best_w = np.clip(best_w, -5000, 5000)

    print(f"  Final loss: {best_loss:.1f}")
    return best_w.astype(np.int32)


def read_current_weights():
    weights = []
    with open(WEIGHTS) as f:
        for line in f:
            line = line.strip()
            if line.startswith("//") or not line:
                continue
            if "//" in line:
                line = line.split("//")[0]
            for token in line.replace(",", " ").split():
                try:
                    weights.append(int(token))
                except ValueError:
                    pass
    return np.array(weights, dtype=np.int32)


def write_weights(weights, path=WEIGHTS):
    piece_names = ["PAWN", "KNIGHT", "BISHOP", "ROOK", "QUEEN", "KING"]
    with open(path, "w") as f:
        f.write("// Auto-generated by self-training pipeline\n")
        f.write("// DO NOT EDIT - generated by scripts/train.py\n\n")
        idx = 0
        for pt_idx, name in enumerate(piece_names):
            f.write(f"// {name} PSQT ({idx}-{idx+63})\n")
            for row in range(8):
                f.write(" ")
                for col in range(8):
                    f.write(f"{weights[idx]:4d},")
                    idx += 1
                f.write("\n")
            f.write("\n")
        f.write("// Mobility (384)\n")
        f.write(f" {weights[384]:4d},\n\n")
        f.write("// Isolated pawns white (385), black (386)\n")
        f.write(f" {weights[385]:4d}, {weights[386]:4d},\n\n")
        f.write("// Passed pawns white (387), black (388)\n")
        f.write(f" {weights[387]:4d}, {weights[388]:4d},\n\n")
        f.write("// King safety damage white (389), black (390)\n")
        f.write(f" {weights[389]:4d}, {weights[390]:4d},\n")


def bench(engine_path):
    r = subprocess.run([engine_path, "--bench"], capture_output=True, text=True, timeout=60)
    for line in r.stdout.strip().split("\n"):
        if line.startswith("info depth 1"):
            return True
    return r.returncode == 0


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=30, help="Training games")
    parser.add_argument("--sf-depth", type=int, default=6, help="Stockfish play depth")
    parser.add_argument("--eval-depth", type=int, default=12, help="Stockfish eval depth")
    parser.add_argument("--movetime", type=int, default=50, help="ms per move")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--pr-branch", type=str, default="auto-training",
                       help="Branch name for auto-PR")
    parser.add_argument("--no-pr", action="store_true", help="Skip PR creation")
    args = parser.parse_args()

    print("=" * 60)
    print("ChessAI Self-Training Pipeline")
    print("=" * 60)
    print(f"Games: {args.games}, SF depth: {args.sf_depth}, "
          f"Epochs: {args.epochs}, LR: {args.lr}")

    # Save original weights
    original_weights = read_current_weights()
    orig_mse = 0  # Will compute baseline

    # Start both engines
    print("\n[1] Starting engines...")
    eng = popen(ENGINE)
    handshake(eng, 5)
    sf = popen(STOCKFISH)
    handshake(sf, 5)

    # Build baseline
    print("\n[2] Building baseline evaluation...")
    set_position(sf)
    baseline_eval = get_sf_eval(sf)
    print(f"    Starting position eval: {baseline_eval} cp")

    # Play and collect
    print(f"\n[3] Playing {args.games} games vs Stockfish (depth={args.sf_depth})...")
    X, y_sf, y_result = play_and_collect(eng, sf, args.games, args.movetime, args.sf_depth)

    if len(X) == 0:
        print("ERROR: No data collected!")
        sys.exit(1)

    # Train weights
    print(f"\n[4] Training weights ({args.epochs} epochs, lr={args.lr})...")
    new_weights = train_weights(X, y_sf, args.lr, args.epochs)

    # Write new weights
    backup_path = WEIGHTS + ".bak"
    import shutil
    shutil.copy2(WEIGHTS, backup_path)
    write_weights(new_weights)
    print(f"\n[5] Wrote new weights to {WEIGHTS}")

    # Build and validate
    print("\n[6] Building engine with new weights...")
    import subprocess
    r = subprocess.run(["make", "clean", "-s"], cwd=PROJECT,
                       capture_output=True, text=True, timeout=30)
    r = subprocess.run(["make", "-j4", "-s"], cwd=PROJECT,
                       capture_output=True, text=True, timeout=30)
    if r.returncode != 0:
        print("ERROR: Build failed, restoring original weights")
        shutil.copy2(backup_path, WEIGHTS)
        sys.exit(1)
    print("    Build OK")

    # Benchmark
    print("\n[7] Benchmark...")
    r = subprocess.run([ENGINE, "--bench"], capture_output=True, text=True, timeout=60)
    print(r.stdout[-500:] if len(r.stdout) > 500 else r.stdout)

    # Quick validation match
    print(f"\n[8] Validation match vs Stockfish...")
    sf2 = popen(STOCKFISH)
    handshake(sf2, 5)

    eng2 = popen(ENGINE)
    handshake(eng2, 5)

    X_val, y_val_sf, y_val_res = play_and_collect(eng2, sf2, min(6, args.games // 3),
                                                   args.movetime, args.sf_depth)

    if len(X_val) > 0:
        # Compute correlation between our eval and Stockfish eval
        X_binary = np.where(X_val > 0, 1.0, np.where(X_val < 0, -1.0, 0.0))
        our_preds = X_binary @ new_weights
        corr = np.corrcoef(our_preds, y_val_sf)[0, 1]
        mae = np.mean(np.abs(our_preds - y_val_sf))
        print(f"    Validation correlation: {corr:.3f}")
        print(f"    Validation MAE: {mae:.1f} cp")

    # Clean up
    for p in [eng, sf, eng2, sf2]:
        try:
            send(p, "quit")
            p.wait(timeout=2)
        except:
            p.kill()

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
