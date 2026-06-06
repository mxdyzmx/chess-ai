#!/usr/bin/env python3
"""
ChessAI Self-Training Pipeline.
Plays games vs Stockfish, collects positions+evals, tunes eval weights,
generates updated default_weights.inc, and validates improvement.
"""

import subprocess, sys, os, math, time, re, threading, queue, shutil
import numpy as np
import chess
from pathlib import Path

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENGINE = os.path.join(PROJECT, "chessai")
STOCKFISH = shutil.which("stockfish") or "/usr/games/stockfish"
WEIGHTS = os.path.join(PROJECT, "src", "default_weights.inc")

NUM_FEATURES = 391


class Engine:
    """UCI engine with background stdout reader thread (macOS-safe)."""

    def __init__(self, path):
        self.proc = subprocess.Popen(
            [path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1
        )
        self._lines = queue.Queue()
        self._closed = False
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self):
        try:
            for line in self.proc.stdout:
                self._lines.put(line.rstrip('\n\r'))
        except ValueError:
            pass
        self._closed = True

    def drain(self):
        """Drain all pending stdout/stderr lines (clear stale data between games)."""
        lines = []
        while not self._lines.empty():
            try:
                self._lines.get_nowait()
            except queue.Empty:
                break
        return lines

    def send(self, cmd):
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()

    def read(self, timeout=30):
        try:
            return self._lines.get(timeout=timeout)
        except queue.Empty:
            return None

    def read_all(self, timeout=10):
        lines = []
        start = time.time()
        while time.time() - start < timeout:
            try:
                l = self._lines.get(timeout=0.5)
                if l:
                    lines.append(l)
            except queue.Empty:
                break
        return lines

    def handshake(self, timeout=5):
        self.send("uci")
        while True:
            l = self.read(timeout)
            if l and "uciok" in l:
                break
        self.send("isready")
        while True:
            l = self.read(timeout)
            if l and "readyok" in l:
                break

    def close(self):
        try:
            self.send("quit")
            self.proc.wait(timeout=3)
        except:
            self.proc.kill()


def set_position(p, moves=None, fen=None):
    if fen:
        p.send(f"position fen {fen}")
    elif moves:
        p.send(f"position startpos moves {' '.join(moves)}")
    else:
        p.send("position startpos")


def get_features(p, moves=None, fen=None):
    """Get feature vector from our engine for a position."""
    p.drain()
    set_position(p, moves, fen)
    p.send("dump_features")
    while True:
        l = p.read(10)
        if l is None:
            return None
        try:
            vals = [float(x) for x in l.strip().split()]
            if len(vals) == NUM_FEATURES:
                return np.array(vals, dtype=np.float32)
        except ValueError:
            continue


def get_sf_eval(p, depth=14):
    """Get Stockfish evaluation in centipawns (from white's perspective).
       Returns None on timeout."""
    # Stop any ongoing search before starting a new eval
    p.send("stop")
    p.drain()
    p.send(f"go depth {depth}")
    score = None
    while True:
        l = p.read(60)
        if l is None:
            # Timeout — stop the search and drain stale bestmove
            p.send("stop")
            p.drain()
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
    # Stop any ongoing search and flush stale output
    p.send("stop")
    p.drain()
    # Read any additional lines that arrive within 500ms (e.g. bestmove from stopped search)
    end = time.time() + 0.5
    while time.time() < end:
        l = p.read(0.1)
        if l is None:
            break
    set_position(p, moves, fen)
    cmd = "go"
    if depth:
        cmd += f" depth {depth}"
    else:
        cmd += f" movetime {movetime}"
    p.send(cmd)
    while True:
        l = p.read(30)
        if l and l.startswith("bestmove"):
            return l.split()[1]
        if l is None:
            return None


def game_result(p, moves):
    """Analyze final position and return score from white's perspective (1/0.5/0)."""
    # Stop any ongoing search before starting new analysis
    p.send("stop")
    p.drain()
    set_position(p, moves)
    p.send("go depth 8")
    is_mate, mate_val = False, 0
    bm = "(none)"
    while True:
        l = p.read(30)
        if l is None:
            p.send("stop")
            p.drain()
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
        # STM has no legal moves = checkmate or stalemate
        return 0.5 if not is_mate else (1.0 if stm == 1 else 0.0)
    if is_mate:
        # mate_val > 0 means STM delivers mate; mate_val < 0 means opponent does.
        # stm=0=white, stm=1=black.
        # White wins: (stm=0 and mate>0) or (stm=1 and mate<0)
        return 1.0 if (stm == 0) == (mate_val > 0) else 0.0
    return 0.5


def play_and_collect(eng_proc, sf_proc, num_games, movetime=50, sf_depth=8, eval_depth=12):
    """
    Play games collecting (features, sf_eval, result) for each position.
    """
    X, y_sf, y_result = [], [], []
    total_positions = 0

    for g in range(num_games):
        # Drain any stale output from both engines between games
        eng_proc.drain()
        sf_proc.drain()
        our_white = (g % 2 == 0)
        board = chess.Board()
        moves = []
        game_positions = 0
        illegal_abort = False

        for ply in range(200):
            our_turn = our_white == (ply % 2 == 0)

            if our_turn:
                bm = get_bestmove(eng_proc, moves, movetime=movetime)
            else:
                bm = get_bestmove(sf_proc, moves, movetime=movetime, depth=sf_depth)

            if bm is None or bm == "(none)":
                break

            # Validate our engine's move against python-chess board
            if our_turn:
                try:
                    m = chess.Move.from_uci(bm)
                    if m not in board.legal_moves:
                        print(f"\n    [ILLEGAL] {bm} | {board.fen()}", flush=True)
                        illegal_abort = True
                        break
                except Exception as e:
                    print(f"\n    [PARSE ERROR] {bm}: {e} | {board.fen()}", flush=True)
                    illegal_abort = True
                    break

            # Apply to python-chess board
            try:
                board.push_uci(bm)
            except Exception as e:
                print(f"\n    [PUSH ERROR] {bm}: {e} | {board.fen()} | our_turn={our_turn}", flush=True)
                illegal_abort = True
                break

            moves.append(bm)

            # Collect position after each full move (both sides played)
            if ply >= 2 and ply % 2 == 1 and len(moves) >= 2:
                # Get features from our engine
                feats = get_features(eng_proc, moves)
                if feats is not None:
                    # Get Stockfish's evaluation of this position
                    set_position(sf_proc, moves)
                    sf_eval = get_sf_eval(sf_proc, depth=eval_depth)

                    if sf_eval is None:
                        continue  # skip if eval timeout

                    # Convert to our perspective
                    if not our_white:
                        sf_eval = -sf_eval

                    X.append(feats)
                    y_sf.append(sf_eval)
                    game_positions += 1

        # Determine game result (skip for illegal games)
        if illegal_abort:
            result = 0.0
            print(f"    Illegal move — game forfeited")
        else:
            result = game_result(sf_proc, moves)
            # Result from our perspective
            if not our_white:
                result = 1.0 - result

        # Add result to all positions from this game
        for _ in range(game_positions):
            y_result.append(result)

        total_positions += game_positions
        sys.stdout.write(f"\r  Game {g+1}/{num_games}: {game_positions} positions, result={result:.1f}" +
                         (" ILLEGAL" if illegal_abort else "") + "   ")
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

    if n < 10:
        print(f"  WARNING: too few samples ({n}), skipping training")
        return read_current_weights().astype(np.int32)

    # Initialize weights from current values (read from file)
    w = read_current_weights().astype(np.float32)

    # Normalize targets to reasonable range (cap extreme values)
    y = np.clip(y, -3000, 3000)

    best_w = w.copy()
    best_loss = float('inf')

    for epoch in range(epochs):
        # Forward: predict = X @ w (features are already ±1/0 for PSQT and floats for the rest)
        pred = X @ w
        error = pred - y

        # MSE loss
        loss = np.mean(error ** 2) + reg * np.mean(w ** 2)

        # Gradient
        grad = (2.0 / n) * (X.T @ error) + 2 * reg * w

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
    eng = Engine(ENGINE)
    eng.handshake(5)
    sf = Engine(STOCKFISH)
    sf.handshake(5)

    # Build baseline
    print("\n[2] Building baseline evaluation...")
    set_position(sf)
    baseline_eval = get_sf_eval(sf)
    print(f"    Starting position eval: {baseline_eval} cp")

    # Play and collect
    print(f"\n[3] Playing {args.games} games vs Stockfish (depth={args.sf_depth})...")
    X, y_sf, y_result = play_and_collect(eng, sf, args.games, args.movetime, args.sf_depth, args.eval_depth)

    if len(X) == 0:
        print("ERROR: No data collected!")
        sys.exit(1)

    # Train weights
    print(f"\n[4] Training weights ({args.epochs} epochs, lr={args.lr})...")
    new_weights = train_weights(X, y_sf, args.lr, args.epochs)

    # Write new weights
    backup_path = WEIGHTS + ".bak"
    shutil.copy2(WEIGHTS, backup_path)
    write_weights(new_weights)
    print(f"\n[5] Wrote new weights to {WEIGHTS}")

    # Build and validate
    print("\n[6] Building engine with new weights...")
    import subprocess
    r = subprocess.run(["make", "clean", "-s"], cwd=PROJECT,
                       capture_output=True, text=True, timeout=30)
    r = subprocess.run(["make", "-j4", "-s"], cwd=PROJECT,
                       capture_output=True, text=True, timeout=120)
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
    sf2 = Engine(STOCKFISH)
    sf2.handshake(5)

    eng2 = Engine(ENGINE)
    eng2.handshake(5)

    X_val, y_val_sf, y_val_res = play_and_collect(eng2, sf2, min(6, args.games // 3),
                                                   args.movetime, args.sf_depth, args.eval_depth)

    if len(X_val) > 0:
        # Compute correlation between our eval and Stockfish eval
        our_preds = X_val @ new_weights
        corr = np.corrcoef(our_preds, y_val_sf)[0, 1]
        mae = np.mean(np.abs(our_preds - y_val_sf))
        print(f"    Validation correlation: {corr:.3f}")
        print(f"    Validation MAE: {mae:.1f} cp")

    # Clean up
    for p in [eng, sf, eng2, sf2]:
        p.close()

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
