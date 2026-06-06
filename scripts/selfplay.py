#!/usr/bin/env python3
"""
ChessAI Self-Play Training.
Plays chessai vs chessai, collects features from both sides,
trains eval weights to predict game outcome from white's perspective.

Usage:
  python3 scripts/selfplay.py --games 50 --movetime 100 --epochs 500
"""

import subprocess, sys, os, math, time, threading, queue, shutil
import numpy as np
import chess

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENGINE = os.path.join(PROJECT, "chessai")
WEIGHTS = os.path.join(PROJECT, "src", "default_weights.inc")
NUM_FEATURES = 391
RESULT_WIN = 500    # centipawn equivalent for a win
RESULT_LOSS = -500  # centipawn equivalent for a loss


class Engine:
    """UCI engine with background stdout reader thread."""

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
        while not self._lines.empty():
            try:
                self._lines.get_nowait()
            except queue.Empty:
                break

    def send(self, cmd):
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()

    def read(self, timeout=30):
        try:
            return self._lines.get(timeout=timeout)
        except queue.Empty:
            return None

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


def get_bestmove_fen(p, fen, movetime=100):
    """Send position fen + go, return best move UCI string or None."""
    # Stop any ongoing search first
    p.send("stop")
    p.drain()
    p.send(f"position fen {fen}")
    p.send(f"go movetime {movetime}")
    while True:
        l = p.read(30)
        if l and l.startswith("bestmove"):
            return l.split()[1]
        if l is None:
            return None


def get_features_fen(p, fen):
    """Get feature vector for a position (by FEN, not move list)."""
    p.drain()
    p.send(f"position fen {fen}")
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


def result_from_white(board, moves_len, illegal_side=None):
    """
    Determine game result from white's perspective using python-chess.
    Returns +1.0 (white wins), 0.0 (black wins), or 0.5 (draw).
    """
    if illegal_side is not None:
        return 0.0 if illegal_side == chess.WHITE else 1.0

    if board.is_checkmate():
        # The side to move is checkmated
        return 0.0 if board.turn == chess.WHITE else 1.0

    if board.is_stalemate() or board.is_insufficient_material():
        return 0.5

    # 50-move rule, repetition, or 200-ply limit
    return 0.5


def play_one_game(white_eng, black_eng, movetime):
    """
    Play one self-play game. Returns (X_list, y_list, result_white).
    X_list: feature vectors (one per ply).
    y_list: training target from white's perspective for each position.
    result_white: +1/0.5/0 from white's perspective.
    """
    board = chess.Board()
    X_game = []
    illegal_side = None

    for ply in range(200):
        eng = white_eng if board.turn == chess.WHITE else black_eng

        # Collect features BEFORE the move
        feats = get_features_fen(eng, board.fen())
        if feats is not None:
            X_game.append(feats)

        # Get best move
        bm = get_bestmove_fen(eng, board.fen(), movetime)
        if bm is None or bm == "(none)":
            break

        # Validate with python-chess
        m = chess.Move.from_uci(bm)
        if m not in board.legal_moves:
            illegal_side = board.turn
            break
        board.push_uci(bm)

        if board.is_game_over():
            break

    # Determine result from white's perspective
    result_white = result_from_white(board, len(X_game), illegal_side)

    # Label all positions: target from WHITE's perspective
    # evaluate() returns X@w for white-to-move and -(X@w) for black-to-move
    # So train X@w ≈ result_from_white regardless of STM
    target_cp = int(RESULT_WIN if result_white == 1.0 else
                    (RESULT_LOSS if result_white == 0.0 else 0))
    y_game = [target_cp] * len(X_game)

    return X_game, y_game, result_white, illegal_side is not None


def play_and_collect(eng_a, eng_b, num_games, movetime):
    """Play N self-play games, alternate which engine plays white."""
    X_all, y_all = [], []
    total_pos = 0
    illegal_games = 0

    for g in range(num_games):
        # Alternate which process plays white
        white_eng = eng_a if g % 2 == 0 else eng_b
        black_eng = eng_b if g % 2 == 0 else eng_a

        X_game, y_game, result_w, was_illegal = play_one_game(
            white_eng, black_eng, movetime
        )

        X_all.extend(X_game)
        y_all.extend(y_game)
        total_pos += len(X_game)
        if was_illegal:
            illegal_games += 1

        result_str = {1.0: "1-0", 0.5: "1/2-1/2", 0.0: "0-1"}[result_w]
        sys.stdout.write(
            f"\r  Game {g+1}/{num_games}: {len(X_game)} positions, "
            f"result={result_str}{' ILLEGAL' if was_illegal else ''}   "
        )
        sys.stdout.flush()

    print(f"\n  Total: {total_pos} positions from {num_games} games"
          f" ({illegal_games} illegal)")
    return np.array(X_all), np.array(y_all)


def train_weights(X, y, lr=0.001, epochs=200, reg=0.01):
    """
    Train eval weights using gradient descent.
    Target: game result from white's perspective (±500 cp).
    Loss: MSE + L2 regularization.
    """
    n, d = X.shape
    print(f"  Training: {n} samples, {d} features")

    if n < 10:
        print(f"  WARNING: too few samples ({n}), skipping training")
        return read_current_weights().astype(np.int32)

    w = read_current_weights().astype(np.float32)
    y = np.clip(y, -3000, 3000)

    best_w = w.copy()
    best_loss = float('inf')

    for epoch in range(epochs):
        pred = X @ w
        error = pred - y
        loss = np.mean(error ** 2) + reg * np.mean(w ** 2)
        grad = (2.0 / n) * (X.T @ error) + 2 * reg * w
        w -= lr * grad

        if loss < best_loss:
            best_loss = loss
            best_w = w.copy()

        if epoch % 20 == 0:
            corr = np.corrcoef(pred, y)[0, 1]
            print(f"    Epoch {epoch:4d}: loss={loss:.1f}, corr={corr:.3f}")

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
        f.write("// Auto-generated by self-play training\n")
        f.write("// DO NOT EDIT - generated by scripts/selfplay.py\n\n")
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
    parser = argparse.ArgumentParser(description="Self-play training for ChessAI")
    parser.add_argument("--games", type=int, default=50, help="Number of self-play games")
    parser.add_argument("--movetime", type=int, default=100, help="Milliseconds per move")
    parser.add_argument("--epochs", type=int, default=500, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--val-games", type=int, default=10, help="Validation games")
    args = parser.parse_args()

    print("=" * 60)
    print("ChessAI Self-Play Training")
    print("=" * 60)
    print(f"Games: {args.games}, movetime: {args.movetime}ms, "
          f"Epochs: {args.epochs}, LR: {args.lr}")

    # Start two engine instances
    print("\n[1] Starting engines...")
    eng_a = Engine(ENGINE)
    eng_a.handshake(5)
    eng_b = Engine(ENGINE)
    eng_b.handshake(5)

    # Play self-play games
    print(f"\n[2] Playing {args.games} self-play games...")
    X, y = play_and_collect(eng_a, eng_b, args.games, args.movetime)

    if len(X) == 0:
        print("ERROR: No data collected!")
        sys.exit(1)

    # Train
    print(f"\n[3] Training weights ({args.epochs} epochs, lr={args.lr})...")
    new_weights = train_weights(X, y, args.lr, args.epochs)

    # Save original weights backup
    backup_path = WEIGHTS + ".bak"
    shutil.copy2(WEIGHTS, backup_path)

    # Write new weights
    write_weights(new_weights)
    print(f"\n[4] Wrote new weights to {WEIGHTS}")

    # Save old binary BEFORE building new one
    old_bin_path = os.path.join(PROJECT, "chessai.old")
    shutil.copy2(ENGINE, old_bin_path)
    print("    Saved old binary as chessai.old")

    # Build new engine with new weights
    print("\n[5] Building engine with new weights...")
    r = subprocess.run(["make", "clean", "-s"], cwd=PROJECT,
                       capture_output=True, text=True, timeout=30)
    r = subprocess.run(["make", "-j4", "-s"], cwd=PROJECT,
                       capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        print("ERROR: Build failed, restoring original weights")
        shutil.copy2(backup_path, WEIGHTS)
        # Restore old binary
        shutil.copy2(old_bin_path, ENGINE)
        os.remove(old_bin_path)
        sys.exit(1)
    print("    Build OK")

    # Benchmark
    print("\n[6] Benchmark...")
    r = subprocess.run([ENGINE, "--bench"], capture_output=True, text=True, timeout=60)
    print(r.stdout[-500:] if len(r.stdout) > 500 else r.stdout)

    # Validation: old binary vs new binary
    print(f"\n[7] Validation: old vs new ({args.val_games} games)...")
    old_eng = Engine(old_bin_path)
    old_eng.handshake(5)
    new_eng = Engine(ENGINE)
    new_eng.handshake(5)

    w = d = l = 0
    for g in range(args.val_games):
        white_eng = new_eng if g % 2 == 0 else old_eng
        black_eng = old_eng if g % 2 == 0 else new_eng

        X_v, y_v, result_w, _ = play_one_game(white_eng, black_eng, args.movetime)

        # result_w is from white's perspective. Is new engine white?
        new_is_white = (g % 2 == 0)
        if new_is_white:
            if result_w == 1.0:
                w += 1
            elif result_w == 0.5:
                d += 1
            else:
                l += 1
        else:
            if result_w == 1.0:
                l += 1
            elif result_w == 0.5:
                d += 1
            else:
                w += 1

        sys.stdout.write(f"\r    Game {g+1}/{args.val_games}: +{w} -{l} ={d}   ")
        sys.stdout.flush()
    print()

    total = w + d + l
    if total > 0:
        score_pct = (w + d / 2) / total * 100
        print(f"    New engine score: {score_pct:.1f}% (+{w} -{l} ={d})")

    # Clean up
    for p in [eng_a, eng_b, old_eng, new_eng]:
        p.close()

    if os.path.exists(old_bin_path):
        os.remove(old_bin_path)

    print("\n" + "=" * 60)
    print("Self-play training complete!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
