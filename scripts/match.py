#!/usr/bin/env python3
"""Play matches between ChessAI and Stockfish. Uses depth-limited Stockfish."""
import subprocess, sys, os, math, time, shutil, threading, queue
import chess

ENGINE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "chessai")
STOCKFISH = shutil.which("stockfish") or "/usr/games/stockfish"

# Approximate Stockfish Elo at fixed depths (bullet)
SF_DEPTH_ELO = {1: 1550, 2: 1800, 3: 1950, 4: 2100, 5: 2200, 6: 2300,
                8: 2500, 10: 2700, 12: 2900}


class EngineProcess:
    """Manages a UCI engine subprocess with background stdout reader thread."""

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
        """Background thread: read lines from stdout into a queue."""
        try:
            for line in self.proc.stdout:
                self._lines.put(line.rstrip('\n\r'))
        except ValueError:
            pass
        self._closed = True

    def send(self, cmd):
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()

    def read_line(self, timeout=30):
        """Read one line with timeout. Returns None on timeout."""
        try:
            return self._lines.get(timeout=timeout)
        except queue.Empty:
            return None

    def wait_for(self, target, timeout=30):
        end = time.time() + timeout
        while time.time() < end:
            line = self.read_line(min(1, timeout))
            if line and target in line:
                return line
        return None

    def handshake(self):
        self.send("uci")
        self.wait_for("uciok", 5)
        self.send("isready")
        self.wait_for("readyok", 5)

    def get_bestmove(self, fen, movetime=20, depth=None):
        """Send position fen + go, return best move string."""
        self.send(f"position fen {fen}")
        cmd = "go"
        if depth:
            cmd += f" depth {depth}"
        else:
            cmd += f" movetime {movetime}"
        self.send(cmd)
        while True:
            line = self.read_line(30)
            if line is None:
                return None
            if line.startswith("bestmove"):
                parts = line.split()
                return parts[1] if len(parts) > 1 else "(none)"

    def close(self):
        try:
            self.send("quit")
            self.proc.wait(timeout=3)
        except:
            self.proc.kill()


def analyze_position(sf, fen):
    """Analyze a position with Stockfish. Returns (cp, is_mate, mate_val, bestmove)."""
    sf.send(f"position fen {fen}")
    sf.send("go depth 8")
    cp, is_mate, mate_val, bm = 0, False, 0, "(none)"
    while True:
        line = sf.read_line(15)
        if line is None:
            break
        if line.startswith("info"):
            parts = line.split()
            for i, p in enumerate(parts):
                if p == "cp" and i + 1 < len(parts) and not is_mate:
                    cp = int(parts[i + 1])
                if p == "mate" and i + 1 < len(parts):
                    is_mate, mate_val = True, int(parts[i + 1])
        elif line.startswith("bestmove"):
            parts = line.split()
            bm = parts[1] if len(parts) > 1 else "(none)"
            return cp, is_mate, mate_val, bm
    return cp, is_mate, mate_val, bm


def format_pgn(moves, result_str):
    """Convert UCI move list to PGN notation."""
    pgn = ""
    for i in range(0, len(moves), 2):
        move_num = i // 2 + 1
        white = moves[i]
        black = moves[i + 1] if i + 1 < len(moves) else ""
        pgn += f"{move_num}. {white} {black} "
    pgn += "{" + result_str + "}"
    return pgn


def play_one_game(our, sf, sf_depth, movetime, our_white=True):
    """Play one game: our engine vs Stockfish.
    Uses python-chess to track position; sends FEN to both engines every turn.
    Returns (score, moves_list, illegal_flag).
    If our engine plays an illegal move, forfeit immediately (score=0.0, illegal=True)."""
    board = chess.Board()
    if not our_white:
        # Our engine plays black: swap assignment so `our` always gets correct ply
        white, black = sf, our
    else:
        white, black = our, sf
    engines = {chess.WHITE: white, chess.BLACK: black}
    moves = []

    for ply in range(200):
        fen = board.fen()
        cur = engines[board.turn]
        if cur is our:
            bm = cur.get_bestmove(fen, movetime=movetime)
        else:
            bm = cur.get_bestmove(fen, movetime=movetime, depth=sf_depth)

        print(bm)
        if bm is None or bm == "(none)":
            break

        # Validate our engine's move with python-chess
        if cur is our:
            try:
                m = chess.Move.from_uci(bm)
                if m not in board.legal_moves:
                    print(f"\n[ILLEGAL] {bm} | FEN: {board.fen()}", flush=True)
                    return 0.0, moves, True  # illegal move, forfeit
            except Exception as e:
                print(f"\n[PARSE ERROR] {bm}: {e} | FEN: {board.fen()}", flush=True)
                return 0.0, moves, True  # unparseable move, forfeit

        # Apply move to python-chess board
        try:
            board.push_uci(bm)
        except Exception:
            return 0.0, moves, True

        moves.append(bm)

    n = len(moves)
    sf_result = analyze_position(sf, board.fen())
    cp, is_mate, mate_val, final_bm = sf_result

    # Determine result from our engine's perspective
    if final_bm == "(none)":
        stm = n % 2
        if is_mate:
            score = 0.0 if stm == 0 else 1.0
        else:
            score = 0.5
    elif is_mate:
        stm = n % 2
        if mate_val > 0:
            score = 1.0 if stm == 0 else 0.0
        else:
            score = 0.0 if stm == 0 else 1.0
    elif n >= 190:
        score = 0.5
    elif cp > 300:
        score = 1.0
    elif cp < -300:
        score = 0.0
    else:
        score = 0.5

    return score, moves, False


def estimate_elo(w, d, l, opp):
    n = w + d + l
    if n == 0:
        return 0, 0
    s = (w + d / 2) / n
    if s <= 0 or s >= 1:
        return (opp + 400) if s >= 1 else (opp - 400), 0
    elo = opp - 400 * math.log10(1.0 / s - 1.0)
    return elo, 400 / math.sqrt(n)


def run_match(sf_depth, games=50, movetime=20):
    sf_elo = SF_DEPTH_ELO.get(sf_depth, 2000)

    our = EngineProcess(ENGINE)
    sf = EngineProcess(STOCKFISH)
    our.handshake()
    sf.handshake()

    w = d = l = 0
    illegal = 0
    pgns = []

    print(f"\n{'='*60}")
    print(f"ChessAI vs Stockfish depth={sf_depth} (~{sf_elo} Elo)")
    print(f"Time: {movetime}ms/move  Games: {games}")
    print(f"{'='*60}")

    for g in range(games):
        if g % 2 == 0:
            score, moves, is_illegal = play_one_game(our, sf, sf_depth, movetime, our_white=True)
        else:
            inv_score, moves, is_illegal = play_one_game(our, sf, sf_depth, movetime, our_white=False)
            score = 1.0 - inv_score

        if is_illegal:
            illegal += 1
            result_str = "0-1"  # forfeit
            l += 1
        elif score == 1.0:
            w += 1
            result_str = "1-0"
        elif score == 0.5:
            d += 1
            result_str = "1/2-1/2"
        else:
            l += 1
            result_str = "0-1"

        pgn = format_pgn(moves, result_str)
        pgns.append((pgn, result_str))
        print(f"\n{pgn}")

        total = w + d + l
        score_pct = (w + d / 2) / total * 100 if total > 0 else 0
        our_elo, err = estimate_elo(w, d, l, sf_elo)

        flag = " ILLEGAL!" if is_illegal else ""
        sys.stdout.write(f"\rGame {g+1:3d}/{games} | "
                        f"W:{w:2d} L:{l:2d} D:{d:2d} | "
                        f"{score_pct:5.1f}% | ChessAI ≈ {our_elo:5.0f} ± {err:.0f}"
                        f"{flag}")
        sys.stdout.flush()

    # Save PGNs
    pgn_path = os.path.join(os.path.dirname(__file__), "..", "games.pgn")
    with open(pgn_path, "w") as f:
        for i, (pgn, res) in enumerate(pgns):
            f.write(f"[Game \"{i+1}\"]\n")
            f.write(f"[Result \"{res}\"]\n")
            f.write("\n")
            f.write(pgn + "\n\n")
    print(f"\nPGNs saved to {pgn_path}")

    # Clean up
    our.close()
    sf.close()

    our_elo, err = estimate_elo(w, d, l, sf_elo)
    print(f"\n{'-'*60}")
    print(f"Results vs Stockfish depth={sf_depth} (~{sf_elo} Elo):")
    print(f"  +{w} -{l} ={d} | Score: {(w+d/2)/(w+d+l)*100:.1f}%")
    if illegal:
        print(f"  Illegal moves by ChessAI: {illegal}")
    print(f"  ChessAI ≈ {our_elo:.0f} ± {err:.0f} Elo")
    print(f"  95% CI: [{our_elo-err*2:.0f}, {our_elo+err*2:.0f}]")
    return our_elo


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--games", type=int, default=50)
    p.add_argument("--movetime", type=int, default=1000)
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    if args.quick:
        results = []
        for d, n in [(1, 10), (2, 10), (3, 10)]:
            elo = run_match(d, games=n, movetime=args.movetime)
            results.append((d, elo, n))
        total_n = sum(n for _, _, n in results)
        avg = sum(e * n for _, e, n in results) / total_n
        print(f"\n{'='*60}")
        print(f"Quick estimate: ChessAI ≈ {avg:.0f} Elo (weighted avg across depths)")
    elif args.sweep:
        for d in [1, 2, 3, 4, 6]:
            run_match(d, games=args.games, movetime=args.movetime)
    else:
        run_match(args.depth, args.games, args.movetime)
