#!/usr/bin/env python3
"""Play matches between ChessAI and Stockfish. Uses depth-limited Stockfish."""
import subprocess, sys, os, math, time, shutil, threading, queue

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

    def get_bestmove(self, moves, movetime=20, depth=None):
        """Send position + go, return best move string."""
        fen = "startpos" if not moves else "startpos moves " + " ".join(moves)
        self.send(f"position {fen}")
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


def analyze_position(sf, moves):
    """Analyze a position with Stockfish. Returns (cp, is_mate, mate_val, bestmove)."""
    fen = "startpos" if not moves else "startpos moves " + " ".join(moves)
    sf.send(f"position {fen}")
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


def play_one_game(our, sf, sf_depth, movetime):
    """Play one game: our engine white, Stockfish black. Returns (score, moves_list)."""
    moves = []
    for ply in range(200):
        if ply % 2 == 0:
            bm = our.get_bestmove(moves, movetime=movetime)
        else:
            bm = sf.get_bestmove(moves, movetime=movetime, depth=sf_depth)

        if bm is None or bm == "(none)":
            break
        moves.append(bm)

    n = len(moves)
    sf_result = analyze_position(sf, moves)
    cp, is_mate, mate_val, final_bm = sf_result

    # Determine result from our engine's perspective
    if final_bm == "(none)":
        stm = n % 2  # 0=white to move, 1=black to move
        if is_mate:
            # Side to move is checkmated
            score = 0.0 if stm == 0 else 1.0  # white mated -> we lose, black mated -> we win
        else:
            score = 0.5  # stalemate
    elif is_mate:
        stm = n % 2
        if mate_val > 0:
            # Side to move delivers mate
            score = 1.0 if stm == 0 else 0.0  # white mates -> we win, black mates -> we lose
        else:
            # Side to move is mated
            score = 0.0 if stm == 0 else 1.0
    elif n >= 190:
        score = 0.5  # max ply limit
    elif cp > 300:
        score = 1.0  # Stockfish says white is winning (we played white)
    elif cp < -300:
        score = 0.0  # Stockfish says black is winning (we played white)
    else:
        score = 0.5

    return score, moves


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
    pgns = []

    print(f"\n{'='*60}")
    print(f"ChessAI vs Stockfish depth={sf_depth} (~{sf_elo} Elo)")
    print(f"Time: {movetime}ms/move  Games: {games}")
    print(f"{'='*60}")

    for g in range(games):
        if g % 2 == 0:
            score, moves = play_one_game(our, sf, sf_depth, movetime)
        else:
            # Swap colors: Stockfish plays white, ChessAI plays black
            inv_score, moves = play_one_game(sf, our, sf_depth, movetime)
            score = 1.0 - inv_score

        if score == 1.0:
            w += 1
            result_str = "1-0"
        elif score == 0.5:
            d += 1
            result_str = "1/2-1/2"
        else:
            l += 1
            result_str = "0-1"

        pgn = format_pgn(moves, result_str)
        pgns.append(pgn)

        total = w + d + l
        score_pct = (w + d / 2) / total * 100
        our_elo, err = estimate_elo(w, d, l, sf_elo)

        sys.stdout.write(f"\rGame {g+1:3d}/{games} | "
                        f"W:{w:2d} L:{l:2d} D:{d:2d} | "
                        f"{score_pct:5.1f}% | ChessAI ≈ {our_elo:5.0f} ± {err:.0f}")
        sys.stdout.flush()

    # Save PGNs
    pgn_path = os.path.join(os.path.dirname(__file__), "..", "games.pgn")
    with open(pgn_path, "w") as f:
        for i, pgn in enumerate(pgns):
            f.write(f"[Game \"{i+1}\"]\n")
            f.write(f"[Result \"{result_str if score == 1.0 else (result_str if score == 0.5 else result_str)}\"]\n")
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
    print(f"  ChessAI ≈ {our_elo:.0f} ± {err:.0f} Elo")
    print(f"  95% CI: [{our_elo-err*2:.0f}, {our_elo+err*2:.0f}]")
    return our_elo


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--games", type=int, default=50)
    p.add_argument("--movetime", type=int, default=20)
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
