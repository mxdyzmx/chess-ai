#!/usr/bin/env python3
"""Play matches between ChessAI and Stockfish. Uses depth-limited Stockfish."""
import subprocess, sys, os, math, select, time, signal

ENGINE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chessai")
STOCKFISH = "/usr/local/bin/stockfish"

# Approximate Stockfish Elo at fixed depths (bullet)
SF_DEPTH_ELO = {1: 1550, 2: 1800, 3: 1950, 4: 2100, 5: 2200, 6: 2300,
                8: 2500, 10: 2700, 12: 2900}


def start_process(path):
    proc = subprocess.Popen([path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, bufsize=1)
    return proc


def send(proc, cmd):
    proc.stdin.write(cmd + "\n")
    proc.stdin.flush()


def read_line(proc, timeout=30):
    r, _, _ = select.select([proc.stdout], [], [], timeout)
    return proc.stdout.readline().strip() if r else None


def wait_for(proc, target, timeout=30):
    end = time.time() + timeout
    while time.time() < end:
        line = read_line(proc, min(1, timeout))
        if line and target in line:
            return line
    return None


def uci_handshake(proc):
    send(proc, "uci")
    wait_for(proc, "uciok", 5)
    send(proc, "isready")
    wait_for(proc, "readyok", 5)


def get_bestmove(proc, moves, movetime=20, depth=None):
    fen = "startpos" if not moves else "startpos moves " + " ".join(moves)
    send(proc, f"position {fen}")
    cmd = "go"
    if depth:
        cmd += f" depth {depth}"
    else:
        cmd += f" movetime {movetime}"
    send(proc, cmd)
    while True:
        line = read_line(proc, 30)
        if line and line.startswith("bestmove"):
            parts = line.split()
            return parts[1] if len(parts) > 1 else "(none)"
        if line is None:
            return None


def analyze_position(proc, moves):
    """Analyze a position and return (cp, is_mate, mate_val, bestmove)."""
    fen = "startpos" if not moves else "startpos moves " + " ".join(moves)
    send(proc, f"position {fen}")
    send(proc, "go depth 8")
    cp, is_mate, mate_val, bm = 0, False, 0, "(none)"
    while True:
        line = read_line(proc, 15)
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


def play_one_game(our_proc, sf_proc, sf_depth, movetime):
    """Play one game: our engine white, Stockfish black. Returns 1/0.5/0."""
    moves = []
    for ply in range(200):
        if ply % 2 == 0:
            bm = get_bestmove(our_proc, moves, movetime=movetime)
        else:
            bm = get_bestmove(sf_proc, moves, movetime=movetime, depth=sf_depth)

        if bm is None or bm == "(none)":
            break
        moves.append(bm)

    # Determine result
    n = len(moves)
    cp, is_mate, mate_val, final_bm = analyze_position(sf_proc, moves)

    if final_bm == "(none)":
        # Side to move has no legal moves
        stm = n % 2
        if is_mate:
            return 1.0 if stm == 1 else 0.0
        return 0.5  # stalemate

    if is_mate:
        stm = n % 2
        if mate_val > 0:
            return 1.0 if stm == 1 else 0.0
        else:
            return 1.0 if stm == 0 else 0.0

    if n >= 190:
        return 0.5

    # Game stopped early (timeout, crash). Use evaluation.
    if cp > 300:
        return 1.0
    elif cp < -300:
        return 0.0
    return 0.5


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

    our_proc = start_process(ENGINE)
    sf_proc = start_process(STOCKFISH)
    uci_handshake(our_proc)
    uci_handshake(sf_proc)

    w = d = l = 0
    print(f"\n{'='*60}")
    print(f"ChessAI vs Stockfish depth={sf_depth} (~{sf_elo} Elo)")
    print(f"Time: {movetime}ms/move  Games: {games}")
    print(f"{'='*60}")

    for g in range(games):
        if g % 2 == 0:
            result = play_one_game(our_proc, sf_proc, sf_depth, movetime)
        else:
            result = 1.0 - play_one_game(sf_proc, our_proc, sf_depth, movetime)

        if result == 1.0:
            w += 1
        elif result == 0.5:
            d += 1
        else:
            l += 1

        total = w + d + l
        score_pct = (w + d / 2) / total * 100
        our_elo, err = estimate_elo(w, d, l, sf_elo)

        sys.stdout.write(f"\rGame {g+1:3d}/{games} | "
                        f"W:{w:2d} L:{l:2d} D:{d:2d} | "
                        f"{score_pct:5.1f}% | ChessAI ≈ {our_elo:5.0f} ± {err:.0f}")
        sys.stdout.flush()

    # Clean up
    for p in [our_proc, sf_proc]:
        try:
            send(p, "quit")
            p.wait(timeout=2)
        except:
            p.kill()

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
        # Quick estimate: sweep depths 1,2,3,4 with few games each
        results = []
        for d, n in [(1, 20), (2, 20), (3, 20), (4, 30)]:
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
