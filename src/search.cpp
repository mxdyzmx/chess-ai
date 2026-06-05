#include "search.h"
#include "eval.h"
#include <algorithm>
#include <iostream>
#include <cstring>
#include <cmath>

namespace engine {

// Search context (passed through recursive search instead of Search class members)
struct SearchContext {
    Board& board;
    SearchStats& stats;
    std::atomic<bool>& stop;
    int64_t start_time;
    int64_t time_limit;
    int sel_depth;
    uint64_t nodes;
    uint64_t qnodes;
    int ply;
    Move pv_table[TT_MAX_PLY * TT_MAX_PLY];
    int pv_length[TT_MAX_PLY];
};

// Piece values
static const int PIECE_VALS[] = { 100, 300, 320, 500, 900, 20000 };

// LMR reduction table
static int lmr_reduction(int depth, int moves_searched) {
    if (depth < 2) return 0;
    if (moves_searched < 3) return 0;
    int reduction = (int)(0.5 + log(depth) * log(moves_searched) / 2.5);
    return std::min(reduction, depth - 1);
}

// Forward declarations
static int alpha_beta_root(SearchContext& ctx, int alpha, int beta, int depth);
static int quiescence(SearchContext& ctx, int alpha, int beta);

// ============== TranspositionTable ==============

void TranspositionTable::resize(size_t mb) {
    delete[] entries_;
    size_t bytes = mb * 1024ULL * 1024ULL;
    size_t num_entries = bytes / sizeof(TTEntry);
    size_t power = 1;
    while (power * 2 <= num_entries) power *= 2;
    size_ = power;
    mask_ = power - 1;
    entries_ = new TTEntry[size_];
    clear();
}

void TranspositionTable::clear() {
    if (entries_) memset(entries_, 0, size_ * sizeof(TTEntry));
}

void TranspositionTable::prefetch(uint64_t hash) const {
    if (entries_) __builtin_prefetch(&entries_[hash & mask_]);
}

TTEntry* TranspositionTable::probe(uint64_t hash) const {
    if (!entries_) return nullptr;
    size_t idx = hash & mask_;
    if (entries_[idx].hash == hash) return &entries_[idx];
    return nullptr;
}

void TranspositionTable::store(uint64_t hash, int score, Move move, int depth, int flag, int ply) {
    if (!entries_) return;
    size_t idx = hash & mask_;
    TTEntry& entry = entries_[idx];

    if (score > TT_MAX_SCORE) score += ply;
    else if (score < TT_MIN_SCORE) score -= ply;

    entry.hash = hash;
    entry.score = (int16_t)score;
    entry.move = move.data;
    entry.depth = (uint8_t)depth;
    entry.flag = (uint8_t)flag;
    entry.age = age_;
}

size_t TranspositionTable::used() const {
    size_t count = 0;
    for (size_t i = 0; i < size_; i++)
        if (entries_[i].hash != 0) count++;
    return count;
}

// ============== Quiescence Search ==============

static int quiescence(SearchContext& ctx, int alpha, int beta) {
    if (ctx.stop.load()) return 0;
    ctx.qnodes++;

    // Stand pat
    int stand_pat = evaluate(ctx.board);
    if (stand_pat >= beta) return beta;
    if (alpha < stand_pat) alpha = stand_pat;

    // Generate captures
    MoveList list;
    generate_captures(ctx.board, list);

    // Order captures by MVV-LVA and search
    for (int i = 0; i < list.size; i++) {
        if (ctx.stop.load()) break;
        // Bubble sort step
        for (int j = i + 1; j < list.size; j++) {
            int vi = ctx.board.piece_type_on(list.moves[i].to());
            int vj = ctx.board.piece_type_on(list.moves[j].to());
            if (vi < vj) {
                Move t = list.moves[i]; list.moves[i] = list.moves[j]; list.moves[j] = t;
            }
        }

        Move m = list.moves[i];
        PieceType victim = ctx.board.piece_type_on(m.to());

        // Delta pruning
        if (stand_pat + (victim <= PT_KING ? PIECE_VALS[victim] : 0) + 200 < alpha)
            continue;

        if (!ctx.board.make_move(m)) continue;
        if (ctx.board.in_check()) {
            ctx.board.unmake_move();
            continue;
        }

        int score = -quiescence(ctx, -beta, -alpha);
        ctx.board.unmake_move();

        if (score >= beta) return beta;
        if (score > alpha) alpha = score;
    }

    return alpha;
}

// ============== Alpha-Beta Search ==============

static int alpha_beta_root(SearchContext& ctx, int alpha, int beta, int depth) {
    int old_ply = ctx.ply;
    ctx.ply++;

    // Stop check
    if (ctx.stop.load()) { ctx.ply--; return 0; }

    // Draw detection
    if (ctx.board.is_draw()) { ctx.ply--; return 0; }

    bool in_check = ctx.board.in_check();
    if (in_check) depth++; // Check extension

    // Mate distance pruning
    if (alpha < -TT_MATE_SCORE + ctx.ply) alpha = -TT_MATE_SCORE + ctx.ply;
    if (beta > TT_MATE_SCORE - ctx.ply) beta = TT_MATE_SCORE - ctx.ply;
    if (alpha >= beta) { ctx.ply--; return alpha; }

    // Base case: quiescence search
    if (depth <= 0) {
        int score = quiescence(ctx, alpha, beta);
        ctx.ply--;
        return score;
    }

    ctx.nodes++;
    if (ctx.ply > ctx.sel_depth) ctx.sel_depth = ctx.ply;

    // Transposition table probe
    TTEntry* tt_entry = Search::tt().probe(ctx.board.hash());
    Move tt_move(0);
    if (tt_entry) {
        ctx.stats.tt_hits++;
        tt_move = Move(tt_entry->move);
        // Validate TT move: from-square must have a piece of the current side
        if (tt_move.data != 0) {
            int from_sq = tt_move.from();
            int to_sq = tt_move.to();
            Color side = Color(ctx.board.side());
            Color enemy = Color(1 - side);
            uint64_t from_bb = sq_bb(from_sq);
            if (!(ctx.board.bb_color(side) & from_bb)) {
                tt_move = Move(0); // Invalid TT move, discard
            }
            // Validate by move type
            else if (tt_move.is_castle()) {
                // Verify king is on correct square and castling rights exist
                int king_from = side == WHITE ? 4 : 60; // e1/e8
                int k_rook = side == WHITE ? 7 : 63;   // h1/h8
                int q_rook = side == WHITE ? 0 : 56;   // a1/a8
                bool kingside = to_sq > from_sq;
                int expected_rook_from = kingside ? k_rook : q_rook;
                uint8_t needed_right = kingside ? (side == WHITE ? CASTLE_WHITE_KING : CASTLE_BLACK_KING)
                                                : (side == WHITE ? CASTLE_WHITE_QUEEN : CASTLE_BLACK_QUEEN);
                if (from_sq != king_from || !(ctx.board.castling_rights() & needed_right)
                    || !(ctx.board.bb_color(side) & sq_bb(expected_rook_from))
                    || ctx.board.piece_type_on(expected_rook_from) != PT_ROOK) {
                    tt_move = Move(0); // Illegal castle
                }
            }
            else if (tt_move.is_enpassant()) {
                // Verify EP square is set, and enemy pawn is on the captured square
                int ep_sq = ctx.board.ep_square();
                int ep_capture_sq = to_sq + (side == WHITE ? -8 : 8);
                if (ep_sq < 0 || ep_sq != to_sq
                    || !(ctx.board.pawns(enemy) & sq_bb(ep_capture_sq))) {
                    tt_move = Move(0); // Illegal EP
                }
            }
            else if (tt_move.is_promotion()) {
                // Verify from-square has a pawn
                if (ctx.board.piece_type_on(from_sq) != PT_PAWN) {
                    tt_move = Move(0);
                }
                // Also do non-capture/capture to-square checks
                if (tt_move.data != 0 && !tt_move.is_capture()) {
                    if (ctx.board.bb_color(side) & sq_bb(to_sq))
                        tt_move = Move(0);
                }
                else if (tt_move.data != 0 && tt_move.is_capture()) {
                    if (!(ctx.board.bb_color(enemy) & sq_bb(to_sq)))
                        tt_move = Move(0);
                }
            }
            else if (!tt_move.is_capture()) {
                // Non-capture, non-castle, non-EP, non-promotion: check no friendly on to-square
                if (ctx.board.bb_color(side) & sq_bb(to_sq)) {
                    tt_move = Move(0);
                }
                // ALSO check no enemy on to-square (captures must have capture flag set)
                if (tt_move.data != 0 && (ctx.board.bb_color(enemy) & sq_bb(to_sq))) {
                    tt_move = Move(0);
                }
            }
            else {
                // Capture (non-EP): check enemy on to-square
                if (!(ctx.board.bb_color(enemy) & sq_bb(to_sq))) {
                    tt_move = Move(0);
                }
            }
        }
        if (tt_entry->depth >= depth) {
            int tt_score = tt_entry->score;
            if (tt_score > TT_MAX_SCORE) tt_score -= ctx.ply;
            else if (tt_score < TT_MIN_SCORE) tt_score += ctx.ply;

            if (tt_entry->flag == TT_EXACT) { ctx.ply--; return tt_score; }
            if (tt_entry->flag == TT_ALPHA && tt_score <= alpha) { ctx.ply--; return alpha; }
            if (tt_entry->flag == TT_BETA && tt_score >= beta) { ctx.ply--; return beta; }
        }
    }

    ctx.pv_length[ctx.ply] = ctx.ply;

    // Null move pruning
    if (depth >= 3 && !in_check) {
        ctx.board.make_null_move();

        int R = 2 + std::min(depth, 8) / 5;
        int null_score = -alpha_beta_root(ctx, -beta, -beta + 1, depth - R - 1);

        ctx.board.unmake_null_move();

        if (null_score >= beta) {
            ctx.stats.null_cutoffs++;
            ctx.ply--;
            return null_score;
        }
    }

    // Generate moves
    MoveList list;
    generate_moves(ctx.board, list);

    // Checkmate/stalemate
    int legal_moves = 0;
    if (list.size == 0) {
        ctx.ply--;
        return ctx.board.in_check() ? (-TT_MATE_SCORE + ctx.ply) : 0;
    }

    Move best_move(0);
    int best_score = -TT_MATE_SCORE;
    int flag = TT_ALPHA;
    int moves_searched = 0;

    for (int i = 0; i < list.size; i++) {
        // Pick move with best score
        int best_idx = i;
        int best_scr = -99999999;
        for (int j = i; j < list.size; j++) {
            Move mj = list.moves[j];
            int scr = 0;

            if (mj == tt_move) scr += 10000000;
            if (mj.is_capture()) {
                int victim = ctx.board.piece_type_on(mj.to());
                int attacker = ctx.board.piece_type_on(mj.from());
                scr += 1000000 + (victim >= 0 && victim <= PT_KING ? PIECE_VALS[victim] * 10 : 0)
                       - (attacker >= 0 && attacker <= PT_KING ? PIECE_VALS[attacker] : 0);
            }
            if (mj.is_promotion()) scr += 500000 + mj.promotion_type() * 10000;
            if (killer_table[old_ply][0] == mj) scr += 60000;
            if (killer_table[old_ply][1] == mj) scr += 50000;

            Color us = Color(ctx.board.side());
            scr += history_table[us][mj.from()][mj.to()] / 50;

            if (scr > best_scr) { best_scr = scr; best_idx = j; }
        }
        if (best_idx != i) {
            Move t = list.moves[i]; list.moves[i] = list.moves[best_idx]; list.moves[best_idx] = t;
        }

        // Periodic stop check for responsive time management
        if ((i & 7) == 0 && ctx.stop.load()) break;

        Move m = list.moves[i];

        // Make move
        if (!ctx.board.make_move(m)) continue;
        // Skip illegal
        if (ctx.board.in_check()) {
            ctx.board.unmake_move();
            continue;
        }
        legal_moves++;

        bool gives_check = ctx.board.in_check();

        // Reset child ply's PV length to prevent stale PV from previous siblings
        ctx.pv_length[ctx.ply] = ctx.ply;

        // Principal Variation Search
        int score;
        if (moves_searched == 0) {
            // First move: full window
            score = -alpha_beta_root(ctx, -beta, -alpha, depth - 1);
        } else {
            // Late Move Reduction
            int reduction = 0;
            if (depth >= 3 && moves_searched >= 3 && !m.is_capture() && !m.is_promotion() && !gives_check && !in_check) {
                reduction = lmr_reduction(depth, moves_searched);
            }

            // Null window search (possibly reduced)
            if (reduction > 0) {
                score = -alpha_beta_root(ctx, -alpha - 1, -alpha, depth - 1 - reduction);
            } else {
                score = alpha + 1; // Force full search
            }

            // If LMR failed high or no LMR, do full null-window search
            if (score > alpha) {
                score = -alpha_beta_root(ctx, -alpha - 1, -alpha, depth - 1);
            }

            // PVS re-search
            if (score > alpha && score < beta) {
                score = -alpha_beta_root(ctx, -beta, -alpha, depth - 1);
            }
        }

        ctx.board.unmake_move();

        if (score > best_score) {
            best_score = score;
            best_move = m;

            if (score > alpha) {
                alpha = score;
                flag = TT_EXACT;

                // Update PV
                ctx.pv_table[old_ply * TT_MAX_PLY] = m;
                // Copy child's PV entries (if any)
                {
                    int pv_idx = ctx.ply * TT_MAX_PLY;
                    int num_child = ctx.pv_length[ctx.ply] - ctx.ply;
                    for (int j = 0; j < num_child; j++) {
                        ctx.pv_table[old_ply * TT_MAX_PLY + 1 + j] = ctx.pv_table[pv_idx + j];
                    }
                }
                ctx.pv_length[old_ply] = ctx.pv_length[ctx.ply];

                if (score >= beta) {
                    flag = TT_BETA;

                    // Update killer moves
                    if (!m.is_capture()) {
                        killer_table[old_ply][1] = killer_table[old_ply][0];
                        killer_table[old_ply][0] = m;

                        // Update history
                        Color us = Color(ctx.board.side());
                        history_table[us][m.from()][m.to()] += depth * depth;
                        if (history_table[us][m.from()][m.to()] > MAX_HISTORY) {
                            for (int a = 0; a < 2; a++)
                                for (int b = 0; b < 64; b++)
                                    for (int c = 0; c < 64; c++)
                                        history_table[a][b][c] /= 2;
                        }
                    }
                    break;
                }
            }
        }
    }

    // If no legal moves, it's stalemate or checkmate
    if (legal_moves == 0) {
        ctx.ply--;
        return ctx.board.in_check() ? (-TT_MATE_SCORE + old_ply) : 0;
    }

    // Store in TT
    Search::tt().store(ctx.board.hash(), best_score, best_move, depth, flag, old_ply);

    ctx.ply--;
    return best_score;
}

// ============== Search Class ==============

Search::Search()
    : prev_estimate_(0) {}

int64_t Search::get_time_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

void Search::search() {
    searching_.store(true);
    stop_.store(false);
    stats_.reset();

    best_move_ = Move(0);
    best_score_ = 0;
    start_time_ = get_time_ms();
    time_limit_ = 0;
    prev_estimate_ = 0;

    // Reset search state
    memset(history_table, 0, sizeof(history_table));
    memset(killer_table, 0, sizeof(killer_table));

    // Time management
    if (params_.movetime > 0) {
        time_limit_ = params_.movetime;
    } else if (params_.wtime > 0 || params_.btime > 0) {
        int our_time = root_.is_white_to_move() ? params_.wtime : params_.btime;
        int our_inc = root_.is_white_to_move() ? params_.winc : params_.binc;
        int moves_to_go = params_.movestogo > 0 ? params_.movestogo : 40;
        time_limit_ = std::max<int64_t>(our_time / moves_to_go + our_inc / 2, 50);
        time_limit_ = std::min(time_limit_, (int64_t)(our_time * 0.33));
    } else if (!params_.infinite) {
        time_limit_ = 10000;
    } else {
        time_limit_ = INT64_MAX;
    }

    // Save FEN for PV string validation
    std::string saved_fen = root_.fen();

    // Setup search context
    SearchContext ctx = {
        root_, stats_, stop_, start_time_, time_limit_,
        0, 0, 0, 0,
        {}, {}
    };
    memset(ctx.pv_table, 0, sizeof(ctx.pv_table));
    memset(ctx.pv_length, 0, sizeof(ctx.pv_length));

    // Iterative deepening
    int alpha = -TT_MATE_SCORE;
    int beta = TT_MATE_SCORE;

    for (int depth = 1; depth <= params_.depth; depth++) {
        if (stop_.load()) break;

        ctx.ply = 0;
        ctx.nodes = 0;
        ctx.qnodes = 0;
        ctx.sel_depth = 0;
        memset(ctx.pv_length, 0, sizeof(ctx.pv_length));
        memset(ctx.pv_table, 0, sizeof(ctx.pv_table));

        // Aspiration windows (disabled for debugging)
        alpha = -TT_MATE_SCORE;
        beta  = TT_MATE_SCORE;

        int score = alpha_beta_root(ctx, alpha, beta, depth);

        // Aspiration fail-low: re-search with full window
        if (score <= alpha || score >= beta) {
            alpha = -TT_MATE_SCORE;
            beta = TT_MATE_SCORE;
            score = alpha_beta_root(ctx, alpha, beta, depth);
        }

        prev_estimate_ = score;
        stats_.depth_reached = depth;
        stats_.nodes = ctx.nodes;
        stats_.qnodes = ctx.qnodes;
        stats_.sel_depth = ctx.sel_depth;

        // Extract best move from PV
        if (ctx.pv_length[0] > 0) {
            best_move_ = ctx.pv_table[0];
        }
        best_score_ = score;

        int64_t elapsed = get_time_ms() - start_time_;
        stats_.total_time_ms = elapsed;

        // Build PV string (validate moves on a working copy of the board)
        std::string pv_str;
        Board pv_board;
        pv_board.set_fen(saved_fen);
        int pv_len = ctx.pv_length[0];
        for (int i = 0; i < pv_len; i++) {
            Move pv_move = ctx.pv_table[i];
            if (pv_move.data == 0) break;
            // Verify move is legal on the board
            int f = pv_move.from(), t = pv_move.to();
            if (!(pv_board.bb_color(pv_board.side()) & sq_bb(f))) {
                break; // Illegal PV move - stop here
            }
            pv_board.make_move(pv_move);
            pv_str += char('a' + (f & 7));
            pv_str += char('1' + (f >> 3));
            pv_str += char('a' + (t & 7));
            pv_str += char('1' + (t >> 3));
            if (pv_move.is_promotion()) {
                const char* p = "nbrq";
                pv_str += p[pv_move.promotion_type() - PT_KNIGHT];
            }
            pv_str += ' ';
        }

        // UCI info - always output from white's perspective
        int uci_score = root_.is_white_to_move() ? score : -score;
        std::cout << "info depth " << depth
                  << " score cp " << uci_score
                  << " nodes " << ctx.nodes
                  << " nps " << (elapsed > 0 ? (ctx.nodes * 1000 / elapsed) : 0)
                  << " time " << elapsed
                  << " pv " << pv_str
                  << std::endl;

        // Progress bar
        std::cout << "progress: [";
        int bar_pos = depth * 20 / params_.depth;
        for (int b = 0; b < 20; b++)
            std::cout << (b < bar_pos ? '#' : '.');
        std::cout << "] " << depth << "/" << params_.depth << std::endl;

        if (time_over()) {
            stop_.store(true);
            break;
        }
    }

    stats_.total_time_ms = get_time_ms() - start_time_;

    // Output best move
    if (best_move_.data != 0) {
        int f = best_move_.from(), t = best_move_.to();
        std::cout << "bestmove ";
        std::cout << char('a' + (f & 7)) << char('1' + (f >> 3));
        std::cout << char('a' + (t & 7)) << char('1' + (t >> 3));
        if (best_move_.is_promotion()) {
            const char* p = "nbrq";
            std::cout << p[best_move_.promotion_type() - PT_KNIGHT];
        }
        std::cout << std::endl;
    }

    searching_.store(false);
}

bool Search::time_over() {
    if (stop_.load()) return true;
    if (params_.infinite) return false;
    if (time_limit_ <= 0) return true;
    return (get_time_ms() - start_time_) >= time_limit_;
}

} // namespace engine
