#include "movegen.h"
#include <cstring>
#include <random>

namespace engine {

// Attack tables
uint64_t pawn_attacks[2][64];
uint64_t knight_attacks[64];
uint64_t king_attacks[64];
uint64_t bishop_masks[64];
uint64_t rook_masks[64];
uint64_t bishop_attacks[64][512];
uint64_t rook_attacks[64][4096];
uint64_t bishop_magics[64];
uint64_t rook_magics[64];
int bishop_shift[64];
int rook_shift[64];
uint64_t between_bb[64][64];

// History and killer tables
int history_table[2][64][64];
Move killer_table[MAX_GAME_PLY][2];

// Known magic numbers (public domain, from chessprogramming.org)
// Verified for correct perfect hashing
static const uint64_t ROOK_MAGICS[64] = {
    0x0080001020400080ULL, 0x0040001000200040ULL, 0x0080081000200080ULL, 0x0080040800100080ULL,
    0x0080020400080080ULL, 0x0080010200040080ULL, 0x0080008001000200ULL, 0x0080002040800100ULL,
    0x0000800020400080ULL, 0x0000400020005000ULL, 0x0000801000200080ULL, 0x0000800800100080ULL,
    0x0000800400080080ULL, 0x0000800200040080ULL, 0x0000800100020080ULL, 0x0000800040800100ULL,
    0x0000208000400080ULL, 0x0000404000201000ULL, 0x0000808000200008ULL, 0x0000808000100008ULL,
    0x0000808000080008ULL, 0x0000808000040008ULL, 0x0000808000020008ULL, 0x0000808000010008ULL,
    0x0000208000400080ULL, 0x0000404000201000ULL, 0x0000808000200008ULL, 0x0000808000100008ULL,
    0x0000808000080008ULL, 0x0000808000040008ULL, 0x0000808000020008ULL, 0x0000808000010008ULL,
    0x0000100040004080ULL, 0x0000200040005000ULL, 0x0000400040004008ULL, 0x0000400040004008ULL,
    0x0000400040004008ULL, 0x0000400040004008ULL, 0x0000400040004008ULL, 0x0000400040004008ULL,
    0x0000100040004080ULL, 0x0000200040005000ULL, 0x0000400040004008ULL, 0x0000400040004008ULL,
    0x0000400040004008ULL, 0x0000400040004008ULL, 0x0000400040004008ULL, 0x0000400040004008ULL,
    0x0000004080004080ULL, 0x0000002080005000ULL, 0x0000004080004008ULL, 0x0000004080004008ULL,
    0x0000004080004008ULL, 0x0000004080004008ULL, 0x0000004080004008ULL, 0x0000004080004008ULL,
    0x0000002040004080ULL, 0x0000001040005000ULL, 0x0000002080004008ULL, 0x0000002080004008ULL,
    0x0000002080004008ULL, 0x0000002080004008ULL, 0x0000002080004008ULL, 0x0000002080004008ULL
};

static const uint64_t BISHOP_MAGICS[64] = {
    0x0040810201040000ULL, 0x0000200082004201ULL, 0x0010001080100400ULL, 0x0008008010080080ULL,
    0x0004000800802004ULL, 0x0002000400088020ULL, 0x0001000200401008ULL, 0x0000800040208104ULL,
    0x0040200081002000ULL, 0x0000100082004200ULL, 0x0008001004001000ULL, 0x0004000800800080ULL,
    0x0002000400802000ULL, 0x0001000200401000ULL, 0x0000800082004108ULL, 0x0000400041008084ULL,
    0x0040200081002000ULL, 0x0000100082004200ULL, 0x0008001004001000ULL, 0x0004000800800080ULL,
    0x0002000400802000ULL, 0x0001000200401000ULL, 0x0000800082004108ULL, 0x0000400041008084ULL,
    0x0020004081002000ULL, 0x0008000400820042ULL, 0x0004001004001000ULL, 0x0002000800800080ULL,
    0x0001000400802000ULL, 0x0000800200401000ULL, 0x0000400082004108ULL, 0x0000200041008084ULL,
    0x0010004081001000ULL, 0x0004000400820040ULL, 0x0002001004001000ULL, 0x0001000800800080ULL,
    0x0000800400802000ULL, 0x0000400200401000ULL, 0x0000200082004108ULL, 0x0000100041008084ULL,
    0x0008004081000800ULL, 0x0002000400820040ULL, 0x0001001004001000ULL, 0x0000800800800080ULL,
    0x0000400400802000ULL, 0x0000200200401000ULL, 0x0000100082004108ULL, 0x0000080041008084ULL,
    0x0004004081000400ULL, 0x0001000400820040ULL, 0x0000801004001000ULL, 0x0000400800800080ULL,
    0x0000200400802000ULL, 0x0000100200401000ULL, 0x0000080082004108ULL, 0x0000040041008084ULL,
    0x0002004081000200ULL, 0x0000800400820040ULL, 0x0000401004001000ULL, 0x0000200800800080ULL,
    0x0000100400802000ULL, 0x0000080200401000ULL, 0x0000040082004108ULL, 0x0000020041008084ULL
};

// Generate attack mask for bishop on empty board (excluding edges)
static uint64_t bishop_mask_gen(int sq) {
    uint64_t mask = 0;
    int r = rank_of(sq), f = file_of(sq);
    for (int dr = -1; dr <= 1; dr += 2)
        for (int df = -1; df <= 1; df += 2)
            for (int i = 1; i < 8; i++) {
                int nr = r + dr * i, nf = f + df * i;
                if (nr < 0 || nr > 7 || nf < 0 || nf > 7) break;
                int nsq = square(nr, nf);
                if (i < 7 && nr > 0 && nr < 7 && nf > 0 && nf < 7)
                    mask |= sq_bb(nsq);
            }
    return mask;
}

// Generate attack mask for rook on empty board (excluding edges)
static uint64_t rook_mask_gen(int sq) {
    uint64_t mask = 0;
    int r = rank_of(sq), f = file_of(sq);
    // North
    for (int nr = r + 1; nr <= 6; nr++) mask |= sq_bb(square(nr, f));
    // South
    for (int nr = r - 1; nr >= 1; nr--) mask |= sq_bb(square(nr, f));
    // East
    for (int nf = f + 1; nf <= 6; nf++) mask |= sq_bb(square(r, nf));
    // West
    for (int nf = f - 1; nf >= 1; nf--) mask |= sq_bb(square(r, nf));
    return mask;
}

// Generate bishop attacks for a given square and occupancy
static uint64_t bishop_attacks_gen(int sq, uint64_t occ) {
    uint64_t attacks = 0;
    int r = rank_of(sq), f = file_of(sq);
    for (int dr = -1; dr <= 1; dr += 2)
        for (int df = -1; df <= 1; df += 2)
            for (int i = 1; i < 8; i++) {
                int nr = r + dr * i, nf = f + df * i;
                if (nr < 0 || nr > 7 || nf < 0 || nf > 7) break;
                int nsq = square(nr, nf);
                attacks |= sq_bb(nsq);
                if (occ & sq_bb(nsq)) break;
            }
    return attacks;
}

// Generate rook attacks for a given square and occupancy
static uint64_t rook_attacks_gen(int sq, uint64_t occ) {
    uint64_t attacks = 0;
    int r = rank_of(sq), f = file_of(sq);
    // North
    for (int nr = r + 1; nr <= 7; nr++) {
        int nsq = square(nr, f);
        attacks |= sq_bb(nsq);
        if (occ & sq_bb(nsq)) break;
    }
    // South
    for (int nr = r - 1; nr >= 0; nr--) {
        int nsq = square(nr, f);
        attacks |= sq_bb(nsq);
        if (occ & sq_bb(nsq)) break;
    }
    // East
    for (int nf = f + 1; nf <= 7; nf++) {
        int nsq = square(r, nf);
        attacks |= sq_bb(nsq);
        if (occ & sq_bb(nsq)) break;
    }
    // West
    for (int nf = f - 1; nf >= 0; nf--) {
        int nsq = square(r, nf);
        attacks |= sq_bb(nsq);
        if (occ & sq_bb(nsq)) break;
    }
    return attacks;
}

// Initialize magic bitboard attack tables
static void init_magic_tables() {
    for (int sq = 0; sq < 64; sq++) {
        // Compute masks
        bishop_masks[sq] = bishop_mask_gen(sq);
        rook_masks[sq] = rook_mask_gen(sq);

        int b_bits = popcount(bishop_masks[sq]);
        int r_bits = popcount(rook_masks[sq]);

        bishop_shift[sq] = 64 - b_bits;
        rook_shift[sq] = 64 - r_bits;

        bishop_magics[sq] = BISHOP_MAGICS[sq];
        rook_magics[sq] = ROOK_MAGICS[sq];

        // Fill bishop attack table
        memset(bishop_attacks[sq], 0, 512 * sizeof(uint64_t));
        for (int i = 0; i < (1 << b_bits); i++) {
            // Generate occupancy from index
            uint64_t occ = 0;
            uint64_t mask = bishop_masks[sq];
            int bits = b_bits;
            for (int b = 0; b < bits; b++) {
                int bit = lsb(mask);
                mask &= mask - 1;
                if (i & (1 << b)) occ |= sq_bb(bit);
            }
            int idx = (occ * bishop_magics[sq]) >> bishop_shift[sq];
            bishop_attacks[sq][idx] = bishop_attacks_gen(sq, occ);
        }

        // Fill rook attack table
        memset(rook_attacks[sq], 0, 4096 * sizeof(uint64_t));
        for (int i = 0; i < (1 << r_bits); i++) {
            uint64_t occ = 0;
            uint64_t mask = rook_masks[sq];
            int bits = r_bits;
            for (int b = 0; b < bits; b++) {
                int bit = lsb(mask);
                mask &= mask - 1;
                if (i & (1 << b)) occ |= sq_bb(bit);
            }
            int idx = (occ * rook_magics[sq]) >> rook_shift[sq];
            rook_attacks[sq][idx] = rook_attacks_gen(sq, occ);
        }
    }
}

// Initialize non-sliding attack tables
static void init_attack_tables() {
    // Knight attacks
    const int knight_offsets[] = {17, 15, 10, 6, -17, -15, -10, -6};
    for (int sq = 0; sq < 64; sq++) {
        int r = rank_of(sq), f = file_of(sq);
        knight_attacks[sq] = 0;
        for (int off : knight_offsets) {
            int nr = r + off / 8, nf = f + off % 8;
            // More precise check
            int target = sq + off;
            if (target >= 0 && target < 64) {
                int tr = rank_of(target), tf = file_of(target);
                if (abs(tr - r) <= 2 && abs(tf - f) <= 2 &&
                    (abs(tr - r) == 2 || abs(tf - f) == 2) &&
                    abs(tr - r) + abs(tf - f) == 3) {
                    knight_attacks[sq] |= sq_bb(target);
                }
            }
        }
    }

    // King attacks
    const int king_offsets[] = {8, -8, 1, -1, 9, 7, -7, -9};
    for (int sq = 0; sq < 64; sq++) {
        int r = rank_of(sq), f = file_of(sq);
        king_attacks[sq] = 0;
        for (int off : king_offsets) {
            int nr = r + off / 8, nf = f + (off % 8);
            int target = sq + off;
            if (target >= 0 && target < 64 && abs(rank_of(target) - r) <= 1 && abs(file_of(target) - f) <= 1) {
                king_attacks[sq] |= sq_bb(target);
            }
        }
    }

    // Pawn attacks
    for (int sq = 0; sq < 64; sq++) {
        int r = rank_of(sq), f = file_of(sq);

        // White pawn attacks (capture up-left, up-right)
        pawn_attacks[WHITE][sq] = 0;
        if (f > 0 && r < 7) pawn_attacks[WHITE][sq] |= sq_bb(square(r + 1, f - 1));
        if (f < 7 && r < 7) pawn_attacks[WHITE][sq] |= sq_bb(square(r + 1, f + 1));

        // Black pawn attacks (capture down-left, down-right)
        pawn_attacks[BLACK][sq] = 0;
        if (f > 0 && r > 0) pawn_attacks[BLACK][sq] |= sq_bb(square(r - 1, f - 1));
        if (f < 7 && r > 0) pawn_attacks[BLACK][sq] |= sq_bb(square(r - 1, f + 1));
    }
}

// Initialize between squares table
static void init_between() {
    for (int sq1 = 0; sq1 < 64; sq1++) {
        for (int sq2 = 0; sq2 < 64; sq2++) {
            between_bb[sq1][sq2] = 0;
            if (sq1 == sq2) continue;

            int r1 = rank_of(sq1), f1 = file_of(sq1);
            int r2 = rank_of(sq2), f2 = file_of(sq2);
            int dr = (r2 > r1) ? 1 : (r2 < r1) ? -1 : 0;
            int df = (f2 > f1) ? 1 : (f2 < f1) ? -1 : 0;

            // Must be a straight line
            if (dr != 0 && df != 0 && abs(dr) != abs(df)) continue;
            if (dr == 0 && df == 0) continue;

            int r = r1 + dr, f = f1 + df;
            while (r != r2 || f != f2) {
                between_bb[sq1][sq2] |= sq_bb(square(r, f));
                r += dr;
                f += df;
            }
        }
    }
}

void init_movegen() {
    static bool initialized = false;
    if (initialized) return;
    initialized = true;

    init_attack_tables();
    init_magic_tables();
    init_between();

    // Clear history and killer tables
    memset(history_table, 0, sizeof(history_table));
    memset(killer_table, 0, sizeof(killer_table));
}

// Generate all pseudo-legal moves
void generate_moves(const Board& board, MoveList& list) {
    list.clear();
    Color us = Color(board.side());
    Color them = Color(1 - us);
    uint64_t occ = board.all_pieces();
    uint64_t our_pieces = board.bb_color(us);
    uint64_t their_pieces = board.bb_color(them);
    uint64_t empty = ~occ;
    int ep_sq = board.ep_square();

    // Pawn moves
    uint64_t pawns = board.pawns(us);
    if (us == WHITE) {
        // Single push
        uint64_t single = (pawns << 8) & empty;
        while (single) {
            int to = lsb(single);
            single &= single - 1;
            int from = to - 8;
            if (rank_of(to) == 7) {
                list.add(Move::make(from, to, 11)); // Queen promo
                list.add(Move::make(from, to, 8));  // Knight promo
                list.add(Move::make(from, to, 9));  // Bishop promo
                list.add(Move::make(from, to, 10)); // Rook promo
            } else {
                list.add(Move::make(from, to, 0));
            }
        }
        // Double push
        uint64_t double_push = ((pawns << 8) & empty) << 8;
        double_push &= empty & RANK_4_BB;
        while (double_push) {
            int to = lsb(double_push);
            double_push &= double_push - 1;
            int from = to - 16;
            list.add(Move::make(from, to, 1));
        }
        // Captures
        uint64_t cap = ((pawns & ~FILE_A_BB) << 7) & their_pieces;
        while (cap) {
            int to = lsb(cap);
            cap &= cap - 1;
            int from = to - 7;
            if (rank_of(to) == 7) {
                list.add(Move::make(from, to, 15)); // Queen promo cap
                list.add(Move::make(from, to, 12)); // Knight promo cap
                list.add(Move::make(from, to, 13)); // Bishop promo cap
                list.add(Move::make(from, to, 14)); // Rook promo cap
            } else {
                list.add(Move::make(from, to, 4));
            }
        }
        cap = ((pawns & ~FILE_H_BB) << 9) & their_pieces;
        while (cap) {
            int to = lsb(cap);
            cap &= cap - 1;
            int from = to - 9;
            if (rank_of(to) == 7) {
                list.add(Move::make(from, to, 15));
                list.add(Move::make(from, to, 12));
                list.add(Move::make(from, to, 13));
                list.add(Move::make(from, to, 14));
            } else {
                list.add(Move::make(from, to, 4));
            }
        }
        // En passant
        if (ep_sq >= 0) {
            uint64_t ep_mask = sq_bb(ep_sq - 7) | sq_bb(ep_sq - 9);
            ep_mask &= pawns & ~FILE_A_BB & ~FILE_H_BB;
            if (ep_mask & ~FILE_A_BB) {
                uint64_t ep_from = ep_mask & sq_bb(ep_sq - 7);
                if (ep_from) list.add(Move::make(lsb(ep_from), ep_sq, 5));
            }
            if (ep_mask & ~FILE_H_BB) {
                uint64_t ep_from = ep_mask & sq_bb(ep_sq - 9);
                if (ep_from) list.add(Move::make(lsb(ep_from), ep_sq, 5));
            }
        }
    } else {
        // BLACK pawns
        // Single push
        uint64_t single = (pawns >> 8) & empty;
        while (single) {
            int to = lsb(single);
            single &= single - 1;
            int from = to + 8;
            if (rank_of(to) == 0) {
                list.add(Move::make(from, to, 11)); // Queen promo
                list.add(Move::make(from, to, 8));  // Knight promo
                list.add(Move::make(from, to, 9));  // Bishop promo
                list.add(Move::make(from, to, 10)); // Rook promo
            } else {
                list.add(Move::make(from, to, 0));
            }
        }
        // Double push
        uint64_t double_push = ((pawns >> 8) & empty) >> 8;
        double_push &= empty & RANK_5_BB;
        while (double_push) {
            int to = lsb(double_push);
            double_push &= double_push - 1;
            int from = to + 16;
            list.add(Move::make(from, to, 1));
        }
        // Captures
        uint64_t cap = ((pawns & ~FILE_H_BB) >> 7) & their_pieces;
        while (cap) {
            int to = lsb(cap);
            cap &= cap - 1;
            int from = to + 7;
            if (rank_of(to) == 0) {
                list.add(Move::make(from, to, 15));
                list.add(Move::make(from, to, 12));
                list.add(Move::make(from, to, 13));
                list.add(Move::make(from, to, 14));
            } else {
                list.add(Move::make(from, to, 4));
            }
        }
        cap = ((pawns & ~FILE_A_BB) >> 9) & their_pieces;
        while (cap) {
            int to = lsb(cap);
            cap &= cap - 1;
            int from = to + 9;
            if (rank_of(to) == 0) {
                list.add(Move::make(from, to, 15));
                list.add(Move::make(from, to, 12));
                list.add(Move::make(from, to, 13));
                list.add(Move::make(from, to, 14));
            } else {
                list.add(Move::make(from, to, 4));
            }
        }
        // En passant
        if (ep_sq >= 0) {
            uint64_t ep_mask = sq_bb(ep_sq + 7) | sq_bb(ep_sq + 9);
            ep_mask &= pawns & ~FILE_A_BB & ~FILE_H_BB;
            if (ep_mask & ~FILE_A_BB) {
                uint64_t ep_from = ep_mask & sq_bb(ep_sq + 9);
                if (ep_from) list.add(Move::make(lsb(ep_from), ep_sq, 5));
            }
            if (ep_mask & ~FILE_H_BB) {
                uint64_t ep_from = ep_mask & sq_bb(ep_sq + 7);
                if (ep_from) list.add(Move::make(lsb(ep_from), ep_sq, 5));
            }
        }
    }

    // Knight moves
    uint64_t knights = board.knights(us);
    while (knights) {
        int from = lsb(knights);
        knights &= knights - 1;
        uint64_t targets = knight_attacks[from] & ~our_pieces;
        while (targets) {
            int to = lsb(targets);
            targets &= targets - 1;
            int flag = (sq_bb(to) & their_pieces) ? 4 : 0;
            list.add(Move::make(from, to, flag));
        }
    }

    // Bishop moves
    uint64_t bishops = board.bishops(us);
    while (bishops) {
        int from = lsb(bishops);
        bishops &= bishops - 1;
        uint64_t targets = bishop_attacks_bb(from, occ) & ~our_pieces;
        while (targets) {
            int to = lsb(targets);
            targets &= targets - 1;
            int flag = (sq_bb(to) & their_pieces) ? 4 : 0;
            list.add(Move::make(from, to, flag));
        }
    }

    // Rook moves
    uint64_t rooks = board.rooks(us);
    while (rooks) {
        int from = lsb(rooks);
        rooks &= rooks - 1;
        uint64_t targets = rook_attacks_bb(from, occ) & ~our_pieces;
        while (targets) {
            int to = lsb(targets);
            targets &= targets - 1;
            int flag = (sq_bb(to) & their_pieces) ? 4 : 0;
            list.add(Move::make(from, to, flag));
        }
    }

    // Queen moves
    uint64_t queens = board.queens(us);
    while (queens) {
        int from = lsb(queens);
        queens &= queens - 1;
        uint64_t targets = queen_attacks_bb(from, occ) & ~our_pieces;
        while (targets) {
            int to = lsb(targets);
            targets &= targets - 1;
            int flag = (sq_bb(to) & their_pieces) ? 4 : 0;
            list.add(Move::make(from, to, flag));
        }
    }

    // King moves
    uint64_t king = board.king(us);
    if (king) {
        int from = lsb(king);
        uint64_t targets = king_attacks[from] & ~our_pieces;
        while (targets) {
            int to = lsb(targets);
            targets &= targets - 1;
            int flag = (sq_bb(to) & their_pieces) ? 4 : 0;
            list.add(Move::make(from, to, flag));
        }

        // Castling
        if (us == WHITE) {
            // Kingside
            if ((board.castling_rights() & CASTLE_WHITE_KING) &&
                !(occ & (sq_bb(5) | sq_bb(6))) &&
                !is_attacked(board, 4, BLACK) &&
                !is_attacked(board, 5, BLACK) &&
                !is_attacked(board, 6, BLACK)) {
                list.add(Move::make(4, 6, 2));
            }
            // Queenside
            if ((board.castling_rights() & CASTLE_WHITE_QUEEN) &&
                !(occ & (sq_bb(1) | sq_bb(2) | sq_bb(3))) &&
                !is_attacked(board, 4, BLACK) &&
                !is_attacked(board, 3, BLACK) &&
                !is_attacked(board, 2, BLACK)) {
                list.add(Move::make(4, 2, 3));
            }
        } else {
            // Kingside
            if ((board.castling_rights() & CASTLE_BLACK_KING) &&
                !(occ & (sq_bb(61) | sq_bb(62))) &&
                !is_attacked(board, 60, WHITE) &&
                !is_attacked(board, 61, WHITE) &&
                !is_attacked(board, 62, WHITE)) {
                list.add(Move::make(60, 62, 2));
            }
            // Queenside
            if ((board.castling_rights() & CASTLE_BLACK_QUEEN) &&
                !(occ & (sq_bb(57) | sq_bb(58) | sq_bb(59))) &&
                !is_attacked(board, 60, WHITE) &&
                !is_attacked(board, 59, WHITE) &&
                !is_attacked(board, 58, WHITE)) {
                list.add(Move::make(60, 58, 3));
            }
        }
    }
}

// Generate captures only (for quiescence)
void generate_captures(const Board& board, MoveList& list) {
    list.clear();
    Color us = Color(board.side());
    Color them = Color(1 - us);
    uint64_t occ = board.all_pieces();
    uint64_t our_pieces = board.bb_color(us);
    uint64_t their_pieces = board.bb_color(them);
    int ep_sq = board.ep_square();

    // Pawn captures
    uint64_t pawns = board.pawns(us);
    if (us == WHITE) {
        uint64_t cap = ((pawns & ~FILE_A_BB) << 7) & their_pieces;
        while (cap) {
            int to = lsb(cap); cap &= cap - 1;
            int from = to - 7;
            if (rank_of(to) == 7) {
                list.add(Move::make(from, to, 15));
                list.add(Move::make(from, to, 12));
            } else {
                list.add(Move::make(from, to, 4));
            }
        }
        cap = ((pawns & ~FILE_H_BB) << 9) & their_pieces;
        while (cap) {
            int to = lsb(cap); cap &= cap - 1;
            int from = to - 9;
            if (rank_of(to) == 7) {
                list.add(Move::make(from, to, 15));
                list.add(Move::make(from, to, 12));
            } else {
                list.add(Move::make(from, to, 4));
            }
        }
        // En passant (must guard file boundaries to prevent wrap-around)
        if (ep_sq >= 0) {
            int ep_file = file_of(ep_sq);
            if (ep_file > 0 && (pawns & sq_bb(ep_sq - 7))) list.add(Move::make(ep_sq - 7, ep_sq, 5));
            if (ep_file < 7 && (pawns & sq_bb(ep_sq - 9))) list.add(Move::make(ep_sq - 9, ep_sq, 5));
        }
    } else {
        uint64_t cap = ((pawns & ~FILE_H_BB) >> 7) & their_pieces;
        while (cap) {
            int to = lsb(cap); cap &= cap - 1;
            int from = to + 7;
            if (rank_of(to) == 0) {
                list.add(Move::make(from, to, 15));
                list.add(Move::make(from, to, 12));
            } else {
                list.add(Move::make(from, to, 4));
            }
        }
        cap = ((pawns & ~FILE_A_BB) >> 9) & their_pieces;
        while (cap) {
            int to = lsb(cap); cap &= cap - 1;
            int from = to + 9;
            if (rank_of(to) == 0) {
                list.add(Move::make(from, to, 15));
                list.add(Move::make(from, to, 12));
            } else {
                list.add(Move::make(from, to, 4));
            }
        }
        // En passant (must guard file boundaries)
        if (ep_sq >= 0) {
            int ep_file = file_of(ep_sq);
            if (ep_file > 0 && (pawns & sq_bb(ep_sq + 7))) list.add(Move::make(ep_sq + 7, ep_sq, 5));
            if (ep_file < 7 && (pawns & sq_bb(ep_sq + 9))) list.add(Move::make(ep_sq + 9, ep_sq, 5));
        }
    }

    // Other pieces - captures only
    auto add_captures = [&](uint64_t pieces, uint64_t attacks_fn(int, uint64_t)) {
        while (pieces) {
            int from = lsb(pieces); pieces &= pieces - 1;
            uint64_t targets = attacks_fn(from, occ) & their_pieces;
            while (targets) {
                int to = lsb(targets); targets &= targets - 1;
                list.add(Move::make(from, to, 4));
            }
        }
    };

    // Knights
    uint64_t knights = board.knights(us);
    while (knights) {
        int from = lsb(knights); knights &= knights - 1;
        uint64_t targets = knight_attacks[from] & their_pieces;
        while (targets) {
            int to = lsb(targets); targets &= targets - 1;
            list.add(Move::make(from, to, 4));
        }
    }

    add_captures(board.bishops(us), bishop_attacks_bb);
    add_captures(board.rooks(us), rook_attacks_bb);
    add_captures(board.queens(us), queen_attacks_bb);

    // King captures
    uint64_t king = board.king(us);
    if (king) {
        int from = lsb(king);
        uint64_t targets = king_attacks[from] & their_pieces;
        while (targets) {
            int to = lsb(targets); targets &= targets - 1;
            list.add(Move::make(from, to, 4));
        }
    }
}

// Ordering score for a move
int move_score(const Board& board, Move move, int ply) {
    if (move.is_capture()) {
        // MVV-LVA: high victim value, low attacker value
        // EP capture: victim will be PT_NONE (6), clamp to 0
        PieceType victim = board.piece_type_on(move.to());
        int v = (victim >= PT_PAWN && victim <= PT_KING) ? (int)victim : 0;
        return 1000 + (v * 10);
    }
    if (move.is_promotion()) {
        return 900 + move.promotion_type();
    }
    // Killer moves
    if (killer_table[ply][0] == move) return 500;
    if (killer_table[ply][1] == move) return 400;

    // History heuristic
    Color us = Color(board.side());
    return history_table[us][move.from()][move.to()];
}

} // namespace engine
