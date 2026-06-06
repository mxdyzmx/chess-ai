#ifndef CHESS_MOVEGEN_H
#define CHESS_MOVEGEN_H

#include "board.h"
#include <cstdint>

namespace engine {

// Move generation result
struct MoveList {
    Move moves[MAX_MOVES];
    int size = 0;
    void add(Move m) { moves[size++] = m; }
    void clear() { size = 0; }
};

// Attack tables
extern uint64_t pawn_attacks[2][64];
extern uint64_t knight_attacks[64];
extern uint64_t king_attacks[64];
extern uint64_t bishop_masks[64];
extern uint64_t rook_masks[64];
extern uint64_t bishop_attacks[64][512];
extern uint64_t rook_attacks[64][4096];
extern uint64_t bishop_magics[64];
extern uint64_t rook_magics[64];
extern int bishop_shift[64];
extern int rook_shift[64];

// Square/rank/file utilities
constexpr int square(int rank, int file) { return rank * 8 + file; }
constexpr int rank_of(int sq) { return sq >> 3; }
constexpr int file_of(int sq) { return sq & 7; }
constexpr uint64_t sq_bb(int sq) { return 1ULL << sq; }

// Bit operations
inline int popcount(uint64_t x) { return __builtin_popcountll(x); }
inline int lsb(uint64_t x) { return __builtin_ctzll(x); }
inline uint64_t pop_lsb(uint64_t& x) {
    uint64_t lsb = x & -x;
    x ^= lsb;
    return lsb;
}
inline uint64_t flip_vertical(uint64_t x) {
    return __builtin_bswap64(x);
}

// Ray directions
constexpr int NORTH = 8, SOUTH = -8, EAST = 1, WEST = -1;
constexpr int NE = 9, NW = 7, SE = -7, SW = -9;

// Move ordering scores
constexpr int MAX_HISTORY = 8192;
extern int history_table[2][64][64];     // [side][from][to]
extern Move killer_table[MAX_GAME_PLY][2];

// Between squares
extern uint64_t between_bb[64][64];

// Pawn attacks
inline uint64_t pawn_attacks_from(Color c, uint64_t pawns) {
    return c == WHITE ? ((pawns << 7) & ~0x8080808080808080ULL) | ((pawns << 9) & ~0x0101010101010101ULL)
                      : ((pawns >> 7) & ~0x0101010101010101ULL) | ((pawns >> 9) & ~0x8080808080808080ULL);
}

// Generate all pseudo-legal moves
void generate_moves(const Board& board, MoveList& list);

// Generate only captures (for quiescence search)
void generate_captures(const Board& board, MoveList& list);

// Check if a square is attacked by the given side
bool is_attacked(const Board& board, int sq, Color attacker);

// Initialize movegen tables
void init_movegen();

// Get sliding attacks (direct computation - no magic bitboards)
inline uint64_t bishop_attacks_bb(int sq, uint64_t occ) {
    uint64_t attacks = 0;
    int r = sq >> 3, f = sq & 7;
    for (int dr = -1; dr <= 1; dr += 2)
        for (int df = -1; df <= 1; df += 2)
            for (int i = 1; i < 8; i++) {
                int nr = r + dr * i, nf = f + df * i;
                if (nr < 0 || nr > 7 || nf < 0 || nf > 7) break;
                int nsq = nr * 8 + nf;
                attacks |= 1ULL << nsq;
                if (occ & (1ULL << nsq)) break;
            }
    return attacks;
}

inline uint64_t rook_attacks_bb(int sq, uint64_t occ) {
    uint64_t attacks = 0;
    int r = sq >> 3, f = sq & 7;
    for (int nr = r + 1; nr <= 7; nr++) {
        int nsq = nr * 8 + f;
        attacks |= 1ULL << nsq;
        if (occ & (1ULL << nsq)) break;
    }
    for (int nr = r - 1; nr >= 0; nr--) {
        int nsq = nr * 8 + f;
        attacks |= 1ULL << nsq;
        if (occ & (1ULL << nsq)) break;
    }
    for (int nf = f + 1; nf <= 7; nf++) {
        int nsq = r * 8 + nf;
        attacks |= 1ULL << nsq;
        if (occ & (1ULL << nsq)) break;
    }
    for (int nf = f - 1; nf >= 0; nf--) {
        int nsq = r * 8 + nf;
        attacks |= 1ULL << nsq;
        if (occ & (1ULL << nsq)) break;
    }
    return attacks;
}

inline uint64_t queen_attacks_bb(int sq, uint64_t occ) {
    return bishop_attacks_bb(sq, occ) | rook_attacks_bb(sq, occ);
}

// Move ordering score
int move_score(const Board& board, Move move, int ply);

} // namespace engine

#endif // CHESS_MOVEGEN_H
