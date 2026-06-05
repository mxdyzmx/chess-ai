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
    return c == WHITE ? ((pawns << 7) & ~0x0101010101010101ULL) | ((pawns << 9) & ~0x8080808080808080ULL)
                      : ((pawns >> 7) & ~0x8080808080808080ULL) | ((pawns >> 9) & ~0x0101010101010101ULL);
}

// Generate all pseudo-legal moves
void generate_moves(const Board& board, MoveList& list);

// Generate only captures (for quiescence search)
void generate_captures(const Board& board, MoveList& list);

// Verify move legality (make sure king not left in check)
bool is_legal_move(const Board& board, Move move);

// Check if a square is attacked by the given side
bool is_attacked(const Board& board, int sq, Color attacker);

// Static exchange evaluation
int see(const Board& board, Move move);

// Initialize movegen tables
void init_movegen();

// Get sliding attacks
inline uint64_t bishop_attacks_bb(int sq, uint64_t occ) {
    occ &= bishop_masks[sq];
    occ *= bishop_magics[sq];
    occ >>= bishop_shift[sq];
    return bishop_attacks[sq][occ];
}

inline uint64_t rook_attacks_bb(int sq, uint64_t occ) {
    occ &= rook_masks[sq];
    occ *= rook_magics[sq];
    occ >>= rook_shift[sq];
    return rook_attacks[sq][occ];
}

inline uint64_t queen_attacks_bb(int sq, uint64_t occ) {
    return bishop_attacks_bb(sq, occ) | rook_attacks_bb(sq, occ);
}

// Move ordering score
int move_score(const Board& board, Move move, int ply);

} // namespace engine

#endif // CHESS_MOVEGEN_H
