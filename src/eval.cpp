#include "eval.h"
#include "movegen.h"

namespace engine {

// Eval weights - initialized to default (will be replaced by Python-generated values)
int16_t eval_weights[NUM_FEATURES] = {
    #include "default_weights.inc"
};

// Piece values for game phase computation
constexpr int PIECE_VALUES[6] = { 100, 300, 320, 500, 900, 20000 };

// Phase values (Tapered eval)
constexpr int PHASE_VALUES[6] = { 0, 1, 1, 2, 4, 0 };
constexpr int MAX_PHASE = 24; // 8*0 + 2*1 + 2*1 + 2*2 + 1*4 = 0+2+2+4+4 = 12 for each side = 24

// King safety pawn shield masks by king position
// Pawn shield: squares in front of king on same file and adjacent files
static uint64_t king_shield_mask(Color c, int king_sq) {
    int r = rank_of(king_sq);
    int f = file_of(king_sq);
    uint64_t mask = 0;
    if (c == WHITE) {
        // Kingside pawn shield (files f-1, f, f+1, ranks r+1, r+2)
        for (int df = -1; df <= 1; df++) {
            int nf = f + df;
            if (nf < 0 || nf > 7) continue;
            for (int dr = 1; dr <= 2; dr++) {
                int nr = r + dr;
                if (nr > 7) continue;
                mask |= sq_bb(square(nr, nf));
            }
        }
    } else {
        for (int df = -1; df <= 1; df++) {
            int nf = f + df;
            if (nf < 0 || nf > 7) continue;
            for (int dr = -1; dr >= -2; dr--) {
                int nr = r + dr;
                if (nr < 0) continue;
                mask |= sq_bb(square(nr, nf));
            }
        }
    }
    return mask;
}

// Check if a pawn at square sq is isolated (no friendly pawns on adjacent files)
static int count_isolated_pawns(uint64_t pawns_bb) {
    const uint64_t FILE_A = 0x0101010101010101ULL;
    const uint64_t FILE_H = 0x8080808080808080ULL;

    int count = 0;
    uint64_t temp = pawns_bb;
    while (temp) {
        int sq = lsb(temp);
        temp &= temp - 1;

        // Check if any friendly pawns on adjacent files
        if (file_of(sq) > 0) {
            uint64_t left_mask = FILE_A << (file_of(sq) - 1);
            if (pawns_bb & left_mask) continue; // Not isolated
        }
        if (file_of(sq) < 7) {
            uint64_t right_mask = FILE_A << (file_of(sq) + 1);
            if (pawns_bb & right_mask) continue; // Not isolated
        }
        count++; // Isolated
    }
    return count;
}

// Check if a pawn is passed (no enemy pawns on same or adjacent files ahead)
static int count_passed_pawns(uint64_t pawns_bb, uint64_t enemy_pawns_bb, Color c) {
    int count = 0;
    uint64_t temp = pawns_bb;
    while (temp) {
        int sq = lsb(temp);
        temp &= temp - 1;
        bool passed = true;
        int f = file_of(sq);

        // Check left, same, and right files for enemy pawns ahead
        for (int df = -1; df <= 1; df++) {
            int nf = f + df;
            if (nf < 0 || nf > 7) continue;
            uint64_t file_mask = 0x0101010101010101ULL << nf;

            if (c == WHITE) {
                // Squares ahead of this pawn (higher ranks)
                uint64_t ahead_mask = file_mask & ~((1ULL << (sq + 1)) - 1);
                if (enemy_pawns_bb & ahead_mask) {
                    passed = false;
                    break;
                }
            } else {
                // Squares ahead of this pawn (lower ranks)
                uint64_t ahead_mask = file_mask & ((1ULL << sq) - 1);
                if (enemy_pawns_bb & ahead_mask) {
                    passed = false;
                    break;
                }
            }
        }
        if (passed) count++;
    }
    return count;
}

// King safety: evaluate pawn shield damage
static int king_safety_damage(uint64_t king_bb, uint64_t our_pawns, Color c) {
    if (king_bb == 0) return 4; // Worst case
    int ksq = lsb(king_bb);
    uint64_t shield = king_shield_mask(c, ksq);
    uint64_t shield_pawns = shield & our_pawns;

    int damage = 0;
    // Count missing shield pawns (out of 6 possible shield squares)
    int missing = 6 - popcount(shield_pawns);
    damage = std::min(missing, 4);
    return damage;
}

int game_phase(const Board& board) {
    int phase = 0;
    for (int pt = PT_PAWN; pt <= PT_KING; pt++) {
        phase += PHASE_VALUES[pt] * (board.num_pieces(PieceType(pt), WHITE) +
                                     board.num_pieces(PieceType(pt), BLACK));
    }
    return std::min(phase, MAX_PHASE);
}

void extract_features(const Board& board, float* features) {
    // Zero out features
    for (int i = 0; i < NUM_FEATURES; i++) features[i] = 0.0f;

    // PSQT features (0-383): +1 for white, -1 for black (mirrored vertically)
    for (int pt = 0; pt < 6; pt++) {
        for (int c = 0; c < 2; c++) {
            float val = (c == WHITE) ? 1.0f : -1.0f;
            uint64_t bb = board.bb_piece(PieceType(pt), Color(c));
            while (bb) {
                int sq = lsb(bb);
                bb &= bb - 1;
                features[pt * 64 + (c == WHITE ? sq : sq ^ 56)] = val;
            }
        }
    }

    // Mobility (384) — must match evaluate() which uses int min(list.size/50, 1)
    MoveList list;
    generate_moves(board, list);
    float mobility = (float)std::min(list.size / 50, 1);
    features[384] = mobility;

    // Pawn structure features (385-388)
    uint64_t w_pawns = board.pawns(WHITE);
    uint64_t b_pawns = board.pawns(BLACK);

    features[385] = (float)count_isolated_pawns(w_pawns); // White isolated
    features[386] = (float)count_isolated_pawns(b_pawns); // Black isolated
    features[387] = (float)count_passed_pawns(w_pawns, b_pawns, WHITE); // White passed
    features[388] = (float)count_passed_pawns(b_pawns, w_pawns, BLACK); // Black passed

    // King safety (389-390)
    features[389] = (float)king_safety_damage(board.king(WHITE), w_pawns, WHITE);
    features[390] = (float)king_safety_damage(board.king(BLACK), b_pawns, BLACK);
}

int evaluate(const Board& board) {
    int score = 0;

    // PSQT features (0-383)
    for (int pt = 0; pt < 6; pt++) {
        // White pieces: +1
        uint64_t bb = board.bb_piece(PieceType(pt), WHITE);
        while (bb) {
            int sq = lsb(bb);
            bb &= bb - 1;
            score += eval_weights[pt * 64 + sq];
        }
        // Black pieces: -1, mirror square vertically (sq ^ 56)
        bb = board.bb_piece(PieceType(pt), BLACK);
        while (bb) {
            int sq = lsb(bb) ^ 56;
            bb &= bb - 1;
            score -= eval_weights[pt * 64 + sq];
        }
    }

    // Mobility (384)
    MoveList list;
    generate_moves(board, list);
    int mobility = std::min(list.size / 50, 1);
    score += (int)eval_weights[384] * mobility;

    // Pawn structure (385-388)
    uint64_t w_pawns = board.pawns(WHITE);
    uint64_t b_pawns = board.pawns(BLACK);

    int w_isolated = count_isolated_pawns(w_pawns);
    int b_isolated = count_isolated_pawns(b_pawns);
    int w_passed = count_passed_pawns(w_pawns, b_pawns, WHITE);
    int b_passed = count_passed_pawns(b_pawns, w_pawns, BLACK);

    score += (int)eval_weights[385] * w_isolated;
    score -= (int)eval_weights[386] * b_isolated;
    score += (int)eval_weights[387] * w_passed;
    score -= (int)eval_weights[388] * b_passed;

    // King safety (389-390)
    int w_king_dmg = king_safety_damage(board.king(WHITE), w_pawns, WHITE);
    int b_king_dmg = king_safety_damage(board.king(BLACK), b_pawns, BLACK);

    score -= (int)eval_weights[389] * w_king_dmg;  // Damage is bad
    score += (int)eval_weights[390] * b_king_dmg;

    return board.is_white_to_move() ? score : -score;
}

} // namespace engine
