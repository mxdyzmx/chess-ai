#ifndef CHESS_EVAL_H
#define CHESS_EVAL_H

#include "board.h"

namespace engine {

// Feature dimensions
constexpr int NUM_PSQT_FEATURES = 384;  // 6 piece types x 64 squares
constexpr int NUM_MOBILITY_FEATURES = 1;
constexpr int NUM_PAWN_FEATURES = 4;    // isolated W/B, passed W/B
constexpr int NUM_KING_FEATURES = 2;    // king safety W/B
constexpr int NUM_FEATURES = NUM_PSQT_FEATURES + NUM_MOBILITY_FEATURES
                           + NUM_PAWN_FEATURES + NUM_KING_FEATURES; // = 391

// PSQT feature layout:
// 0-63:   pawn
// 64-127: knight
// 128-191: bishop
// 192-255: rook
// 256-319: queen
// 320-383: king
// Square ordering: A1=0, B1=1, ..., H1=7, A2=8, ..., H8=63

// Non-PSQT feature layout:
// 384: mobility
// 385: white isolated pawn count
// 386: black isolated pawn count
// 387: white passed pawn count
// 388: black passed pawn count
// 389: white king safety (kingside pawn shield weakness)
// 390: black king safety (kingside pawn shield weakness)

// Eval weights (int16, scaled by 100 from float model)
extern int16_t eval_weights[NUM_FEATURES];

// Default weights (trained approximation, will be overwritten by Python-generated values)
constexpr int16_t DEFAULT_WEIGHTS[NUM_FEATURES] = {0};

// Evaluation result
struct EvalResult {
    int score;              // centipawns, from white's perspective
    float features[NUM_FEATURES]; // for debugging/inspection
};

// Main evaluation function
int evaluate(const Board& board);

// Feature extraction (for Python export compatibility)
void extract_features(const Board& board, float* features);

// Phase: 0 = opening, 256 = endgame
int game_phase(const Board& board);

} // namespace engine

#endif // CHESS_EVAL_H
