#ifndef CHESS_BOARD_H
#define CHESS_BOARD_H

#include <cstdint>
#include <string>
#include <cstring>
#include <vector>
#include <array>
#include <cassert>

namespace engine {

enum PieceType : int { PT_PAWN = 0, PT_KNIGHT = 1, PT_BISHOP = 2, PT_ROOK = 3, PT_QUEEN = 4, PT_KING = 5, PT_NONE = 6 };
enum Color : int { WHITE = 0, BLACK = 1 };
enum CastlingRights : int {
    CASTLE_WHITE_KING  = 1,
    CASTLE_WHITE_QUEEN = 2,
    CASTLE_BLACK_KING  = 4,
    CASTLE_BLACK_QUEEN = 8
};

constexpr int MAX_MOVES = 256;
constexpr int MAX_GAME_PLY = 2048;

constexpr uint64_t FILE_A_BB = 0x0101010101010101ULL;
constexpr uint64_t FILE_B_BB = 0x0202020202020202ULL;
constexpr uint64_t FILE_C_BB = 0x0404040404040404ULL;
constexpr uint64_t FILE_D_BB = 0x0808080808080808ULL;
constexpr uint64_t FILE_E_BB = 0x1010101010101010ULL;
constexpr uint64_t FILE_F_BB = 0x2020202020202020ULL;
constexpr uint64_t FILE_G_BB = 0x4040404040404040ULL;
constexpr uint64_t FILE_H_BB = 0x8080808080808080ULL;
constexpr uint64_t RANK_1_BB = 0x00000000000000FFULL;
constexpr uint64_t RANK_2_BB = 0x000000000000FF00ULL;
constexpr uint64_t RANK_3_BB = 0x0000000000FF0000ULL;
constexpr uint64_t RANK_4_BB = 0x00000000FF000000ULL;
constexpr uint64_t RANK_5_BB = 0x000000FF00000000ULL;
constexpr uint64_t RANK_6_BB = 0x0000FF0000000000ULL;
constexpr uint64_t RANK_7_BB = 0x00FF000000000000ULL;
constexpr uint64_t RANK_8_BB = 0xFF00000000000000ULL;

// Move encoding: 16-bit
// bits 0-5: from square (0-63)
// bits 6-11: to square (0-63)
// bits 12-15: flags
// flags: 0=quiet, 1=double push, 2=kingside castle, 3=queenside castle,
//        4=normal capture, 5=en-passant, 8=knight promo, 9=bishop promo,
//        10=rook promo, 11=queen promo, 12=knight promo capture,
//        13=bishop promo capture, 14=rook promo capture, 15=queen promo capture
struct Move {
    uint16_t data;
    Move() : data(0) {}
    Move(uint16_t d) : data(d) {}
    int from() const { return data & 0x3f; }
    int to() const { return (data >> 6) & 0x3f; }
    int flags() const { return (data >> 12) & 0xf; }
    bool is_capture() const { return (flags() & 4) != 0; }
    bool is_promotion() const { return (flags() & 8) != 0; }
    bool is_castle() const { return flags() == 2 || flags() == 3; }
    bool is_enpassant() const { return flags() == 5; }
    int promotion_type() const {
        if (!is_promotion()) return PT_NONE;
        return (flags() & 3) + PT_KNIGHT; // knight=1, bishop=2, rook=3, queen=4
    }
    static Move make(int from, int to, int flags) {
        return Move(uint16_t(from | (to << 6) | (flags << 12)));
    }
    bool operator==(const Move& o) const { return data == o.data; }
    bool operator!=(const Move& o) const { return data != o.data; }
};

class Board {
public:
    Board();
    explicit Board(const std::string& fen);

    void set_fen(const std::string& fen);
    std::string fen() const;

    bool make_move(Move move);
    void unmake_move();

    // Bitboard access
    uint64_t bb_piece(PieceType pt, Color c) const { return bitboards_[pt][c]; }
    uint64_t bb_color(Color c) const { return by_color_[c]; }
    uint64_t all_pieces() const { return by_color_[0] | by_color_[1]; }

    PieceType piece_type_on(int sq) const;
    Color color_on(int sq) const;

    Color side() const { return side_; }
    int  ep_square() const { return ep_sq_; }
    int  castling_rights() const { return castle_; }
    int  halfmove_clock() const { return halfmove_; }
    int  fullmove_number() const { return fullmove_; }
    uint64_t hash() const { return hash_; }
    int  game_ply() const { return game_ply_; }
    bool is_white_to_move() const { return side_ == WHITE; }

    // Null move support
    void toggle_side() { side_ = Color(1 - side_); hash_ ^= zobrist_side_; }

    bool in_check() const;
    bool is_draw() const;
    bool is_mated() const;
    bool is_legal(Move m) const;

    // Get piece list for eval
    int num_pieces(PieceType pt, Color c) const;
    uint64_t pawns(Color c) const { return bitboards_[PT_PAWN][c]; }
    uint64_t knights(Color c) const { return bitboards_[PT_KNIGHT][c]; }
    uint64_t bishops(Color c) const { return bitboards_[PT_BISHOP][c]; }
    uint64_t rooks(Color c) const { return bitboards_[PT_ROOK][c]; }
    uint64_t queens(Color c) const { return bitboards_[PT_QUEEN][c]; }
    uint64_t king(Color c) const { return bitboards_[PT_KING][c]; }

    static void init();

private:
    uint64_t bitboards_[6][2];  // [piece][color]
    uint64_t by_color_[2];
    Color side_;
    int ep_sq_;
    uint8_t castle_;
    int halfmove_;
    int fullmove_;
    int game_ply_;
    uint64_t hash_;
    uint64_t stack_hash_[MAX_GAME_PLY];
    int stack_ep_[MAX_GAME_PLY];
    uint8_t stack_castle_[MAX_GAME_PLY];
    int stack_halfmove_[MAX_GAME_PLY];
    uint16_t stack_move_[MAX_GAME_PLY];
    uint8_t stack_captured_[MAX_GAME_PLY];
    uint8_t stack_piece_pt_[MAX_GAME_PLY];
    int stack_size_;

    void put_piece(PieceType pt, Color c, int sq);
    void remove_piece(PieceType pt, Color c, int sq);
    void move_piece(PieceType pt, Color c, int from, int to);
    void recompute_hash();
    void recompute_by_color();

    static uint64_t zobrist_piece_[6][2][64];
    static uint64_t zobrist_castle_[16];
    static uint64_t zobrist_ep_[64];
    static uint64_t zobrist_side_;
    static bool initialized_;
};

} // namespace engine

#endif // CHESS_BOARD_H
