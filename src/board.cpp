#include "board.h"
#include "movegen.h"
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <random>

namespace engine {

// Static Zobrist tables
uint64_t Board::zobrist_piece_[6][2][64] = {{{0}}};
uint64_t Board::zobrist_castle_[16] = {0};
uint64_t Board::zobrist_ep_[64] = {0};
uint64_t Board::zobrist_side_ = 0;
bool Board::initialized_ = false;

// Piece characters
static const char* PIECE_CHARS = "PNBRQKpnbrqk";
static const PieceType CHAR_TO_PT[] = {
    PT_PAWN, PT_KNIGHT, PT_BISHOP, PT_ROOK, PT_QUEEN, PT_KING,
    PT_PAWN, PT_KNIGHT, PT_BISHOP, PT_ROOK, PT_QUEEN, PT_KING
};

void Board::init() {
    if (initialized_) return;
    initialized_ = true;

    // We also need to init movegen tables
    init_movegen();

    // Initialize Zobrist hash keys using a deterministic RNG
    std::mt19937_64 rng(0x3C5A9B8F1D4E7A20ULL);
    for (int pt = 0; pt < 6; pt++)
        for (int c = 0; c < 2; c++)
            for (int sq = 0; sq < 64; sq++)
                zobrist_piece_[pt][c][sq] = rng();

    for (int i = 0; i < 16; i++)
        zobrist_castle_[i] = rng();

    for (int sq = 0; sq < 64; sq++)
        zobrist_ep_[sq] = rng();

    zobrist_side_ = rng();
}

Board::Board() {
    set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
}

Board::Board(const std::string& fen) {
    set_fen(fen);
}

void Board::set_fen(const std::string& fen) {
    // Reset
    memset(bitboards_, 0, sizeof(bitboards_));
    by_color_[0] = by_color_[1] = 0;
    side_ = WHITE;
    ep_sq_ = -1;
    castle_ = 0;
    halfmove_ = 0;
    fullmove_ = 1;
    game_ply_ = 0;
    stack_size_ = 0;

    std::istringstream ss(fen);
    std::string board_str, side_str, castle_str, ep_str;
    ss >> board_str >> side_str >> castle_str >> ep_str;
    ss >> halfmove_;
    if (!ss.fail()) ss >> fullmove_;

    // Parse board
    int sq = 56; // A8
    for (char c : board_str) {
        if (c == '/') {
            sq -= 16; // Move to next rank (e.g., from A8 to A7)
        } else if (c >= '1' && c <= '8') {
            sq += (c - '0');
        } else {
            bool is_white = (c >= 'A' && c <= 'Z');
            Color col = is_white ? WHITE : BLACK;
            PieceType pt = PT_PAWN;
            switch (tolower(c)) {
                case 'p': pt = PT_PAWN; break;
                case 'n': pt = PT_KNIGHT; break;
                case 'b': pt = PT_BISHOP; break;
                case 'r': pt = PT_ROOK; break;
                case 'q': pt = PT_QUEEN; break;
                case 'k': pt = PT_KING; break;
            }
            put_piece(pt, col, sq);
            sq++;
        }
    }

    // Side to move
    side_ = (side_str == "w") ? WHITE : BLACK;

    // Castling rights
    if (castle_str.find('K') != std::string::npos) castle_ |= CASTLE_WHITE_KING;
    if (castle_str.find('Q') != std::string::npos) castle_ |= CASTLE_WHITE_QUEEN;
    if (castle_str.find('k') != std::string::npos) castle_ |= CASTLE_BLACK_KING;
    if (castle_str.find('q') != std::string::npos) castle_ |= CASTLE_BLACK_QUEEN;

    // En passant
    if (ep_str != "-") {
        int file = ep_str[0] - 'a';
        int rank = ep_str[1] - '1';
        ep_sq_ = square(rank, file);
    } else {
        ep_sq_ = -1;
    }

    recompute_by_color();
    recompute_hash();
}

std::string Board::fen() const {
    std::string fen;

    // Board
    for (int rank = 7; rank >= 0; rank--) {
        int empty = 0;
        for (int file = 0; file < 8; file++) {
            int sq = square(rank, file);
            PieceType pt = piece_type_on(sq);
            if (pt == PT_NONE) {
                empty++;
            } else {
                if (empty > 0) { fen += char('0' + empty); empty = 0; }
                Color c = color_on(sq);
                char p = PIECE_CHARS[pt + (c == WHITE ? 0 : 6)];
                fen += p;
            }
        }
        if (empty > 0) fen += char('0' + empty);
        if (rank > 0) fen += '/';
    }

    // Side
    fen += (side_ == WHITE) ? " w " : " b ";

    // Castling
    std::string castle_str;
    if (castle_ & CASTLE_WHITE_KING) castle_str += 'K';
    if (castle_ & CASTLE_WHITE_QUEEN) castle_str += 'Q';
    if (castle_ & CASTLE_BLACK_KING) castle_str += 'k';
    if (castle_ & CASTLE_BLACK_QUEEN) castle_str += 'q';
    if (castle_str.empty()) castle_str = "-";
    fen += castle_str;

    // En passant
    fen += ' ';
    if (ep_sq_ >= 0) {
        fen += char('a' + file_of(ep_sq_));
        fen += char('1' + rank_of(ep_sq_));
    } else {
        fen += '-';
    }

    // Halfmove, fullmove
    fen += ' ' + std::to_string(halfmove_);
    fen += ' ' + std::to_string(fullmove_);

    return fen;
}

void Board::put_piece(PieceType pt, Color c, int sq) {
    bitboards_[pt][c] |= sq_bb(sq);
    by_color_[c] |= sq_bb(sq);
}

void Board::remove_piece(PieceType pt, Color c, int sq) {
    bitboards_[pt][c] ^= sq_bb(sq);
    by_color_[c] ^= sq_bb(sq);
}

void Board::move_piece(PieceType pt, Color c, int from, int to) {
    uint64_t from_to = sq_bb(from) | sq_bb(to);
    bitboards_[pt][c] ^= from_to;
    by_color_[c] ^= from_to;
}

void Board::recompute_by_color() {
    by_color_[0] = 0;
    by_color_[1] = 0;
    for (int pt = 0; pt < 6; pt++) {
        by_color_[0] |= bitboards_[pt][0];
        by_color_[1] |= bitboards_[pt][1];
    }
}

void Board::recompute_hash() {
    hash_ = 0;
    for (int pt = 0; pt < 6; pt++) {
        for (int c = 0; c < 2; c++) {
            uint64_t bb = bitboards_[pt][c];
            while (bb) {
                int sq = lsb(bb);
                bb &= bb - 1;
                hash_ ^= zobrist_piece_[pt][c][sq];
            }
        }
    }
    hash_ ^= zobrist_castle_[castle_];
    if (ep_sq_ >= 0) hash_ ^= zobrist_ep_[ep_sq_];
    if (side_ == BLACK) hash_ ^= zobrist_side_;
}

PieceType Board::piece_type_on(int sq) const {
    uint64_t mask = sq_bb(sq);
    for (int pt = 0; pt < 6; pt++)
        if (bitboards_[pt][0] & mask) return PieceType(pt);
    for (int pt = 0; pt < 6; pt++)
        if (bitboards_[pt][1] & mask) return PieceType(pt);
    return PT_NONE;
}

Color Board::color_on(int sq) const {
    uint64_t mask = sq_bb(sq);
    if (by_color_[0] & mask) return WHITE;
    if (by_color_[1] & mask) return BLACK;
    return WHITE; // shouldn't happen
}

int Board::num_pieces(PieceType pt, Color c) const {
    return popcount(bitboards_[pt][c]);
}

bool Board::in_check() const {
    uint64_t king_bb = bitboards_[PT_KING][side_];
    if (king_bb == 0) return false;
    int king_sq = lsb(king_bb);
    Color enemy = Color(1 - side_);

    return is_attacked(*this, king_sq, enemy);
}

bool Board::is_draw() const {
    // Insufficient material
    if (popcount(all_pieces()) == 2) return true; // K vs K

    if (popcount(all_pieces()) == 3) {
        // K vs K+N or K vs K+B
        bool has_minor = false;
        for (Color c = WHITE; c <= BLACK; c = Color(c + 1)) {
            if (knights(c) || bishops(c)) {
                if (has_minor) return false; // both sides have minor
                has_minor = true;
            }
        }
        return true;
    }

    // 50-move rule
    if (halfmove_ >= 100) return true;

    // Repetition detection
    int count = 0;
    for (int i = 0; i < stack_size_; i++)
        if (stack_hash_[i] == hash_) count++;
    if (count >= 2) return true; // 3-fold (current position plus 2 previous)

    return false;
}

bool Board::is_mated() const {
    if (!in_check()) return false;
    MoveList list;
    generate_moves(*this, list);
    Board& self = const_cast<Board&>(*this);
    for (int i = 0; i < list.size; i++) {
        Move m = list.moves[i];
        self.make_move(m);
        bool legal = !self.in_check();
        self.unmake_move();
        if (legal) return false;
    }
    return true;
}

bool Board::make_move(Move m) {
    // Save state
    if (stack_size_ < MAX_GAME_PLY) {
        stack_hash_[stack_size_] = hash_;
        stack_ep_[stack_size_] = ep_sq_;
        stack_castle_[stack_size_] = castle_;
        stack_halfmove_[stack_size_] = halfmove_;
        stack_move_[stack_size_] = m.data;
    }
    stack_size_++;

    int from = m.from();
    int to = m.to();
    int flags = m.flags();
    PieceType pt = piece_type_on(from);
    if (pt < PT_PAWN || pt > PT_KING) {
        std::cerr << "ERROR: make_move: from=" << from << " to=" << to
                  << " flags=" << m.flags() << " side=" << side_
                  << " pieces=" << popcount(all_pieces())
                  << " fen=" << fen() << std::endl;
        stack_size_--; // undo the save
        return false;
    }
    // Verify piece color on from-square matches moving side
    if (color_on(from) != side_) {
        std::cerr << "ERROR: make_move: color mismatch: from=" << from << " side=" << side_
                  << " color_on_from=" << color_on(from)
                  << " fen=" << fen() << std::endl;
        stack_size_--; // undo the save
        return false;
    }
    Color us = side_;
    Color them = Color(1 - us);
    bool is_capture = m.is_capture();
    bool is_promotion = m.is_promotion();
    bool is_castle = m.is_castle();
    bool is_ep = m.is_enpassant();

    // Save captured piece type
    uint8_t captured_pt = PT_NONE;
    if (is_ep) {
        captured_pt = PT_PAWN;
    } else if (is_capture) {
        captured_pt = (uint8_t)piece_type_on(to);
    }
    if (stack_size_ > 0 && stack_size_ <= MAX_GAME_PLY) {
        stack_captured_[stack_size_ - 1] = captured_pt;
        stack_piece_pt_[stack_size_ - 1] = (uint8_t)pt;
    }

    // Remove hash of from/to squares
    hash_ ^= zobrist_piece_[pt][us][from];
    hash_ ^= zobrist_piece_[pt][us][to];

    // Move piece
    move_piece(pt, us, from, to);

    if (is_capture && !is_ep && captured_pt != PT_NONE) {
        hash_ ^= zobrist_piece_[captured_pt][them][to];
        remove_piece(PieceType(captured_pt), them, to);
    }

    if (is_ep) {
        int ep_capture_sq = to + (us == WHITE ? -8 : 8);
        hash_ ^= zobrist_piece_[PT_PAWN][them][ep_capture_sq];
        remove_piece(PT_PAWN, them, ep_capture_sq);
    }

    // Promotions
    if (is_promotion) {
        remove_piece(PT_PAWN, us, to);
        hash_ ^= zobrist_piece_[PT_PAWN][us][to];  // Undo pawn hash from line 317
        PieceType promo_pt = PieceType(m.promotion_type());
        put_piece(promo_pt, us, to);
        hash_ ^= zobrist_piece_[promo_pt][us][to];
    }

    // Castling
    if (is_castle) {
        int rook_from, rook_to;
        if (to > from) { // Kingside
            rook_from = to + 1;
            rook_to = to - 1;
        } else { // Queenside
            rook_from = to - 2;
            rook_to = to + 1;
        }
        hash_ ^= zobrist_piece_[PT_ROOK][us][rook_from];
        hash_ ^= zobrist_piece_[PT_ROOK][us][rook_to];
        move_piece(PT_ROOK, us, rook_from, rook_to);
    }

    // Update en passant square
    int old_ep = ep_sq_;
    if (old_ep >= 0) {
        hash_ ^= zobrist_ep_[old_ep];
    }
    ep_sq_ = -1;

    if (pt == PT_PAWN && abs(to - from) == 16) {
        ep_sq_ = (from + to) / 2;
        hash_ ^= zobrist_ep_[ep_sq_];
    }

    // Update castling rights
    uint8_t new_castle = castle_;
    if (castle_) {
        if (pt == PT_KING) {
            if (us == WHITE) new_castle &= ~(CASTLE_WHITE_KING | CASTLE_WHITE_QUEEN);
            else new_castle &= ~(CASTLE_BLACK_KING | CASTLE_BLACK_QUEEN);
        }
        if (from == 0 || to == 0) new_castle &= ~CASTLE_WHITE_QUEEN;
        if (from == 7 || to == 7) new_castle &= ~CASTLE_WHITE_KING;
        if (from == 56 || to == 56) new_castle &= ~CASTLE_BLACK_QUEEN;
        if (from == 63 || to == 63) new_castle &= ~CASTLE_BLACK_KING;

        if (new_castle != castle_) {
            hash_ ^= zobrist_castle_[castle_];
            hash_ ^= zobrist_castle_[new_castle];
            castle_ = new_castle;
        }
    }

    // Side to move
    side_ = them;
    hash_ ^= zobrist_side_;

    // Halfmove clock
    if (pt == PT_PAWN || is_capture)
        halfmove_ = 0;
    else
        halfmove_++;

    if (side_ == WHITE) fullmove_++;
    game_ply_++;

    return true;
}

void Board::unmake_move() {
    stack_size_--;
    if (stack_size_ < 0) return;

    // Restore state from stack
    hash_ = stack_hash_[stack_size_];
    ep_sq_ = stack_ep_[stack_size_];
    castle_ = stack_castle_[stack_size_];
    halfmove_ = stack_halfmove_[stack_size_];

    Move m = Move(stack_move_[stack_size_]);
    uint8_t captured_pt = stack_captured_[stack_size_];

    int from = m.from();
    int to = m.to();
    Color us = Color(1 - side_); // was the moving side
    Color them = side_;
    // Use saved piece type from stack instead of piece_type_on(to),
    // which can return wrong type if board is corrupted
    PieceType pt = PieceType(stack_piece_pt_[stack_size_]);
    bool is_promotion = m.is_promotion();

    // Handle promotion undo
    if (is_promotion) {
        // Remove the promoted piece from 'to' (NOT the pawn - the promoted
        // piece is what's actually on 'to' after make_move).
        PieceType promo_pt = PieceType(m.promotion_type());
        remove_piece(promo_pt, us, to);
        // Restore pawn to 'from'. Use put_piece (not move_piece) since the
        // pawn bitboard is empty on both squares after promotion.
        put_piece(PT_PAWN, us, from);
    } else {
        // Move piece back
        move_piece(pt, us, to, from);
    }

    // Handle captures
    if (m.is_enpassant()) {
        int ep_cap_sq = to + (us == WHITE ? -8 : 8);
        put_piece(PT_PAWN, them, ep_cap_sq);
    } else if (m.is_capture() && captured_pt != PT_NONE) {
        put_piece(PieceType(captured_pt), them, to);
    }

    // Handle castling
    if (m.is_castle()) {
        int rook_from, rook_to;
        if (to > from) { // Kingside
            rook_from = to - 1;
            rook_to = to + 1;
        } else { // Queenside
            rook_from = to + 1;
            rook_to = to - 2;
        }
        move_piece(PT_ROOK, us, rook_from, rook_to);
    }

    side_ = us;
    if (side_ == BLACK) fullmove_--;
    game_ply_--;
}

void Board::make_null_move() {
    // Save state to stack for consistency with normal move stack
    if (stack_size_ < MAX_GAME_PLY) {
        stack_hash_[stack_size_] = hash_;
        stack_ep_[stack_size_] = ep_sq_;
        stack_castle_[stack_size_] = castle_;
        stack_halfmove_[stack_size_] = halfmove_;
        stack_move_[stack_size_] = 0; // null move marker
        stack_captured_[stack_size_] = PT_NONE;
        stack_piece_pt_[stack_size_] = PT_NONE;
    }
    stack_size_++;

    // Clear EP square — it's only valid for one turn
    if (ep_sq_ >= 0) {
        hash_ ^= zobrist_ep_[ep_sq_];
        ep_sq_ = -1;
    }
    side_ = Color(1 - side_);
    hash_ ^= zobrist_side_;
}

void Board::unmake_null_move() {
    stack_size_--;
    if (stack_size_ < 0) return;

    // Restore state from stack
    hash_ = stack_hash_[stack_size_];
    ep_sq_ = stack_ep_[stack_size_];
    castle_ = stack_castle_[stack_size_];
    halfmove_ = stack_halfmove_[stack_size_];

    side_ = Color(1 - side_);
}

bool is_attacked(const Board& board, int sq, Color attacker) {
    uint64_t occ = board.all_pieces();

    // Pawn attacks
    uint64_t pawns = board.pawns(attacker);
    uint64_t pawn_att = pawn_attacks_from(Color(1 - attacker), sq_bb(sq));
    if (pawns & pawn_att) return true;

    // Knight attacks
    if (board.knights(attacker) & knight_attacks[sq]) return true;

    // King attacks
    if (board.king(attacker) & king_attacks[sq]) return true;

    // Bishop/queen attacks
    uint64_t bishops_queens = board.bishops(attacker) | board.queens(attacker);
    if (bishops_queens) {
        uint64_t b_att = bishop_attacks_bb(sq, occ);
        if (bishops_queens & b_att) return true;
    }

    // Rook/queen attacks
    uint64_t rooks_queens = board.rooks(attacker) | board.queens(attacker);
    if (rooks_queens) {
        uint64_t r_att = rook_attacks_bb(sq, occ);
        if (rooks_queens & r_att) return true;
    }

    return false;
}

bool Board::verify_integrity(std::string* out) const {
    // Check cross-color double occupancy
    uint64_t total_occ = by_color_[0] | by_color_[1];
    uint64_t cross = by_color_[0] & by_color_[1];
    if (cross && out) {
        *out += " cross_color_sq=" + std::to_string(lsb(cross));
    }

    // Count pieces per side from bitboards directly (not by_color_)
    int piece_count[2] = {0, 0};
    for (int pt = 0; pt < 6; pt++) {
        piece_count[0] += popcount(bitboards_[pt][0]);
        piece_count[1] += popcount(bitboards_[pt][1]);
    }
    if (piece_count[0] + piece_count[1] != popcount(total_occ) && out) {
        *out += " piece_vs_occ=" + std::to_string(piece_count[0] + piece_count[1])
                + "vs" + std::to_string(popcount(total_occ));
    }

    // Check that by_color_ matches sum of piece bitboards
    uint64_t expected_by_color[2] = {0, 0};
    for (int pt = 0; pt < 6; pt++) {
        expected_by_color[0] |= bitboards_[pt][0];
        expected_by_color[1] |= bitboards_[pt][1];
    }
    bool ok = true;
    for (int c = 0; c < 2; c++) {
        uint64_t diff = expected_by_color[c] ^ by_color_[c];
        if (diff) {
            if (out) {
                *out += " byc_" + std::to_string(c) + "_diff_sq=" + std::to_string(lsb(diff));
            }
            ok = false;
        }
    }

    // Check hash
    uint64_t expected_hash = 0;
    for (int pt = 0; pt < 6; pt++) {
        for (int c = 0; c < 2; c++) {
            uint64_t bb = bitboards_[pt][c];
            while (bb) {
                int sq = lsb(bb);
                bb &= bb - 1;
                expected_hash ^= zobrist_piece_[pt][c][sq];
            }
        }
    }
    expected_hash ^= zobrist_castle_[castle_];
    if (ep_sq_ >= 0) expected_hash ^= zobrist_ep_[ep_sq_];
    if (side_ == BLACK) expected_hash ^= zobrist_side_;

    if (expected_hash != hash_) {
        if (out) *out += " hash_mismatch";
        ok = false;
    }
    if (out) {
        *out += " w=" + std::to_string(piece_count[0]) + " b=" + std::to_string(piece_count[1]);
    }
    return ok;
}

} // namespace engine

