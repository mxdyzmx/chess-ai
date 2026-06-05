#include "uci.h"
#include "eval.h"
#include <iostream>
#include <sstream>
#include <thread>
#include <cctype>

namespace engine {

UCI::UCI() : search_(), board_() {
    Board::init();
    Search::tt().resize(256); // Default 256MB TT
}

void UCI::loop() {
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;

        std::istringstream ss(line);
        std::string cmd;
        ss >> cmd;

        if (cmd == "uci") {
            cmd_uci();
        } else if (cmd == "isready") {
            cmd_isready();
        } else if (cmd == "setoption") {
            std::string name, value;
            ss >> name; // "name"
            if (name == "name") {
                ss >> name >> value; // "Hash" "value" or similar
                // Actually, setoption name Hash value 128
                // Read "name" keyword, then option name, then "value", then value
                std::string option_name;
                ss >> option_name; // e.g. "Hash"
                std::string val_token;
                ss >> val_token; // "value"
                if (val_token == "value") {
                    ss >> value;
                }
                cmd_setoption(option_name, value);
            }
        } else if (cmd == "ucinewgame") {
            cmd_ucinewgame();
        } else if (cmd == "position") {
            std::string rest;
            std::getline(ss, rest);
            cmd_position(rest);
        } else if (cmd == "go") {
            std::string rest;
            std::getline(ss, rest);
            cmd_go(rest);
        } else if (cmd == "stop") {
            cmd_stop();
        } else if (cmd == "quit") {
            cmd_quit();
            break;
        } else if (cmd == "d") {
            // Debug: print board
            std::cout << board_.fen() << std::endl;
            // Print board visually
            for (int r = 7; r >= 0; r--) {
                std::cout << (r + 1) << " ";
                for (int f = 0; f < 8; f++) {
                    int sq = r * 8 + f;
                    char c = '.';
                    for (int pt = 0; pt < 6; pt++) {
                        if (board_.bb_piece(PieceType(pt), WHITE) & (1ULL << sq)) {
                            c = "PNBRQK"[pt];
                            break;
                        }
                        if (board_.bb_piece(PieceType(pt), BLACK) & (1ULL << sq)) {
                            c = "pnbrqk"[pt];
                            break;
                        }
                    }
                    std::cout << c << " ";
                }
                std::cout << std::endl;
            }
            std::cout << "  a b c d e f g h" << std::endl;
            std::cout << "Side: " << (board_.is_white_to_move() ? "white" : "black") << std::endl;
            std::cout << "Hash: " << board_.hash() << std::endl;
            std::cout << "Eval: " << evaluate(board_) << std::endl;
        } else if (cmd == "eval") {
            std::cout << "Evaluation: " << evaluate(board_) << " cp" << std::endl;
        } else if (cmd == "print_weights") {
            for (int i = 0; i < NUM_FEATURES; i++) {
                std::cout << eval_weights[i];
                if (i < NUM_FEATURES - 1) std::cout << ", ";
                if ((i + 1) % 16 == 0) std::cout << std::endl;
            }
            std::cout << std::endl;
        }
    }
}

void UCI::cmd_uci() {
    std::cout << "id name ChessAI 1.0" << std::endl;
    std::cout << "id author ChessAI" << std::endl;
    std::cout << "option name Hash type spin default 256 min 1 max 4096" << std::endl;
    std::cout << "uciok" << std::endl;
}

void UCI::cmd_isready() {
    std::cout << "readyok" << std::endl;
}

void UCI::cmd_setoption(const std::string& name, const std::string& value) {
    if (name == "Hash") {
        int mb = std::stoi(value);
        Search::tt().resize(mb);
    }
}

void UCI::cmd_ucinewgame() {
    board_.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    Search::tt().clear();
    memset(history_table, 0, sizeof(history_table));
    memset(killer_table, 0, sizeof(killer_table));
}

void UCI::cmd_position(const std::string& input) {
    std::istringstream ss(input);
    std::string token;
    ss >> token;

    if (token == "startpos") {
        board_.set_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        ss >> token; // "moves"
        if (token == "moves") {
            std::string move_str;
            while (ss >> move_str) {
                Move m = parse_move(move_str);
                if (m.data != 0) board_.make_move(m);
            }
        }
    } else if (token == "fen") {
        std::string fen;
        for (int i = 0; i < 6; i++) {
            ss >> token;
            if (i > 0) fen += " ";
            fen += token;
            if (i == 3 && token == "-") { /* ep */ }
        }
        board_.set_fen(fen);

        ss >> token; // "moves" (if any)
        if (token == "moves") {
            std::string move_str;
            while (ss >> move_str) {
                Move m = parse_move(move_str);
                if (m.data != 0) board_.make_move(m);
            }
        }
    }
}

void UCI::cmd_go(const std::string& input) {
    SearchParams params;
    std::istringstream ss(input);
    std::string token;

    while (ss >> token) {
        if (token == "depth") {
            ss >> params.depth;
        } else if (token == "movetime") {
            ss >> params.movetime;
        } else if (token == "wtime") {
            ss >> params.wtime;
        } else if (token == "btime") {
            ss >> params.btime;
        } else if (token == "winc") {
            ss >> params.winc;
        } else if (token == "binc") {
            ss >> params.binc;
        } else if (token == "movestogo") {
            ss >> params.movestogo;
        } else if (token == "infinite") {
            params.infinite = true;
        }
    }

    search_.set_position(board_);
    search_.set_params(params);

    // Run search in a thread so we can respond to "stop"
    std::thread search_thread([this]() {
        search_.search();
    });
    search_thread.detach();

    // Wait for search to finish (or be stopped)
    // For simplicity with the UCI protocol, since output goes to cout
    // we need to handle the stop command potentially arriving during search
    // We'll use a simple approach: wait a bit and check
    // Actually, for proper UCI, we should not block here
    // But since we're using cout for both search output and command handling,
    // let's use a simple polling approach
}

void UCI::cmd_stop() {
    search_.stop();
}

void UCI::cmd_quit() {
    search_.stop();
    // Give search thread time to stop
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
}

Move UCI::parse_move(const std::string& str) {
    if (str.length() < 4) return Move(0);

    int from_file = str[0] - 'a';
    int from_rank = str[1] - '1';
    int to_file = str[2] - 'a';
    int to_rank = str[3] - '1';

    if (from_file < 0 || from_file > 7 || from_rank < 0 || from_rank > 7 ||
        to_file < 0 || to_file > 7 || to_rank < 0 || to_rank > 7) {
        return Move(0);
    }

    int from = from_rank * 8 + from_file;
    int to = to_rank * 8 + to_file;

    // Determine move type
    PieceType pt = board_.piece_type_on(from);
    bool is_capture = board_.piece_type_on(to) != PT_NONE;
    bool is_ep = (pt == PT_PAWN && to == board_.ep_square());

    // Check for promotion
    if (str.length() >= 5) {
        int promo = 0;
        switch (str[4]) {
            case 'n': promo = PT_KNIGHT; break;
            case 'b': promo = PT_BISHOP; break;
            case 'r': promo = PT_ROOK; break;
            case 'q': promo = PT_QUEEN; break;
        }
        if (promo > 0) {
            int flags = (promo - PT_KNIGHT) + 8; // 8-11 for quiet promo
            if (is_capture) flags += 4; // 12-15 for capture promo
            return Move::make(from, to, flags);
        }
    }

    // Check for castling
    if (pt == PT_KING && abs(to - from) == 2) {
        int flags = (to > from) ? 2 : 3; // kingside = 2, queenside = 3
        return Move::make(from, to, flags);
    }

    // Normal move or capture or en-passant
    int flags = 0;
    if (is_ep) flags = 5;
    else if (is_capture) flags = 4;

    // Check for double pawn push
    if (pt == PT_PAWN && abs(to - from) == 16) {
        flags = 1;
    }

    return Move::make(from, to, flags);
}

std::string UCI::move_to_string(Move m) {
    std::string str;
    str += char('a' + (m.from() & 7));
    str += char('1' + (m.from() >> 3));
    str += char('a' + (m.to() & 7));
    str += char('1' + (m.to() >> 3));
    if (m.is_promotion()) {
        const char* p = "nbrq";
        str += p[m.promotion_type() - PT_KNIGHT];
    }
    return str;
}

} // namespace engine
