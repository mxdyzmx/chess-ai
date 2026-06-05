#include "board.h"
#include "movegen.h"
#include "eval.h"
#include "search.h"
#include "uci.h"
#include <iostream>
#include <signal.h>

namespace engine {

// Signal handler for Ctrl+C
static volatile sig_atomic_t quit_flag = 0;
void handle_signal(int) {
    quit_flag = 1;
}

} // namespace engine

int main(int argc, char* argv[]) {
    using namespace engine;

    // Register signal handler
    signal(SIGINT, handle_signal);
    signal(SIGTERM, handle_signal);

    // Initialize the engine
    Board::init();

    // Parse command line arguments
    if (argc >= 3 && std::string(argv[1]) == "--tt-size") {
        int mb = std::atoi(argv[2]);
        if (mb > 0) Search::tt().resize(mb);
    }

    if (argc >= 2 && std::string(argv[1]) == "--bench") {
        // Benchmark mode: run search on starting position at depth 6
        Search::tt().resize(1); // 1MB TT for testing
        Board board;
        Search search;
        SearchParams params;
        params.depth = 6;
        params.movetime = 0;

        search.set_position(board);
        search.set_params(params);

        std::cout << "Benchmark: searching starting position at depth " << params.depth << std::endl;
        std::cout << "Searching..." << std::endl;

        auto start = std::chrono::steady_clock::now();
        search.search();
        std::cout << "Search complete." << std::endl;
        auto end = std::chrono::steady_clock::now();

        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        const auto& stats = search.stats();

        std::cout << "Benchmark results:" << std::endl;
        std::cout << "  Depth: " << stats.depth_reached << std::endl;
        std::cout << "  Nodes: " << stats.nodes << std::endl;
        std::cout << "  Time: " << ms << " ms" << std::endl;
        std::cout << "  NPS: " << (ms > 0 ? stats.nodes * 1000 / ms : 0) << std::endl;
        std::cout << "  TT hits: " << stats.tt_hits << std::endl;
        std::cout << "  Best move: ";
        Move bm = search.best_move();
        if (bm.data != 0) {
            std::cout << char('a' + (bm.from() & 7)) << char('1' + (bm.from() >> 3));
            std::cout << char('a' + (bm.to() & 7)) << char('1' + (bm.to() >> 3));
        }
        std::cout << std::endl;

        return 0;
    }

    if (argc >= 2 && std::string(argv[1]) == "--eval") {
        // Evaluate a position from FEN
        std::string fen;
        for (int i = 2; i < argc; i++) {
            if (i > 2) fen += " ";
            fen += argv[i];
        }
        if (fen.empty()) fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
        Board board(fen);
        std::cout << "FEN: " << board.fen() << std::endl;
        std::cout << "Evaluation: " << evaluate(board) << " cp" << std::endl;
        return 0;
    }

    // Default: run UCI
    UCI uci;
    uci.loop();

    return 0;
}
