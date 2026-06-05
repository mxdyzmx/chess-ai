#ifndef CHESS_SEARCH_H
#define CHESS_SEARCH_H

#include "board.h"
#include "movegen.h"
#include <atomic>
#include <chrono>

namespace engine {

// Transposition table
enum TTFlag : uint8_t { TT_EXACT, TT_ALPHA, TT_BETA };

struct TTEntry {
    uint64_t hash;   // full hash for verification
    int16_t score;   // score (capped for mate)
    uint16_t move;   // best move
    uint8_t depth;   // search depth
    uint8_t flag;    // TTFlag
    uint8_t age;     // for aging
};

constexpr int TT_MATE_SCORE = 32000;
constexpr int TT_MAX_PLY = 256;
constexpr int TT_MIN_SCORE = -TT_MATE_SCORE + TT_MAX_PLY;
constexpr int TT_MAX_SCORE = TT_MATE_SCORE - TT_MAX_PLY;

class TranspositionTable {
public:
    TranspositionTable() : entries_(nullptr), size_(0), mask_(0) {}
    ~TranspositionTable() { delete[] entries_; }

    void resize(size_t mb);
    void clear();
    void prefetch(uint64_t hash) const;

    TTEntry* probe(uint64_t hash) const;
    void store(uint64_t hash, int score, Move move, int depth, int flag, int ply);

    size_t size() const { return size_; }
    size_t used() const;

    // Reset age for new search
    void new_search() { age_++; }

private:
    TTEntry* entries_;
    size_t size_;
    size_t mask_;
    uint8_t age_ = 0;
};

// Search parameters
struct SearchParams {
    int depth = 64;          // max search depth
    int movetime = 0;        // fixed time per move (ms)
    int wtime = 0, btime = 0; // time remaining
    int winc = 0, binc = 0;  // increment per move
    int movestogo = 0;       // moves until time control
    bool infinite = false;   // search until "stop"
    int tt_size_mb = 256;    // TT size in MB
    int threads = 1;         // number of search threads
};

// Search statistics
struct SearchStats {
    uint64_t nodes = 0;
    uint64_t qnodes = 0;
    uint64_t tt_hits = 0;
    uint64_t null_cutoffs = 0;
    int depth_reached = 0;
    int sel_depth = 0;
    int64_t total_time_ms = 0;

    void reset() {
        nodes = qnodes = tt_hits = null_cutoffs = 0;
        depth_reached = sel_depth = 0;
        total_time_ms = 0;
    }
};

// Search state (per-thread)
class Search {
public:
    Search();
    ~Search() = default;

    // Set position and search
    void set_position(const Board& board) { root_ = board; }
    void set_params(const SearchParams& params) { params_ = params; }

    // Start search (blocking)
    void search();

    // Stop search
    void stop() { stop_.store(true); }
    bool is_searching() const { return searching_.load(); }

    // Results
    Move best_move() const { return best_move_; }
    int best_score() const { return best_score_; }
    const SearchStats& stats() const { return stats_; }

    // TT access
    static TranspositionTable& tt() { static TranspositionTable tt_; return tt_; }

private:
    Board root_;
    SearchParams params_;
    std::atomic<bool> stop_{false};
    std::atomic<bool> searching_{false};
    Move best_move_;
    int best_score_;
    SearchStats stats_;
    int64_t start_time_;
    int64_t time_limit_;
    int prev_estimate_;

    // Time management
    bool time_over();
    int64_t get_time_ms();
};

} // namespace engine

#endif // CHESS_SEARCH_H
