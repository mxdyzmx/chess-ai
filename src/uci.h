#ifndef CHESS_UCI_H
#define CHESS_UCI_H

#include <string>
#include "board.h"
#include "search.h"

namespace engine {

class UCI {
public:
    UCI();
    ~UCI() = default;

    // Main loop
    void loop();

private:
    Search search_;
    Board board_;

    // UCI commands
    void cmd_uci();
    void cmd_isready();
    void cmd_setoption(const std::string& name, const std::string& value);
    void cmd_position(const std::string& input);
    void cmd_go(const std::string& input);
    void cmd_stop();
    void cmd_quit();
    void cmd_ucinewgame();

    // Parse helpers
    Move parse_move(const std::string& str);
    std::string move_to_string(Move m);
};

} // namespace engine

#endif // CHESS_UCI_H
