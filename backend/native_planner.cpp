#include <algorithm>
#include <array>
#include <cstdint>
#include <unordered_map>

namespace {

using Board = std::uint64_t;
std::array<std::uint16_t, 65536> left_rows{};
std::array<std::uint16_t, 65536> right_rows{};
std::array<float, 65536> smoothness{};
std::array<float, 65536> monotonicity{};
std::array<std::uint8_t, 65536> empty_rows{};
bool initialized = false;
std::unordered_map<std::uint64_t, float> cache[5];
constexpr int POSITION_WEIGHTS[16] = {16, 15, 14, 13, 8, 9, 10, 11,
                                      7, 6, 5, 4, 0, 1, 2, 3};

inline int nibble(Board board, int index) {
    return static_cast<int>((board >> (4 * index)) & 15U);
}

inline Board set_nibble(Board board, int index, int value) {
    const Board mask = Board(15) << (4 * index);
    return (board & ~mask) | (Board(value) << (4 * index));
}

std::uint16_t reverse_row(std::uint16_t row) {
    return static_cast<std::uint16_t>(((row & 0x000F) << 12)
        | ((row & 0x00F0) << 4)
        | ((row & 0x0F00) >> 4)
        | ((row & 0xF000) >> 12));
}

void init_tables() {
    if (initialized) return;
    for (int row = 0; row < 65536; ++row) {
        int values[4]{};
        int packed[4]{};
        int count = 0;
        for (int index = 0; index < 4; ++index) {
            values[index] = (row >> (4 * index)) & 15;
            if (values[index]) packed[count++] = values[index];
        }
        int merged[4]{};
        int merged_count = 0;
        for (int index = 0; index < count; ++index) {
            if (index + 1 < count && packed[index] == packed[index + 1]) {
                merged[merged_count++] = packed[index] + 1;
                ++index;
            } else {
                merged[merged_count++] = packed[index];
            }
        }
        std::uint16_t left = 0;
        for (int index = 0; index < merged_count; ++index) {
            left |= static_cast<std::uint16_t>(merged[index] << (4 * index));
        }
        left_rows[row] = left;
        empty_rows[row] = 0;
        for (int index = 0; index < 4; ++index) empty_rows[row] += values[index] == 0;
        float increasing = 0.0F;
        float decreasing = 0.0F;
        smoothness[row] = 0.0F;
        for (int index = 0; index < 3; ++index) {
            if (values[index] && values[index + 1]) {
                smoothness[row] -= std::abs(values[index] - values[index + 1]);
            }
            increasing += std::max(0, values[index + 1] - values[index]);
            decreasing += std::max(0, values[index] - values[index + 1]);
        }
        monotonicity[row] = std::max(increasing, decreasing);
    }
    for (int row = 0; row < 65536; ++row) right_rows[row] = reverse_row(left_rows[reverse_row(static_cast<std::uint16_t>(row))]);
    initialized = true;
}

Board move_board(Board board, int action) {
    Board output = 0;
    if (action == 1 || action == 3) {
        const auto& table = action == 1 ? right_rows : left_rows;
        for (int row = 0; row < 4; ++row) output |= Board(table[(board >> (16 * row)) & 65535U]) << (16 * row);
        return output;
    }
    const auto& table = action == 0 ? left_rows : right_rows;
    for (int column = 0; column < 4; ++column) {
        std::uint16_t line = 0;
        for (int row = 0; row < 4; ++row) line |= static_cast<std::uint16_t>(nibble(board, row * 4 + column) << (4 * row));
        const std::uint16_t moved = table[line];
        for (int row = 0; row < 4; ++row) output = set_nibble(output, row * 4 + column, (moved >> (4 * row)) & 15);
    }
    return output;
}

float heuristic(Board board) {
    int empty = 0;
    int maximum = 0;
    float smooth = 0.0F;
    float monotonic = 0.0F;
    float position = 0.0F;
    for (int row = 0; row < 4; ++row) {
        const auto line = static_cast<std::uint16_t>((board >> (16 * row)) & 65535U);
        empty += empty_rows[line];
        smooth += smoothness[line];
        monotonic += monotonicity[line];
        for (int column = 0; column < 4; ++column) maximum = std::max(maximum, (line >> (4 * column)) & 15);
    }
    for (int index = 0; index < 16; ++index) position += nibble(board, index) * POSITION_WEIGHTS[index];
    for (int column = 0; column < 4; ++column) {
        std::uint16_t line = 0;
        for (int row = 0; row < 4; ++row) line |= static_cast<std::uint16_t>(nibble(board, row * 4 + column) << (4 * row));
        smooth += smoothness[line];
        monotonic += monotonicity[line];
    }
    const float corner = nibble(board, 0) == maximum ? 500.0F : 0.0F;
    return empty * 300.0F + position * 4.0F + monotonic * 20.0F + smooth * 30.0F + corner;
}

float expectimax(Board board, int depth) {
    if (depth <= 0) return heuristic(board);
    const auto found = cache[depth].find(board);
    if (found != cache[depth].end()) return found->second;
    float best = -1.0e30F;
    for (int action = 0; action < 4; ++action) {
        const Board moved = move_board(board, action);
        if (moved == board) continue;
        int empty[16]{};
        int count = 0;
        for (int index = 0; index < 16; ++index) if (!nibble(moved, index)) empty[count++] = index;
        float value = 0.0F;
        if (!count) {
            value = heuristic(moved);
        } else {
            for (int index = 0; index < count; ++index) {
                value += (0.9F * expectimax(set_nibble(moved, empty[index], 1), depth - 1)
                    + 0.1F * expectimax(set_nibble(moved, empty[index], 2), depth - 1)) / count;
            }
        }
        best = std::max(best, value);
    }
    cache[depth][board] = best == -1.0e30F ? heuristic(board) : best;
    return cache[depth][board];
}

}  // namespace

extern "C" int choose_action(const std::uint8_t* cells, int depth) {
    init_tables();
    Board board = 0;
    for (int index = 0; index < 16; ++index) board = set_nibble(board, index, cells[index]);
    depth = std::max(1, std::min(depth, 4));
    float best_value = -1.0e30F;
    int best_action = -1;
    for (int action = 0; action < 4; ++action) {
        const Board moved = move_board(board, action);
        if (moved == board) continue;
        int empty[16]{};
        int count = 0;
        for (int index = 0; index < 16; ++index) if (!nibble(moved, index)) empty[count++] = index;
        float value = 0.0F;
        if (!count) {
            value = heuristic(moved);
        } else {
            for (int index = 0; index < count; ++index) {
                value += (0.9F * expectimax(set_nibble(moved, empty[index], 1), depth - 1)
                    + 0.1F * expectimax(set_nibble(moved, empty[index], 2), depth - 1)) / count;
            }
        }
        if (value > best_value) {
            best_value = value;
            best_action = action;
        }
    }
    return best_action;
}

extern "C" void clear_planner_cache() {
    for (auto& table : cache) table.clear();
}
