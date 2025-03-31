#include <iostream>
#include <stdio.h>
#include <omp.h>
#include <algorithm>
#include <random>
#include <chrono>
#include <unordered_set>

const intmax_t e = 7;
intmax_t p = 1e9 + 7;

struct Board {
    bool player = true;
    uint8_t last_played = 10;
    uint8_t board[3][3][3][3] = {0};
};

struct Hash {
    intmax_t operator()(const Board &board) {
        intmax_t pow = 1;
        intmax_t output = 0;
        for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) for (int k = 0; k < 3; k++) for (int l = 0; l < 3; l++) {
            output = (output + board.board[i][j][k][l] * pow) % p;
        }
        return output;
    }
};

// Board nextBoard(Board &board) {
//     Board board;

// }

uint8_t evaluateSubboard(uint8_t board[3][3]) {
    for (int i = 0; i < 3; i++) {
        if (board[i][0] == board[i][1] && board[i][1] == board[i][2] && board[i][0] != 0) {
            return board[i][0];
        }
        if (board[0][i] == board[1][i] && board[1][i] == board[2][i] && board[0][i] != 0) {
            return board[0][i];
        }
    }

    if (board[0][0] == board[1][1] && board[1][1] == board[2][2] && board[0][0] != 0) {
        return board[0][0]; 
    }
    if (board[0][2] == board[1][1] && board[1][1] == board[2][0] && board[0][2] != 0) {
        return board[0][2];
    }

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            if (board[i][j] == 0) {
                return 0;
            }
        }
    }

    return 3;
}

uint8_t evaluateBoard(Board &b) {
    uint8_t local_board[3][3];

    for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) {
        auto sb = b.board[i][j];
        local_board[i][j] = evaluateSubboard(sb);
    }

    return evaluateSubboard(local_board);
}

std::unordered_set<Board, Hash> boards;

std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());

int main() {
    omp_set_num_threads(8);

    Board board;

    #pragma omp parallel 
    {
        std::printf("%d \n", evaluateBoard(board));
    }
}