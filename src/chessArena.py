import numpy as np
import ChessGame
import torch
import chess
import random
from datetime import datetime

import multiprocessing as mp

def arenaWorker(nns):

    old, new = nns
    oldResult = 0
    newResult = 0

    coinFlip = random.random()

    if coinFlip < 0.5:
        # old is white, new is black
        cg = ChessGame.ChessGame(old, new)
        moves = cg.gameLoop()

        oldResult = moves[chess.WHITE][0][2]
        newResult = -oldResult
    else:
        cg = ChessGame.ChessGame(new, old)
        moves = cg.gameLoop()

        oldResult = moves[chess.BLACK][0][2]
        newResult = -oldResult

    ds = []
    ds += moves[0]
    ds += moves[1]

    # Clamp to 0 because we're only counting wins
    oldResult = max(oldResult, 0)
    newResult = max(newResult, 0)

    return (oldResult, newResult), ds

def ChessArena(oldNN, newNN, numGames = 50):

    numOldWins = 0
    numNewWins = 0

    tSet = []

    pool = mp.Pool()
    outcomes = pool.imap_unordered(arenaWorker, [(oldNN, newNN)] * numGames)

    for o in outcomes:
        numOldWins += o[0][0]
        numNewWins += o[0][1]
        for datapoint in o[1]:
            tSet.append(datapoint)

    # for i in range(numGames):
    #     if i % 2 == 0:
    #         cg = ChessGame.ChessGame(oldNN, newNN)
    #         moves = cg.gameLoop()
    #         tSet.append(moves)
    #
    #         if moves[chess.WHITE][0][2] == 1:
    #             numOldWins += 1
    #         elif moves[chess.WHITE][0][2] == -1:
    #             numNewWins += 1
    #     else:
    #         cg = ChessGame.ChessGame(newNN, oldNN)
    #         moves = cg.gameLoop()
    #         tSet.append(moves)
    #
    #         if moves[chess.WHITE][0][2] == 1:
    #             numNewWins += 1
    #         elif moves[chess.WHITE][0][2] == -1:
    #             numOldWins += 1

    #Return condition. Returns the new NN if it won over 55% of its games.
    # Otherwise returns the old one.

    # TRUE if new NN, FALSE otherwise.

    # Also returns training samples generated by the arena.
    if numOldWins + numNewWins > 0:
        if numNewWins / (numOldWins + numNewWins) >= 0.55:

            # Save the nn to a file
            path = f"./{datetime.today().strftime('%Y-%m-%d-%H-%M-%S')}"
            torch.save(newNN.state_dict(), path)

            return True, newNN, []
        else:
            return False, oldNN, []

    else:
        return False, oldNN, []




