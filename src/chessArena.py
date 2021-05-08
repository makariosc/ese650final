import numpy as np
import ChessGame
import torch
import chess
import threading
from datetime import datetime

c = threading.con

def ChessArena(oldNN, newNN, numGames = 50):

    numOldWins = 0
    numNewWins = 0

    tSet = []

    for i in range(numGames):
        if i % 2 == 0:
            cg = ChessGame.ChessGame(oldNN, newNN)
            moves = cg.gameLoop()
            tSet.append(moves)

            if moves[chess.WHITE][0][2] == 1:
                numOldWins += 1
            elif moves[chess.WHITE][0][2] == -1:
                numNewWins += 1
        else:
            cg = ChessGame.ChessGame(newNN, oldNN)
            moves = cg.gameLoop()
            tSet.append(moves)

            if moves[chess.WHITE][0][2] == 1:
                numNewWins += 1
            elif moves[chess.WHITE][0][2] == -1:
                numOldWins += 1

    #Return condition. Returns the new NN if it won over 55% of its games.
    # Otherwise returns the old one.

    # TRUE if new NN, FALSE otherwise.

    # Also returns training samples generated by the arena.
    if numNewWins / (numOldWins + numNewWins) >= 0.55:

        # Save the nn to a file
        path = f"./{datetime.today().strftime('%Y-%m-%d-%H-%M-%S')}"
        torch.save(newNN.state_dict(), path)

        return True, newNN, tSet

    else:
        return False, oldNN, tSet




