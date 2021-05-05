import chess
from chess import SquareSet

import numpy as np

def fenToBitBoard(board, player = chess.WHITE):
    boardState = np.zeros((0, 8, 8))

    # Pawn = 1, King = 6
    for currPiece in range(1,7):
        pieceBits = np.array(SquareSet(board.pieces(currPiece, player)).tolist(), dtype = np.uint8).reshape(8, 8)
        pieceBits = np.flip(pieceBits)

        # If the current player is black, then rotate the board 180 degrees.
        if player == chess.BLACK:
            pieceBits = np.rot90(pieceBits, k=2)

        boardState = np.append(boardState, [pieceBits], axis = 0)

    for currPiece in range(1,7):
        pieceBits = np.array(SquareSet(board.pieces(currPiece, not player)).tolist(), dtype = np.uint8).reshape(8, 8)
        pieceBits = np.flip(pieceBits)

        # If the current player is black, then rotate the board 180 degrees.
        if player == chess.BLACK:
            pieceBits = np.rot90(pieceBits, k=2)

        boardState = np.append(boardState, [pieceBits], axis = 0)

    # Returns a 12 x 8 x 8 array of the current board state.
    return boardState

