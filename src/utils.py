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

# Given a chess.Move object, and the board it came from,
# return the idx in the probability vector
def moveIdx(m, board):
    fromSquare = m.from_square()
    toSquare = m.to_square()

    fromRank, fromFile = chess.square_rank(fromSquare), chess.square_file(fromSquare)
    toRank, toFile = chess.square_rank(toSquare), chess.square_file(toSquare)

    dR = fromRank - toRank
    dF = fromFile - toFile
    dS = chess.square_distance(fromSquare, toSquare)

    # Horsey rules
    if board.piece_at(fromSquare).symbol() == 'N':
        if dR == 2 and dF == 1:
            idx = 56
        elif dR == 1 and dF == 2:
            idx = 57
        elif dR == -1 and dF == 2:
            idx = 58
        elif dR == -2 and dF == 1:
            idx = 59
        elif dR == -2 and dF == -1:
            idx = 60
        elif dR == -1 and dF == -2:
            idx = 61
        elif dR == 1 and dF == -2:
            idx = 62
        elif dR == 2 and dR == -1:
            idx = 63

    # Pawn underpromotions
    elif m.promotion < 5:
        if dF < 0:
            idx = 64 + m.promotion - 2  # 64, 65, 66
        elif dF == 0:
            idx = 67 + m.promotion - 2  # 67, 68, 69
        elif dF > 0:
            idx = 70 + m.promotion - 2  # 70, 71, 72

    # "Queen moves"
    else:
        if dR > 0 and dF == 0:
            idx = 0 + dS - 1
        elif dR > 0 and dF > 0:
            idx = 7 + dS - 1
        elif dR == 0 and dF > 0:
            idx = 14 + dS - 1
        elif dR < 0 and dF > 0:
            idx = 21 + dS - 1
        elif dR < 0 and dF == 0:
            idx = 28 + dS - 1
        elif dR < 0 and dF < 0:
            idx = 35 + dS - 1
        elif dR == 0 and dF < 0:
            idx = 42 + dS - 1
        elif dR > 0 and dF < 0:
            idx = 49 + dS - 1

    return 73 * fromSquare + idx

# Given a board, return a mask that zeros out all illegal moves.
def moveMask(board):
    mv = np.zeros(64 * 73)
    for m in board.legal_moves:
        mv[moveIdx(m, board)] = 1
    return mv

# Given a 64 x 73 movement probability vector and the board it came from,
# return the mpv with the legal move probabilities normalized
def normalizeMPV(mv, board):
    mask = moveMask(board)
    maskedMV = mask * mv

    return maskedMV / np.linalg.norm(maskedMV)






