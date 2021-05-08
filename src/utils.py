import chess
from chess import SquareSet

import numpy as np

# Given a chess.Board(), return a 12 x 8 x 8 stack representing where the pieces are.
def boardToPieceFeatures(board, player = chess.WHITE):
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

# Pieces: from the board
# Repetitions: from the board

# Color: from the board
# Total move count: from board.fullmove_number
# P1 castling (k, q)
# P2 castling (k, q)

# No progress count: from the board
def makeFeatures(board):
    # Location of the pieces on the board
    pieceFeatures = boardToPieceFeatures(board, board.turn)

    # Number of repetitions in this state
    if board.is_repetition(1):
        repetition = np.ones((8,8))
    elif board.is_repetition(2):
        repetition = np.ones((8,8)) * 2
    elif board.is_repetition(3):
        repetition = np.ones((8,8)) * 3
    else:
        repetition = np.zeros((8,8))

    # Current player
    color = np.ones((8,8)) * board.turn

    # total number of moves
    moves = np.ones((8,8)) * board.fullmove_number

    # Castling rights
    p1castlek = np.ones((8,8)) * board.has_kingside_castling_rights(board.turn)
    p1castleq = np.ones((8,8)) * board.has_queenside_castling_rights(board.turn)

    p2castlek = np.ones((8,8)) * board.has_kingside_castling_rights(not board.turn)
    p2castleq = np.ones((8,8)) * board.has_queenside_castling_rights(not board.turn)

    # No progress counter for 50-turn-move
    no_progress_count = np.ones((8,8)) * board.halfmove_clock

    otherFeatures = np.stack((
        repetition,
        color,
        moves,
        p1castlek,
        p1castleq,
        p2castlek,
        p2castleq,
        no_progress_count
    ))

    return np.append(pieceFeatures, otherFeatures, axis=0)


horseyDeltas = {
    56: (2, 1),
    57: (1, 2),
    58: (-1, 2),
    59: (-2, 1),
    60: (-2, -1),
    61: (-1, -2),
    62: (1, -2),
    63: (2, -1)
}

# Given an int from 0 to 64 x 73
# Return the associated move object
def idxToMove(i, board):
    fromSquare = i // 73
    toSquareIdx = i % 73

    fromRank = chess.square_rank(fromSquare)
    fromFile = chess.square_file(fromSquare)

    promotion = None

    if board.turn == chess.WHITE:
        promotionRank = 7
    else:
        promotionRank = 2

    if board.piece_at(fromSquare).piece_type == chess.PAWN and fromRank + 1 == promotionRank:
        promotion = chess.QUEEN

    # Queen moves
    if 0 <= toSquareIdx < 7:
        dR, dF = toSquareIdx % 7 + 1, 0
    elif 7 <= toSquareIdx < 14:
        dR, dF = toSquareIdx % 7 + 1, toSquareIdx % 7 + 1
    elif 14 <= toSquareIdx < 21:
        dR, dF = 0, toSquareIdx % 7 + 1
    elif 21 <= toSquareIdx < 28:
        dR, dF = -(toSquareIdx % 7 + 1), toSquareIdx % 7 + 1
    elif 28 <= toSquareIdx < 35:
        dR, dF = -(toSquareIdx % 7 + 1), 0
    elif 35 <= toSquareIdx < 42:
        dR, dF = -(toSquareIdx % 7 + 1), -(toSquareIdx % 7 + 1)
    elif 42 <= toSquareIdx < 49:
        dR, dF = 0, -(toSquareIdx % 7 + 1)
    elif 49 <= toSquareIdx < 56:
        dR, dF = toSquareIdx % 7 + 1, -(toSquareIdx % 7 + 1)

    # Horsey moves
    elif 56 <= toSquareIdx < 64:
        dR, dF = horseyDeltas[toSquareIdx]

    # Pawn underpromotion moves
    elif 64 <= toSquareIdx < 67:
        if board.turn == chess.WHITE:
            dR, dF = 1, -1
        else:
            dR, dF = -1, -1
        promotion = toSquareIdx % 63 + 2
    elif 67 <= toSquareIdx < 70:
        if board.turn == chess.WHITE:
            dR, dF = 1, 0
        else:
            dR, dF = -1, 0
        promotion = toSquareIdx % 67 + 2
    elif 70 <= toSquareIdx < 73:
        if board.turn == chess.WHITE:
            dR, dF = 1, 1
        else:
            dR, dF = -1, 1
        promotion = toSquareIdx % 70 + 2
    else:
        print("CRASHING")
        return 0

    toSquare = chess.square(fromFile + dF, fromRank + dR)

    return chess.Move(fromSquare, toSquare, promotion)


# Given a chess.Move object, and the board it came from,
# return the idx in the probability vector
def moveIdx(m):
    fromSquare = m.from_square
    toSquareIdx = m.to_square

    fromRank, fromFile = chess.square_rank(fromSquare), chess.square_file(fromSquare)
    toRank, toFile = chess.square_rank(toSquareIdx), chess.square_file(toSquareIdx)

    dR = toRank - fromRank
    dF = toFile - fromFile
    dS = chess.square_distance(fromSquare, toSquareIdx)

    # Horsey rules
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
    elif dR == 2 and dF == -1:
        idx = 63

    # Pawn underpromotions
    elif m.promotion is not None and m.promotion < 5:
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
        else:
            print("CRASHING: Invalid Move")
            return -1

    return 73 * fromSquare + idx

# Given a board, return a mask that zeros out all illegal moves.
def moveMask(board):
    mv = np.zeros(64 * 73)
    idxs = list(map(moveIdx, board.legal_moves))
    mv[idxs] = 1
    return mv

# Given a 64 x 73 movement probability vector and the board it came from,
# return the mpv with the legal move probabilities normalized
def normalizeMPV(mv, board):
    mask = moveMask(board)
    maskedMV = mask * mv.detach().numpy().squeeze()

    return maskedMV / np.linalg.norm(maskedMV)
