import utils
import chess
import mcNode
import numpy as np

import ChessGame

n = mcNode.MCNode(chess.Board())
pi = utils.normalizeMPV(np.ones(64*73), chess.Board())
n.createChildren(pi)

move = chess.Move.from_uci("b1a3")
mi = utils.moveIdx(move)

cg = ChessGame.ChessGame(None, None)
cg.gameLoop()
