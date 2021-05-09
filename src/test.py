from alphaZeroNet import ChessNet
import utils
import chess
import mcNode
import numpy as np

import torch

import ChessGame

n = mcNode.MCNode(chess.Board())
pi = utils.normalizeMPV(torch.ones(64*73), chess.Board())
n.createChildren(pi)

move = chess.Move.from_uci("b1a3")
mi = utils.moveIdx(move)

nn = ChessNet()
nn.load_state_dict(torch.load("./genDataModel.pt"))

cg = ChessGame.ChessGame(nn, nn)
s = cg.gameLoop()
