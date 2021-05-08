import mcNode as mc
import chess
import utils

from collections import deque
import numpy as np 
import torch

import math
import queue
import random
import copy
import numpy as np
import itertools


class MCTS:

    #Initializes by starting a game from the beginning.
    def __init__(self, game = chess.Board()):
        #Intializes a root with empty parents.
        self.game = game
        self.root = mc.MCNode(game)

    # Picks a move from the current tree
    def select(self, tau = 1, explore = True):
        idxs = list(self.root.children.keys())
        ns = np.array([self.root.children[i].N for i in idxs])
        ns = ns / np.sum(ns)

        if explore:
            ns = ns**(1 / tau)
            nextIdx = random.choices(idxs, ns)[0]
        else:
            nextIdx = np.argmax(ns)

        # nextIdx should be an int from 0 to 64 x 73 - 1.
        # Use utils.idxToMove() to convert to a move.
        return nextIdx, self.root.pActs


    # Expand down into the tree recursively and find a leaf node.
    # When we find the leaf node, query the NN to initialize its children.
    def search(self, node, net):

        print("SEARCH")

        stack = deque()

        while node.has_children():
            a = node.bestAction()
            stack.append(a)
            node = a.nextState

        #Once we reach the leaf node, return the NN's assesment of the current state.

        # result = node.state.outcome()
        # if result is not None:
        #     if result.winner is None:
        #         p, v = np.ones(64 * 73), 0
        #     elif result.winner == (not node.state.turn):
        #         p, v = np.ones(64 * 73), -1
        #     else:
        #         p, v = np.ones(64 * 73), 1
        #
        # else:
        #     # Get the features and upload them to the nn

        # Get the features and upload them to the nn for state evaluation
        stateFeatures = utils.makeFeatures(node.state)

        p, v = net(torch.tensor(stateFeatures))

        if node.state.outcome() is None:
            p = utils.normalizeMPV(p, node.state)
            node.createChildren(p)

        while stack:
            v = -v
            a = stack.pop()
            a.N += 1
            a.W += v
            a.Q = 0 if a.W == 0 else a.N / a.W

        return -v

        






    
