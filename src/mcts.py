import mcNode as mc
import chess

from collections import defaultdict
import numpy as np 
import torch as th

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
        self.root = mc.MCNode(game)

    # Picks a move from the current tree
    def select(self, tau = 1, explore = True):
        idxs = self.root.children.keys()
        ns = np.array([self.root.children[i].N for i in idxs])
        ns = ns / np.sum(ns)

        if explore:
            ns = ns**(1 / tau)
            nextIdx = random.choice(idxs, ns)
        else:
            nextIdx = np.argmax(ns)

        # nextIdx should be an int from 0 to 64 x 73 - 1.
        # Use utils.idxToMove() to convert to a move.
        return nextIdx, self.root.pActs


    # Expand down into the tree recursively and find a leaf node.
    # When we find the leaf node, query the NN to initialize its children.
    def search(self, node, net):

        #Once we reach the leaf node, return the NN's assesment of the current state.
        if not node.has_children():
            p, v = net(node.state)

            node.createChildren(p)
            return -v

        a = node.actionWithHighestValue()
        v = self.search(a)

        # Backpropagation step
        a.N += 1
        a.W += v
        a.Q = a.N / a.W

        return -v

        






    
