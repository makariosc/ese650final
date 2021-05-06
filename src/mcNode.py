import collections
import numpy as np
import abc

from utils import *
from copy import deepcopy

class MCNode(abc.ABC):

    #Initializes the node.
    def __init__(self, state):

        #Saves the board state.
        self.state = state

        self.pActs = np.array([])

        #Saves the node's parent and the action the parent took to reach this node. Useful for backpropogation.
        # self.pAct = pAct
        # self.parent = parent

        #Initializes a default dictionary to save the values of "edges" (i.e. state/action pairs) important to this node.
        self.actionDict = collections.Counter()

        # Dictionary mapping move indices (see utils moveToIdx) to Children nodes and action probabilities.
        self.children = {}

        # Something like this. Runs some sort of list of possible child states on node initialization.
        self.PotentialChildren = list(self.state.legal_moves())


    def createChildren(self, pActs):
        self.pActs = pActs
        pActIdxs = np.where(pActs)
        for moveIndex in pActIdxs:
            move = idxToMove(moveIndex)
            self.children[moveIndex] = Action(MCNode(deepcopy(self.state).push(move)), pActs[moveIndex])

    #Checks to see if the node is unexpanded (NOTE: Different from terminal nodes. No children in this case implies unexpanded children.)
    def has_children(self):
        return len(self.children) > 0

    #Picks a random child to expand and expands it.
    #def select_random(self):

class Action:
    def __init__(self, child, P):
        self.P = P
        self.N = 0
        self.W = 0
        self.Q = 0

        self.nextState = child


    

