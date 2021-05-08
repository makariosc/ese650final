import collections
import numpy as np
import abc

from utils import *
from copy import deepcopy

class MCNode:

    #Initializes the node.
    def __init__(self, state):

        #Saves the board state.
        self.state = state

        self.pActs = np.array([])

        #Initializes a default dictionary to save the values of "edges" (i.e. state/action pairs) important to this node.
        self.actionDict = collections.Counter()

        # Dictionary mapping move indices (see utils moveToIdx) to Children nodes and action probabilities.
        self.children = {}

    def createChildren(self, pActs):
        self.pActs = pActs
        pActIdxs = np.where(pActs)[0]
        for moveIndex in pActIdxs:

            fromSquare = moveIndex // 73
            if self.state.piece_at(fromSquare):
                move = idxToMove(moveIndex, self.state)
                self.children[moveIndex] = Action(MCNode(deepcopy(self.state)), pActs[moveIndex])
                self.children[moveIndex].nextState.state.push(move)


    #Checks to see if the node is unexpanded (NOTE: Different from terminal nodes. No children in this case implies unexpanded children.)
    def has_children(self):
        return len(self.children) > 0

    #Uses PUCT to find the Best Action.
    def bestAction(self):
        puctList = {}

        #Set c to  1 (can be tuned later).
        c = 1

        #Calculates the N sum.
        NList = [self.children[act].N for act in self.children.keys()]
        
        NSum = sum(NList)
        NSum  = np.sqrt(NSum)

        for acts in self.children.keys():
            edge = self.children[acts]

            U = c * edge.P * NSum / (1+edge.N)
            Q = edge.Q

            puctList[acts] = Q+U

        bestAct = max(puctList, key = puctList.get)

        bestEdge = self.children[bestAct]

        return bestEdge

class Action:
    def __init__(self, child, P):
        self.P = P
        self.N = 0
        self.W = 0
        self.Q = 0

        self.nextState = child


    

