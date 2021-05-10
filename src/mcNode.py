import collections
import numpy as np
import abc

from copy import deepcopy

class MCNode:

    #Initializes the node.
    def __init__(self, state):

        #Saves the board state.
        self.state = state
        self.pActs = np.array([])

        self.children = [None] * 7

    def createChildren(self, pActs):
        self.pActs = pActs
        pActIdxs = np.where(pActs)[0]
        for moveIndex in pActIdxs:
            self.children[moveIndex] = Action(MCNode(deepcopy(self.state)), pActs[moveIndex])
            self.children[moveIndex].nextState.state.go(moveIndex)


    #Checks to see if the node is unexpanded (NOTE: Different from terminal nodes. No children in this case implies unexpanded children.)
    def has_children(self):
        return any(self.children)

    #Uses PUCT to find the Best Action.
    def bestAction(self):
        puctList = {}

        #Set c to  1 (can be tuned later).
        c = 1

        #Calculates the N sum.
        NList = [self.children[act].N if self.children[act] is not None else 0 for act in range(len(self.children))]
        
        NSum = sum(NList)
        NSum  = np.sqrt(NSum)

        for act in range(len(self.children)):
            edge = self.children[act]

            if edge is not None:

                U = c * edge.P * NSum / (1+edge.N)
                Q = edge.Q

                puctList[act] = Q+U
            else:
                puctList[act] = -float('inf')

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


    

