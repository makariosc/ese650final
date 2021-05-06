import ChessGame as cg
import mcNode as mc

from collections import defaultdict
import numpy as np 
import torch as th

import math
import queue
import random
import copy
import numpy as np
import itertools


class MCTS(object):

    #Initializes by starting a game from the beginning.
    def __init__(self):
        self.game = cg.ChessGame()

        #Intializes a root with empty parents.
        self.root = mc.mcNode(self.game.get_board(), None, None)

    #Setting it to random expansion for now. Early iterations can be random but later ones will need PUCT. Selection assumes that this node has children.
    def select(self,node):

        cur_node = node
        #Can only run selection for nodes that have been explored.
        while(cur_node.has_children() == True):

            #Just doing it randomly for now.
            cNodes = cur_node.children()

            move = random.choice(cNodes)

            #Pushes the game-state forward by the move.
            self.game.get_board().push(move)

            #How should we link the nodes to their children? Another dict?
            cur_node = cur_node.children[]




    #Nodes that are "selected" and not "expanded" can be left without parents (parent and pAct are only used for backprop)

    def expand(self):

        #Picks a child. Should just be done randomly for now as a proof of concept. We'll change this later. 
        
        #Expansion precedes simulation, so we should be creating new boards instead of updating ours.


        #Starts a simulation using the chosen expansion node.
        self.simulate(child)

    def simulate(self,node):

        #Once we reach the leaf node, begins backpropogation.
        if (node.state.is_checkmate() == True or node.state.is_stalemate() == True):
            #p, v = NN Magic
            p, v = 0,0

            #Begins backpropogation step.
            self.backProp(node, p, v)

        #If we're not in the backprop phase, we keep going.

        #TODO: The keep going part
        else:
            print('I put this here so python wouldnt give me a syntax error')


    #Backpropogation algorithm. Starts at the leaf node and moves upwards to update.
    def backProp(self,leaf, p, v):

        curNode = leaf
        
        #Makes sure to break when we've reached the root. Might need to update criteria (or reset the root) as the game goes on.
        while(curNode.parent != None):

            #Moves up to the parent node.
            upNode = curNode.parent

            #Updates N, W, Q, for the parent and action that led to the leaf.
            upNode.actionDict[(curNode.pAct, 'N')] += 1
            upNode.actionDict[(curNode.pAct, 'W')] += v 
            upNode.actionDict[(curNode.pAct, 'Q')] = (curNode.pAct, 'W') / (curNode.pAct, 'N')

            '''Don't think probability backprops the same way the rest do, unsure how to work it in.'''

            curNode = upNode
        






    
