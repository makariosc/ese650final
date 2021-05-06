import collections
import abc 

class mcNode(abc.ABC):

    #Initializes the node.
    def __init__(self, state, parent, pAct):

        #Saves the board state.
        self.state = state

        #Saves the node's parent and the action the parent took to reach this node. Useful for backpropogation.
        self.pAct = pAct
        self.parent = parent

        #Initializes a default dictionary to save the values of "edges" (i.e. state/action pairs) important to this node.
        self.actionDict = collections.Counter()

        #Initializes Children. Probably just a list of nodes. Starts empty because nodes are unexpanded.
        self.children = []

        #Something like this. Runs some sort of list of possible child states on node initialization.
        self.PotentialChildren = list(self.state.legal_moves())


    #Checks to see if the node is unexpanded (NOTE: Different from terminal nodes. No children in this case implies unexpanded children.)
    def has_children(self):
        if len(self.children) > 0:
            return False
        else:
            return True

    #Picks a random child to expand and expands it.
    #def select_random(self):



    

