import numpy as np
import ChessGame

def ChessArena(oldNN, newNN):

    newWin = []









    #Return condition. Returns the new NN if it won over 55% of its games. Otherwise returns the old one.
    if np.sum(newWin)/len(newWin) >= 0.55:
        return newNN

    else:
        return oldNN




