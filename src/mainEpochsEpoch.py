from generateData import genData, loadData
from alphaZeroNet import ChessNet, train
import chessArena
import torch
import copy


fromScratch = True # flag if running with no saved NN

if fromScratch:
    # initialize some new network
    player1 = ChessNet()
else:
    player1 = ChessNet()
    player1.load_state_dict(torch.load("bestplayer.pth"))
    
    
player2 = copy.deepcopy(player1)
player2.eval()

numNewPlayers = 10 # end the updates after 10 new better versions have been released

# player1 will be the best current player that we try to train
iters = 0
while iters < numNewPlayers:
    # generate data
    genData(player1, num_games = 1000, saveFile = True)
    dataset = loadData()
    
    train(player1, player1)
    replaced, winner, games = chessArena(player2, player1)
    if replaced:
        player1 = copy.deepcopy(winner)
        player2 = copy.deepcopy(winner)
        player2.eval()
        iters += 1