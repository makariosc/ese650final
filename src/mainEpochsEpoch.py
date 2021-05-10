from OthelloNet import OthelloNNet, train
import torch
import copy
import Arena

fromScratch = True # flag if running with no saved NN

if fromScratch:
    # initialize some new network
    player1 = OthelloNNet()
else:
    player1 = OthelloNNet()
    player1.load_state_dict(torch.load("bestplayer.pt"))

player2 = copy.deepcopy(player1)

player1.eval()
player2.eval()

numNewPlayers = 10 # end the updates after 10 new better versions have been released

# player1 will be the best current player that we try to train
iters = 0
while iters < numNewPlayers:
    # generate data
    _, dataset = Arena.Arena(player1, player1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    player1.to(device)

    train(player1, dataset)

    player1.to('cpu')

    replaced, winner, games = Arena.Arena(player2, player1, 8)
    if replaced:
        player1 = copy.deepcopy(winner)
        player2 = copy.deepcopy(winner)
        player2.eval()
        iters += 1
