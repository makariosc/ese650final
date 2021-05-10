from alphaZeroNet import ChessNet, train
import torch
import copy
import Arena

from datetime import datetime


if __name__ == "__main__":

    datestring = datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
    logpath = f"./trainingLog{datestring}.txt"
    logFile = open(logpath, "w")
    logFile.write("Beginning training.")
    logFile.close()

    import torch.multiprocessing as mp
    mp.set_start_method('spawn')
    mp.set_sharing_strategy('file_system')

    fromScratch = True # flag if running with no saved NN

    if fromScratch:
        # initialize some new network
        player1 = ChessNet()
    else:
        player1 = ChessNet()
        player1.load_state_dict(torch.load("bestplayer.pt"))

    player2 = type(player1)()
    player2.load_state_dict(player1.state_dict())

    player1.eval()
    player2.eval()

    numNewPlayers = 10 # end the updates after 10 new better versions have been released

    # player1 will be the best current player that we try to train
    iters = 0
    dataset = []
    while True:
        # generate data
        _, ds = Arena.Arena(player1, player1, 100)
        dataset += ds

        print("Done generating dataset.")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        player1.to(device)

        train(player1, dataset)

        player1.to('cpu')
        player1.eval()

        if iters % 1 == 0:
            replaced, dataset = Arena.Arena(player2, player1, 50, tournament=True)
            if replaced:
                logFile = open(logpath, "a")
                logFile.write(f"iteration {i}: Current NN replaced.")
                logFile.close()

                player2.load_state_dict(player1.state_dict())
                player2.eval()
            else:
                logFile = open(logpath, "a")
                logFile.write(f"iteration {i}: Current NN not replaced.")
                logFile.close()
        iters += 1
