from alphaZeroNet import ChessNet, train
import torch
import copy
import Arena


if __name__ == "__main__":

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
        _, ds = Arena.Arena(player1, player1, 150)
        dataset += ds

        print("Done generating dataset.")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        player1.to(device)

        train(player1, dataset)

        player1.to('cpu')
        player1.eval()

        if iters % 5 == 0:
            replaced, dataset = Arena.Arena(player2, player1, 100, tournament=True)
            if replaced:
                player2.load_state_dict(player1.state_dict())
                player2.eval()
        iters += 1
