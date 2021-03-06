if __name__ == "__main__":

    from generateData import genData, loadData
    from alphaZeroNet import ChessNet, train
    import chessArena
    import torch
    import copy
    from chessArena import ChessArena

    import torch.multiprocessing as mp
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.multiprocessing.set_start_method("spawn")

    fromScratch = True # flag if running with no saved NN

    if fromScratch:
        # initialize some new network
        player1 = ChessNet()
    else:
        player1 = ChessNet()
        player1.load_state_dict(torch.load("2021-05-09-04-42-48.pt"))



    player2 = copy.deepcopy(player1)
    player2.eval()


    numNewPlayers = 10 # end the updates after 10 new better versions have been released

    # player1 will be the best current player that we try to train
    iters = 0
    while iters < numNewPlayers:
        # generate data

        genData(player1, num_games = 32, saveFile = True)
        dataset = loadData()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        player1.to(device)

        train(player1, dataset)

        player1.to('cpu')

        replaced, winner, games = ChessArena(player2, player1, 1)
        if replaced:
            player1 = copy.deepcopy(winner)
            player2 = copy.deepcopy(winner)
            player2.eval()
            iters += 1
