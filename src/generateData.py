from alphaZeroNet import ChessNet, train
import utils
import mcts
import torch
import chess
import pickle
from ChessGame import ChessGame

import multiprocessing as mp

def gameWorker(nn):
    ds = []

    cg = ChessGame(nn, nn)
    examples = cg.gameLoop()

    ds += examples[0]
    ds += examples[1]

    return ds

# best current player plays 25000 games against itself
# uses MCTS to select a move
# at each move, store: game state, search probs from MCTS, who won game (add after game ends)

def genData(model, num_games = 1000, saveFile = True):
    """
    function runs best current player vs itself to generate dataset

    Parameters
    ----------
    model: current best NN model
    num_games : number of games for NN to play itself. The default is 10000.
    saveFile: flag to save the dataset generated

    Returns
    -------
    None.

    """
    num_games = num_games
    
    ds = []

    pool = mp.Pool()

    # List of list of lists
    # [[[1,2,3],[1,2,3]], [[1,2,3],[1,2,3]]]
    outData = pool.map(gameWorker, [model] * num_games, 25)

    for d in outData:
        for example in d:
            ds.append(example)

    # for i in range(num_games):
    #     game = ChessGame(model, model) # start a new game with current model vs itself
    #     data = game.gameLoop() # play the game
    #
    #     dataset += data[0]
    #     dataset += data[1]

    # save dataset to a .txt fil
    if saveFile:
        with open("current_dataset.txt","wb") as fp:
            pickle.dump(ds,fp)
    
def loadData():
    with open("current_dataset.txt","rb") as fp:
        ds = pickle.load(fp)
    return ds

if __name__=="__main__":
    """
    debugging purposes
    """
    
    print('poop')
    # need to load best current model
    # should be saved from torch.save(model.state_dict(), PATH)
    # loaded with:
    # model = modelclass()
    # model.load_state_dict(torch.load(PATH))
    # model.eval()
    
    
    ## loading a previously saved model
    # path = 'currentNN.pth'  # whatever the current best model is saved as
    # torch.save(model.state_dict(), path)
    
    # model = ChessNet()
    # model.load_state_dict(torch.load(path))
    # model.eval() # make sure model doesn't change
    
    model = ChessNet()
    
    game = ChessGame(model, model)
    
    playGame = False
    
    if playGame:
        data = game.gameLoop()
        
        print('data collected')
        
        dataset = []
        dataset += data[0]
        dataset += data[1]
        
        with open("test.txt","wb") as fp:
            pickle.dump(dataset,fp)
    else:
        with open("test.txt","rb") as fp:
            dataset = pickle.load(fp)
    

    #Convert model to cuda if the device is cuda.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train(model, dataset)
    
    # store data from each game state during the game to something (a list?)
    # save this to some file