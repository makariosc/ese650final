from alphaZeroNet import ChessNet, train
import utils
import mcts
import torch
import chess
from ChessGame import ChessGame

# best current player plays 25000 games against itself
# uses MCTS to select a move
# at each move, store: game state, search probs from MCTS, who won game (add after game ends)

def selfPlay(num_games = 10000):
    """
    function runs best current player to generate dataset

    Parameters
    ----------
    num_games : TYPE, optional
        DESCRIPTION. The default is 10000.

    Returns
    -------
    None.

    """
    num_games = num_games
    
    pass
    

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
    # path = 'currentNN.pt'  # whatever the current best model is saved as
    
    # model = ChessNet()
    # model.load_state_dict(torch.load(path))
    # model.eval() # make sure model doesn't change
    
    dataset = []
    
    num_games = 25000 # used later
    
    model = ChessNet()
    
    game = ChessGame(model, model)
    data = game.gameLoop()
    
    dataset += data[0]
    dataset += data[1]
    
    train(model, dataset)
    
    # store data from each game state during the game to something (a list?)
    # save this to some file