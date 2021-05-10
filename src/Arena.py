import Connect4
import tqdm
from Connect4Game import Connect4Game
import random
import torch
from datetime import datetime


def arenaWorker(nns):
    old, new = nns

    coinflip = random.random()
    if coinflip < 0.5:
        game = Connect4Game(old, new)
        data = game.runGame()

        oldResult = data[Connect4.RED][0][2]
        newResult = -oldResult

    else:
        game = Connect4Game(new, old)
        data = game.runGame()

        oldResult = data[Connect4.YELLOW][0][2]
        newResult = -oldResult

    # Clamp to 0 because we're only counting wins
    oldResult = max(oldResult, 0)
    newResult = max(newResult, 0)

    return oldResult, newResult, data[0] + data[1]

def Arena(oldNN, newNN, numGames = 50, tournament = False):

    numOldWins = 0
    numNewWins = 0

    outcomes = []
    for i in tqdm.tqdm(range(numGames)):
        outcomes.append(arenaWorker((oldNN, newNN)))

    ds = []
    for o in outcomes:
        numOldWins += o[0]
        numNewWins += o[1]
        for example in o[2]:
            ds.append(example)

    print(f"old wins: {numOldWins}")
    print(f"new wins: {numNewWins}")

    if numOldWins + numNewWins > 0:
        if numNewWins / (numOldWins + numNewWins) >= 0.55:
            print("saving nn")
            # Save the nn to a file
            if not tournament:
                path = f"./{datetime.today().strftime('%Y-%m-%d-%H-%M-%S')}.pt"
            else:
                path = f"./bestCheckersNet{datetime.today().strftime('%Y-%m-%d-%H-%M-%S')}.pt"
            torch.save(newNN.state_dict(), path)

            return True, ds, (numOldWins, numNewWins)
        else:
            return False, ds, (numOldWins, numNewWins)

    else:
        return False, ds, (numOldWins, numNewWins)






