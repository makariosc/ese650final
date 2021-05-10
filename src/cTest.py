from Connect4 import Connect4
from Connect4Game import Connect4Game
import OthelloNet

import multiprocessing as mp


def gameWrapper(net):
    game = Connect4Game(net, net)
    return game.runGame()

if __name__ == "__main__":

    nn = OthelloNet.OthelloNNet()
    nn.eval()

    p = mp.Pool(3)

    out = p.map(gameWrapper, [nn] * 6)

    # set = game.runGame()

    # c = Connect4()
    # c.go(0)
    # c.validActions()
    #
    # c.show()

    # assert c.validActions() == list(range(7))
    #
    # c1 = Connect4()
    # c.state[4].append(c.turn)
    # c.state[5].append(c.turn)
    # c.state[2].append(c.turn)
    # c.state[3].append(c.turn)
    #
    # # assert c.gameOver(0,1) == True
    #
    #
    # c2 = Connect4()
    # c2.state[0].append(c2.turn)
    # # assert not c2.gameOver(0,0)
    #
    # c2.state[0].append(c2.turn)
    # # assert not c2.gameOver(1,0)
    #
    # c2.state[0].append(c2.turn)
    # # assert not c2.gameOver(2,0)
    #
    # c2.state[0].append(c2.turn)
    # # assert c2.gameOver(3,0)
    #
    #
    # c3 = Connect4()
    # c3.state[0].append(c3.turn)
    #
    # c3.state[1].append(not c3.turn)
    # c3.state[1].append(c3.turn)
    #
    # c3.state[2].append(not c3.turn)
    # c3.state[2].append(not c3.turn)
    # c3.state[2].append(c3.turn)
    #
    # c3.state[3].append(not c3.turn)
    # c3.state[3].append(not c3.turn)
    # c3.state[3].append(not c3.turn)
    # c3.state[3].append(c3.turn)
    #
    # c3.state[4].append(not c3.turn)
    # c3.state[4].append(not c3.turn)
    # c3.state[4].append(not c3.turn)
    # c3.state[4].append(not c3.turn)
    # c3.state[4].append(c3.turn)
    #
    # c3.state[5].append(not c3.turn)
    # c3.state[5].append(not c3.turn)
    # c3.state[5].append(not c3.turn)
    # c3.state[5].append(not c3.turn)
    # c3.state[5].append(c3.turn)
    # c3.state[5].append(c3.turn)
    #
    # c3.state[6].append(not c3.turn)
    # c3.state[6].append(not c3.turn)
    # c3.state[6].append(not c3.turn)
    # c3.state[6].append(not c3.turn)
    # c3.state[6].append(c3.turn)
    # c3.state[6].append(c3.turn)
    # c3.state[6].append(c3.turn)
    #
    # print(c3.features())
    #
    # # c3.show()
    # #
    # # for i in range(7):
    # #     assert c3.gameOver(i,i)
    #
    # c4 = Connect4()
    #
    # for i in range(6):
    #     for j in range(5 - i):
    #         c4.state[i].append(not c4.turn)
    #     c4.state[i].append(c4.turn)
    #
    # c4.show()
    #
    # assert c4.gameOver(2,3)

    
    
