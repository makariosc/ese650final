from Connect4 import Connect4
import torch
import Connect4
import mcts

class Connect4Game:

    def __init__(self, nn1, nn2):
        self.nn1 = nn1
        self.nn2 = nn2

        self.game = Connect4.Connect4()
        self.gameTree = mcts.MCTS(Connect4.Connect4())

        self.moves = {
            Connect4.RED: [],
            Connect4.YELLOW: []
        }

    def selectMove(self, iterations = 25):

#        print("======")
#        self.game.show()

        if self.game.turn == Connect4.RED:
            currNN = self.nn1
        else:
            currNN = self.nn2

        for _ in range(iterations):
            self.gameTree.search(self.gameTree.root, currNN)

        move, pi = self.gameTree.select(1, True)

        self.game.go(move)
        self.moves[not self.game.turn].append(
            [torch.tensor(self.game.features()), torch.tensor(pi), 0]
        )

        if not self.game.finished:
            self.gameTree.root = self.gameTree.root.children[move].nextState


    def runGame(self):
#        print("Starting game.")
        while not self.game.finished:
            self.selectMove()

#        print("======")
#        self.game.show()

        if self.game.winner is None:
#            print(f"Terminated by draw.")
             return self.moves
        else:
#            print(f"Terminated by win.")

            redVal = 1 if self.game.winner == Connect4.RED else -1
            yellowVal = 1 if self.game.winner == Connect4.YELLOW else -1

            for sample in self.moves[Connect4.RED]:
                sample[2] = redVal
            for sample in self.moves[Connect4.YELLOW]:
                sample[2] = yellowVal

            return self.moves


