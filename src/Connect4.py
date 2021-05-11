from collections import deque
import numpy as np

RED = True
YELLOW = False

class Connect4:
    def __init__(self):
        self.state = [
            deque(),
            deque(),
            deque(),
            deque(),
            deque(),
            deque(),
            deque()
        ]

        # Red goes first
        self.turn = RED
        self.finished = False
        self.winner = None

        self.numMoves = 0

    def validActions(self):
        actions = []
        for i, col in enumerate(self.state):
            if len(col) < 6:
                actions.append(i)

        return actions

    def validActionsMask(self):
        mask = np.zeros(7)
        mask[self.validActions()] = 1

        return mask

    # Checks if position at row and col is part of a connect 4.
    # Row is from 0 to 5
    # Col is from 0 to 6
    def gameOver(self, row, col):

        # Check horizontal
        numAdjacentLeft = 0
        for i in range(col - 1, -1, -1):

            if len(self.state[i]) - 1 >= row:
                if self.state[i][row] == self.turn:
                    numAdjacentLeft += 1
                else:
                    break
            else:
                break

        numAdjacentRight = 0
        for i in range(col + 1, 7):

            if len(self.state[i]) - 1 >= row:
                if self.state[i][row] == self.turn:
                    numAdjacentRight += 1
                else:
                    break
            else:
                break

        if numAdjacentLeft + numAdjacentRight + 1 >= 4:
            self.winner = self.turn
            return True

        # Check vertical
        numVert = 0
        for i in range(row - 1, -1, -1):
            if self.state[col][i] == self.turn:
                numVert += 1
            else:
                break

        if numVert + 1 >= 4:
            self.winner = self.turn
            return True

        # Check positive diagonal left
        numDiagLeftPos = 0
        for i in range(1, 7):
            if col - i >= 0 and row - i >= 0:
                if len(self.state[col - i]) - 1 >= row - i:
                    if self.state[col - i][row - i] == self.turn:
                        numDiagLeftPos += 1
                    else:
                        break
                else:
                    break
            else:
                break

        # Check positive diagonal right
        numDiagRightPos = 0
        for i in range(1, 7):
            if col + i < 7  and row + i < 6:
                if len(self.state[col + i]) - 1 >= row + i:
                    if self.state[col + i][row + i] == self.turn:
                        numDiagRightPos += 1
                    else:
                        break
                else:
                    break
            else:
                break

        if numDiagLeftPos + numDiagRightPos + 1 >= 4:
            self.winner = self.turn
            return True

        # Check negative diagonal left
        numDiagLeftNeg = 0
        for i in range(1, 7):
            if col - i >= 0 and row + i < 7:
                if len(self.state[col - i]) - 1 >= row + i:
                    if self.state[col - i][row + i] == self.turn:
                        numDiagLeftNeg += 1
                    else:
                        break
                else:
                    break
            else:
                break

        # Check negative diagonal right
        numDiagRightNeg = 0
        for i in range(1, 7):
            if col + i < 7  and row - i > 0:
                if len(self.state[col + i]) - 1 >= row - i:
                    if self.state[col + i][row - i] == self.turn:
                        numDiagRightNeg += 1
                    else:
                        break
                else:
                    break
            else:
                break

        if numDiagLeftNeg + numDiagRightNeg + 1 >= 4:
            self.winner = self.turn
            return True

        if self.numMoves == 7 * 6:
            self.winner = None
            return True

        return False

    # Places a piece and checks the pieces around it.
    # Returns whether the piece placement was successful.
    def go(self, action):
        if len(self.state[action]) < 7:
            self.state[action].append(self.turn)

            col = action
            row = len(self.state[col]) - 1

            self.finished = self.gameOver(row, col)
            # self.show()
            self.turn = not self.turn

            self.numMoves += 1

            return True
        else:
            return False

    def show(self):
        for row in range(5, -1, -1):
            rowStr = ""
            for col in range(7):

                if len(self.state[col]) > row:
                    rowStr += str(int(self.state[col][row]))
                else:
                    rowStr += "-"
                rowStr += " "
            print(rowStr)

    def features(self):
        f = np.zeros((7,6))
        for row in range(6):
            for col in range(7):
                if len(self.state[col]) > row:
                    f[col][row] = 2 * (self.state[col][row] - 0.5)

        return np.stack((f, np.ones_like(f) * self.turn))


