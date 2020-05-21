# Marin Pavlinov Marinov
# File and code reference that achieved 8192

from BaseAI_3 import BaseAI
from time import process_time


class PlayerAI(BaseAI):

    def __init__(self):
        # bounds picked as a result of the highest and lowest values found after 25 runs
        self.upperBound = float('inf')

        # start time will be used to ensure algorithm is system invariant
        self.startTime = 0

    def getMove(self, grid):

        self.startTime = process_time()

        # get the move from search algorithm
        maxScore, move = self.expectiAlphaBeta(grid)

        return maxScore, move

    def getHeuristics(self, grid):
        """ Returns weighted sum of the 5 heuristics """
        return (self.monotonicHeuristic(grid) * 3 + self.openHeuristic(grid) * 4
                + self.mergeHeuristic(grid) + self.patternHeuristic(grid) * 2
                - self.clusterHeuristic(grid) * 5)

    def monotonicHeuristic(self, grid):
        """ This heuristic tries to ensure tiles align monotonically """
        score = 0

        multiplierL = 4
        multiplierR = 4

        for i in range(3):

            # this means the desired pattern stopped holding
            if (multiplierR == 0):
                break

            # these vars will be used for comparison between the values on the upper edge
            currR = grid.getCellValue((0, 3 - i))
            prevR = grid.getCellValue((0, 3 - i - 1))

            if currR >= prevR:
                score += currR * multiplierR
                multiplierR *= 2
            else:
                multiplierR = 0

        return score

    def clusterHeuristic(self, grid):
        """ Heuristic that penalizes tiles that have a big difference with their neightbors """

        penalty = 0
        for i in range(4):
            for j in range(4):
                # check for neighbors...
                neighborUp = grid.getCellValue((i - 1, j))
                neighborDown = grid.getCellValue((i + 1, j))
                neighborLeft = grid.getCellValue((i, j + 1))
                neighborRight = grid.getCellValue((i, j - 1))

                # Find absolute value of their differences
                if neighborUp is not None:
                    penalty = penalty + \
                        abs(grid.getCellValue((i, j)) - neighborUp)
                if neighborDown is not None:
                    penalty = penalty + \
                        abs(grid.getCellValue((i, j)) - neighborDown)
                if neighborLeft is not None:
                    penalty = penalty + \
                        abs(grid.getCellValue((i, j)) - neighborLeft)
                if neighborRight is not None:
                    penalty = penalty + \
                        abs(grid.getCellValue((i, j)) - neighborRight)

        # this will be assigned a negative because we are penalizing
        return penalty

    def patternHeuristic(self, grid):
        """ Heuristic that tries to ensure that the tiles follow a pattern """
        score = 0

        # I want the tiles to keep the highest number in the upper right corner
        score += grid.getCellValue((0, 0)) * 144
        score += grid.getCellValue((0, 1)) * 169
        score += grid.getCellValue((0, 2)) * 196
        score += grid.getCellValue((0, 3)) * 225  # corner
        score += grid.getCellValue((1, 0)) * 121
        score += grid.getCellValue((1, 1)) * 100
        score += grid.getCellValue((1, 2)) * 81
        score += grid.getCellValue((1, 3)) * 64
        score += grid.getCellValue((2, 0)) * 16
        score += grid.getCellValue((2, 1)) * 25
        score += grid.getCellValue((2, 2)) * 36
        score += grid.getCellValue((2, 3)) * 49
        score += grid.getCellValue((3, 0)) * 0
        score += grid.getCellValue((3, 1)) * 1
        score += grid.getCellValue((3, 2)) * 4
        score += grid.getCellValue((3, 3)) * 9

        return score

    def openHeuristic(self, grid):
        """ Heuristic that grants bonuses for the number of available tiles"""
        return len(grid.getAvailableCells()) * 50

    def mergeHeuristic(self, grid):
        """ Heuristics that rewards for the same values next to each other """
        score = 0
        for i in range(4):
            for j in range(4):
                curr = grid.getCellValue((i, j))
                neighborUp = grid.getCellValue((i - 1, j))
                neighborDown = grid.getCellValue((i + 1, j))
                neighborLeft = grid.getCellValue((i, j + 1))
                neighborRight = grid.getCellValue((i, j - 1))

                if (curr == neighborUp or curr == neighborDown or
                        curr == neighborLeft or curr == neighborRight):
                    score += curr * 50

        return score

    def expectimax(self, grid, isMaxPlayer=True, depth=3):
        """ performs expectimax algorithm to determine best move """

        playerMoveset = grid.getAvailableMoves([0, 3, 2, 1])

        # base case
        if (depth == 0 or process_time() - self.startTime > 0.15
                or not playerMoveset):
            return self.getHeuristics(grid), 0

        if isMaxPlayer:
            maxScore = -1 * float("inf")
            for move, gridCopy in playerMoveset:
                currScore, currMove = self.expectimax(
                    gridCopy, False, depth - 1)

                if currScore > maxScore:
                    bestMove = move
                    maxScore = currScore

            return maxScore, bestMove
        else:
            # Ai moveset consists of it putting a 2 or a 4 in any of the availabe cells left
            aiMoveset = [(n, tile) for n in [2, 4]
                         for tile in grid.getAvailableCells()]
            avg = 0
            twos = 0
            fours = 0
            for val, tile in aiMoveset:
                gridCopy = grid.clone()

                if tile and gridCopy.canInsert(tile):
                    gridCopy.setCellValue(tile, val)

                    currScore, currMove = self.expectimax(
                        gridCopy, True, depth - 1)

                    # weight based on probability (90% for 2 and 10% for 4)
                    if val == 2:
                        avg += currScore * 0.9
                        twos += 1
                    else:
                        avg += currScore * 0.1
                        fours += 1

            # return weighted average
            return avg / (0.9 * twos + 0.1 * fours), 0

    def alphaBeta(self, grid, isMaxPlayer=True, depth=4, alpha=-1 * float("inf"), beta=float("inf")):
        """ Performs minimax with alpha beta pruning """

        playerMoveset = grid.getAvailableMoves()

        # base case
        if (depth == 0 or process_time() - self.startTime > 0.15
                or not playerMoveset):
            return self.getHeuristics(grid), 0

        if isMaxPlayer:
            maxScore = -1 * float("inf")
            for move, gridCopy in playerMoveset:
                currScore, currMove = self.alphaBeta(
                    gridCopy, False, depth - 1, alpha, beta)

                if currScore > maxScore:
                    bestMove = move
                    maxScore = currScore

                # alpha-beta pruning check step
                alpha = max(alpha, currScore)
                if beta <= alpha:
                    break

            return maxScore, bestMove
        else:
            aiMoveset = [(n, tile) for n in [2, 4]
                         for tile in grid.getAvailableCells()]
            minScore = float("inf")
            for val, tile in aiMoveset:
                gridCopy = grid.clone()

                if tile and gridCopy.canInsert(tile):
                    gridCopy.setCellValue(tile, val)

                    currScore, currMove = self.alphaBeta(
                        gridCopy, True, depth - 1, alpha, beta)

                    if currScore < minScore:
                        minScore = currScore

                    # alpha-beta prunning check step
                    beta = min(beta, currScore)
                    if beta <= alpha:
                        break

            return minScore, 0

    # The parameter node takes the following values (I found this to perform better than True and False for some reason)
    # 1 = max node
    # 2 = chance node
    def expectiAlphaBeta(self, grid, node=1, depth=3, alpha=-1 * float("inf"), beta=float("inf")):
        """ Performs expectimax with alpha beta pruning """

        # Order of moves (UP, RIGHT, LEFT,DOWN)
        playerMoveset = grid.getAvailableMoves([0, 3, 2, 1])

        # base case
        if (depth == 0 or process_time() - self.startTime > 0.15 or not playerMoveset):
            return self.getHeuristics(grid), 0

        if node == 1:
            maxScore = -1 * float("inf")
            for move, gridCopy in playerMoveset:
                currScore = (self.expectiAlphaBeta(
                    gridCopy, 2, depth - 1, alpha, beta))[0]

                if currScore > maxScore:
                    alpha = currScore
                    bestMove = move
                    maxScore = currScore

            return maxScore, bestMove
        else:
            # Ai moveset consists of it putting a 2 or a 4 in any of the availabe cells left
            aiMoveset = [(n, tile) for n in [2, 4]
                         for tile in grid.getAvailableCells()]

            # get the total amount of weight if we were do got through all the possibilites
            totalWeight = (0.9 * len([move for move in aiMoveset if move[0] == 2])
                           + 0.1 * len([move for move in aiMoveset if move[0] == 4]))
            avg = 0
            currWeight = 0
            for val, tile in aiMoveset:
                gridCopy = grid.clone()

                if tile and gridCopy.canInsert(tile):
                    gridCopy.setCellValue(tile, val)

                    currScore = (self.expectiAlphaBeta(
                        gridCopy, 1, depth - 1, -1 * float("inf"), beta))[0]

                    # weight based on probability (90% for 2 and 10% for 4)
                    if val == 2:
                        avg += currScore * 0.9
                        currWeight += 0.9
                    else:
                        avg += currScore * 0.1
                        currWeight += 0.1

                    # alpha-beta pruning check step
                    beta = (avg + (totalWeight - currWeight)
                            * self.upperBound) / totalWeight
                    if beta <= alpha:
                        break

            # return weighted average
            return avg / currWeight, 0
