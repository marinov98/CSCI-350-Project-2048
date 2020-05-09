from Grid_3       import Grid
from ComputerAI_3 import ComputerAI
from PlayerAI_extra import PlayerAI
from Displayer_3  import Displayer

import time
import random
import math
import multiprocessing
import sys

defaultInitialTiles = 2
defaultProbability  = 0.9

actionDic = {
    0: "UP",
    1: "DOWN",
    2: "LEFT",
    3: "RIGHT",
    None: "NONE" # For error logging
}

(PLAYER_TURN, COMPUTER_TURN) = (0, 1)

# Time Limit Before Losing
timeLimit = 2
allowance = 0.05
maxTime   = timeLimit + allowance

class GameManager:
    def __init__(self, size=4, playerAI=None, computerAI=None, displayer=None):
        self.grid = Grid(size)
        self.possibleNewTiles = [2, 4]
        self.probability = defaultProbability
        self.initTiles   = defaultInitialTiles
        self.over        = False

        # Initialize the AI players
        self.computerAI = computerAI or ComputerAI()
        self.playerAI   = playerAI   or PlayerAI()
        self.displayer  = displayer  or Displayer()

    def updateAlarm(self) -> None:
        """ Checks if move exceeded the time limit and updates the alarm """
        if time.process_time() - self.prevTime > maxTime:
            print("timed out")
            self.over = True

        self.prevTime = time.process_time()

    def getNewTileValue(self) -> int:
        """ Returns 2 with probability 0.95 and 4 with 0.05 """
        return self.possibleNewTiles[random.random() > self.probability]

    def insertRandomTiles(self, numTiles:int):
        """ Insert numTiles number of random tiles. For initialization """
        for i in range(numTiles):
            tileValue = self.getNewTileValue()
            cells     = self.grid.getAvailableCells()
            cell      = random.choice(cells) if cells else None
            self.grid.setCellValue(cell, tileValue)

    def start(self) -> int:
        """ Main method that handles running the game of 2048 """

        # Initialize the game
        self.insertRandomTiles(self.initTiles)
        self.displayer.display(self.grid)
        turn          = PLAYER_TURN # Player AI Goes First
        self.prevTime = time.process_time()

        while self.grid.canMove() and not self.over:
            # Copy to Ensure AI Cannot Change the Real Grid to Cheat
            gridCopy = self.grid.clone()
            #time.sleep(.1)
            move = None

            if turn == PLAYER_TURN:
                print("Player's Turn: ", end="")
                move = self.playerAI.getMove(gridCopy)
                print(actionDic[move])

                # If move is valid, attempt to move the grid
                if move != None and 0 <= move < 4:
                    if self.grid.canMove([move]):
                        self.grid.move(move)

                    else:
                        print("Invalid PlayerAI Move - Cannot move")
                        self.over = True
                else:
                    print("Invalid PlayerAI Move - Invalid input")
                    self.over = True
            else:
                print("Computer's turn: ")
                move = self.computerAI.getMove(gridCopy)

                # Validate Move
                if move and self.grid.canInsert(move):
                    self.grid.setCellValue(move, self.getNewTileValue())
                else:
                    print("Invalid Computer AI Move")
                    self.over = True

            # Comment out during heuristing optimizations to increase runtimes.
            # Printing slows down computation time.
            self.displayer.display(self.grid)

            # Exceeding the Time Allotted for Any Turn Terminates the Game
            self.updateAlarm()
            turn = 1 - turn

        return self.grid.getMaxTile()

    def start_no_disp(self) -> int:
        """ Main method that handles running the game of 2048 """

        # Initialize the game
        self.insertRandomTiles(self.initTiles)
        #self.displayer.display(self.grid)
        turn          = PLAYER_TURN # Player AI Goes First
        self.prevTime = time.process_time()

        while self.grid.canMove() and not self.over:
            # Copy to Ensure AI Cannot Change the Real Grid to Cheat
            gridCopy = self.grid.clone()
            #time.sleep(.1)
            move = None

            if turn == PLAYER_TURN:
                #print("Player's Turn: ", end="")
                move = self.playerAI.getMove(gridCopy)
                #print(actionDic[move])

                # If move is valid, attempt to move the grid
                if move != None and 0 <= move < 4:
                    if self.grid.canMove([move]):
                        self.grid.move(move)

                    else:
                        #print("Invalid PlayerAI Move - Cannot move")
                        self.over = True
                else:
                    #print("Invalid PlayerAI Move - Invalid input")
                    self.over = True
            else:
                #print("Computer's turn: ")
                move = self.computerAI.getMove(gridCopy)

                # Validate Move
                if move and self.grid.canInsert(move):
                    self.grid.setCellValue(move, self.getNewTileValue())
                else:
                    #print("Invalid Computer AI Move")
                    self.over = True

            # Comment out during heuristing optimizations to increase runtimes.
            # Printing slows down computation time.
            #self.displayer.display(self.grid)

            # Exceeding the Time Allotted for Any Turn Terminates the Game
            self.updateAlarm()
            turn = 1 - turn

        return self.grid.getMaxTile()

def main():
    if len(sys.argv) < 2:
        playerAI    = PlayerAI()
        computerAI  = ComputerAI()
        displayer   = Displayer()
        gameManager = GameManager(4, playerAI, computerAI, displayer)
        maxTile     = gameManager.start()
        maxHeur = -1 * float("inf")
        maxHeur = max(maxHeur, playerAI.max_heur)
        print(maxTile)
        print(maxHeur)
    elif sys.argv[1] == 'd':
        playerAI    = PlayerAI()
        computerAI  = ComputerAI()
        displayer   = Displayer()
        gameManager = GameManager(4, playerAI, computerAI, displayer)
        maxHeur = -1 * float("inf")
        maxTile = gameManager.start()
        maxHeur = max(maxHeur, playerAI.max_heur)
        print(maxTile)
        print(maxHeur)
    elif sys.argv[1] == 't':
        tot = 0
        trials = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        scores = []
        max_heur = -float('inf')
        for _ in range(trials):
            playerAI    = PlayerAI()
            computerAI  = ComputerAI()
            displayer   = Displayer()
            gameManager = GameManager(4, playerAI, computerAI, displayer)
            maxTile     = gameManager.start_no_disp()
            memo_dict = playerAI.memo
            scores.append(maxTile)
            print(maxTile)
            tot += math.log(maxTile, 2)
            max_heur = max(max_heur, playerAI.max_heur)
        print(tot/trials)
        print(sorted(scores, reverse=True)[0:5])
        print("Max value of heuristic: ", max_heur)
        print("number of 1024s: ", scores.count(1024))
        print("number of 2048s: ", scores.count(2048))
        print("number of 4096s: ", scores.count(4096))
        print("number of 8192s: ", scores.count(8192))
    elif sys.argv[1] == 's':
        num_vals_for_weights = 6
        trial_weights_temp = list(range(pow(num_vals_for_weights, 3)))
        # trial_weights_temp = [73,103,121,122,133,140,151,163,169,177,187,191,193,201,212]
        trial_weights = []
        for i in trial_weights_temp:
            weights = [0]*4
            i_copy = i
            for j in range(3):
                weights[j] = i % num_vals_for_weights
                i = i // num_vals_for_weights
            for n in range(2,num_vals_for_weights):
                for j in range(3):
                    if weights[j] % n != 0:
                        break
                else:
                    break
            else:
                trial_weights.append(i_copy)
        num_trials = len(trial_weights)
        runs = int(sys.argv[2])
        run_num = int(sys.argv[3])
        filename = 'output_' + str(run_num)+'_of_'+str(runs) + '.txt'
        file = open(filename, 'w')
        file.write('RUN ' + str(run_num) + ' OF ' + str(runs) + '\n')
        file.write('TRIALS ' + str(num_trials*(run_num-1)//runs) + ' TO ' + str(num_trials*run_num//runs) + ' OF ' + str(num_trials) + '\n\n')
        for i in trial_weights[num_trials*(run_num-1)//runs : num_trials*run_num//runs]:
            out_str = ''
            weights_as_int = i
            out_str += str(weights_as_int) + ','
            weights = [0]*4
            for j in range(3):
                weights[j] = i % num_vals_for_weights
                i = i // num_vals_for_weights
                out_str += str(weights[j]) + ','
            #print("weights: ", weights, " num: ", weights_as_int)
            trials = 20
            scores = []
            for _ in range(trials):
                playerAI    = PlayerAI(weights)
                computerAI  = ComputerAI()
                displayer   = Displayer()
                gameManager = GameManager(4, playerAI, computerAI, displayer)
                maxTile     = gameManager.start_no_disp()
                scores.append(math.log(maxTile,2))
                out_str += str(maxTile) + ','
            scores = sorted(scores, reverse=True)[0:trials//2]
            for score in scores:
                out_str += str(score) + ','
            avg_score = sum(scores) * 2 / trials
            out_str += str(avg_score) + '\n'
            file.write(out_str)
            #print("weights: ", weights, " score: ", real_score)
        file.close()
    else:
        playerAI    = PlayerAI()
        computerAI  = ComputerAI()
        displayer   = Displayer()
        gameManager = GameManager(4, playerAI, computerAI, displayer)
        maxTile     = gameManager.start()
        print(maxTile)
    #'''
    #'''
    '''
    goodOnes = []
    rlygoodones = []
    def multiprocessing_func(i):
        weights = [0]*4
        x = i
        for j in range(3):
            weights[j] = i % 5
            i = i // 5
        #print("weights: ", weights, " num: ", x)
        trials = 10
        scores = []
        for _ in range(trials):
            playerAI    = PlayerAI(weights)
            computerAI  = ComputerAI()
            displayer   = Displayer()
            gameManager = GameManager(4, playerAI, computerAI, displayer)
            maxTile     = gameManager.start_no_disp()
            scores.append(math.log(maxTile,2))
            #print(maxTile)
        real_score = sum(sorted(scores, reverse=True)[0:trials//2]) * 2 / trials
        if real_score >= 10.0:
            rlygoodones.append((x,weights))
        elif real_score >= 9.8:
            goodOnes.append((x,weights))
        print("weights: ", weights, " num: ", x, " score: ", real_score)
    trial_weights = range(125)
    processes = []
    for i in trial_weights:
        p = multiprocessing.Process(target=multiprocessing_func, args=(i,))
        processes.append(p)
        p.start()
    for process in processes:
        process.join()
    num_trials = len(trial_weights)
    print("at least 10: ", rlygoodones)
    print("at least 9.8: ", goodOnes)
    #'''
    #'''


if __name__ == '__main__':
    main()
