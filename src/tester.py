from Grid_3 import Grid
from ComputerAI_3 import ComputerAI
from PlayerAI_extra import PlayerAI
from Displayer_3 import Displayer
import CMAES

import time
import random
import math
import numpy as np
import multiprocessing
import sys

defaultInitialTiles = 2
defaultProbability = 0.9

actionDic = {
    0: "UP",
    1: "DOWN",
    2: "LEFT",
    3: "RIGHT",
    None: "NONE"  # For error logging
}

(PLAYER_TURN, COMPUTER_TURN) = (0, 1)

# Time Limit Before Losing
timeLimit = 3.95
allowance = 0.05
maxTime = timeLimit + allowance


class GameManager:
    def __init__(self, size=4, playerAI=None, computerAI=None, displayer=None):
        self.grid = Grid(size)
        self.possibleNewTiles = [2, 4]
        self.probability = defaultProbability
        self.initTiles = defaultInitialTiles
        self.over = False

        # Initialize the AI players
        self.computerAI = computerAI or ComputerAI()
        self.playerAI = playerAI or PlayerAI()
        self.displayer = displayer or Displayer()

    def updateAlarm(self) -> None:
        """ Checks if move exceeded the time limit and updates the alarm """
        if time.process_time() - self.prevTime > maxTime:
            print("timed out")
            self.over = True

        self.prevTime = time.process_time()

    def getNewTileValue(self) -> int:
        """ Returns 2 with probability 0.95 and 4 with 0.05 """
        return self.possibleNewTiles[random.random() > self.probability]

    def insertRandomTiles(self, numTiles: int):
        """ Insert numTiles number of random tiles. For initialization """
        for i in range(numTiles):
            tileValue = self.getNewTileValue()
            cells = self.grid.getAvailableCells()
            cell = random.choice(cells) if cells else None
            self.grid.setCellValue(cell, tileValue)

    def start(self) -> int:
        """ Main method that handles running the game of 2048 """

        # Initialize the game
        self.insertRandomTiles(self.initTiles)
        self.displayer.display(self.grid)
        turn = PLAYER_TURN  # Player AI Goes First
        self.prevTime = time.process_time()

        while self.grid.canMove() and not self.over:
            # Copy to Ensure AI Cannot Change the Real Grid to Cheat
            gridCopy = self.grid.clone()
            # time.sleep(.1)
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
        # self.displayer.display(self.grid)
        turn = PLAYER_TURN  # Player AI Goes First
        self.prevTime = time.process_time()

        while self.grid.canMove() and not self.over:
            # Copy to Ensure AI Cannot Change the Real Grid to Cheat
            gridCopy = self.grid.clone()
            # time.sleep(.1)
            move = None

            if turn == PLAYER_TURN:
                #print("Player's Turn: ", end="")
                move = self.playerAI.getMove(gridCopy)
                # print(actionDic[move])

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
            # self.displayer.display(self.grid)

            # Exceeding the Time Allotted for Any Turn Terminates the Game
            self.updateAlarm()
            turn = 1 - turn

        return self.grid.getMaxTile()


def main():
    if len(sys.argv) < 2:
        playerAI = PlayerAI()
        computerAI = ComputerAI()
        displayer = Displayer()
        gameManager = GameManager(4, playerAI, computerAI, displayer)
        maxTile = gameManager.start()
        maxHeur = -1 * float("inf")
        maxHeur = max(maxHeur, playerAI.max_heur)
        print("Max Tile: {}  Score: {}".format(maxTile, playerAI.high_score))
        print("Max Heuristic value:", maxHeur)
        print("Weights used:", playerAI.weights)
    elif sys.argv[1] == 'd':
        playerAI = PlayerAI()
        computerAI = ComputerAI()
        displayer = Displayer()
        gameManager = GameManager(4, playerAI, computerAI, displayer)
        maxHeur = -1 * float("inf")
        maxTile = gameManager.start()
        maxHeur = max(maxHeur, playerAI.max_heur)
        print("Max tile:", maxTile)
        print("Max Heuristic value:", maxHeur)
        print("Weights used:", playerAI.weights)
    elif sys.argv[1] == 't':
        tot = 0
        trials = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        scores = []
        max_heur = -float('inf')
        max_high_score = 0
        for _ in range(trials):
            playerAI = PlayerAI()
            computerAI = ComputerAI()
            displayer = Displayer()
            gameManager = GameManager(4, playerAI, computerAI, displayer)
            maxTile = gameManager.start_no_disp()
            memo_dict = playerAI.memo
            scores.append(maxTile)
            print("Max Tile: {}  Score: {}".format(
                maxTile, playerAI.high_score))
            tot += math.log(maxTile, 2)
            max_heur = max(max_heur, playerAI.max_heur)
            max_high_score = max(playerAI.high_score, max_high_score)
        print(tot/trials)
        print(sorted(scores, reverse=True)[0:5])
        print("Max value of heuristic: ", max_heur)
        print("Highest score achieved: ", max_high_score)
        print("number of 1024s: ", scores.count(1024))
        print("number of 2048s: ", scores.count(2048))
        print("number of 4096s: ", scores.count(4096))
        print("number of 8192s: ", scores.count(8192))
        print("Weights used:", playerAI.weights)
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
            for n in range(2, num_vals_for_weights):
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
        file.write('TRIALS ' + str(num_trials*(run_num-1)//runs) + ' TO ' +
                   str(num_trials*run_num//runs) + ' OF ' + str(num_trials) + '\n\n')
        for i in trial_weights[num_trials*(run_num-1)//runs: num_trials*run_num//runs]:
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
                playerAI = PlayerAI(weights)
                computerAI = ComputerAI()
                displayer = Displayer()
                gameManager = GameManager(4, playerAI, computerAI, displayer)
                maxTile = gameManager.start_no_disp()
                scores.append(math.log(maxTile, 2))
                out_str += str(maxTile) + ','
            scores = sorted(scores, reverse=True)[0:trials//2]
            for score in scores:
                out_str += str(score) + ','
            avg_score = sum(scores) * 2 / trials
            out_str += str(avg_score) + '\n'
            file.write(out_str)
            #print("weights: ", weights, " score: ", real_score)
        file.close()
    # CMA-ES
    elif sys.argv[1] == 'c':

        generations = 50
        runs = 10
        samples = 50

        # Generate a random set of weight combinations and their mean
        weight_combinations, means = CMAES.generate_data(
            0, 2000, samples=samples)
        combinations = np.array(weight_combinations)

        print("CMA-ES")
        print("Generations:", generations)
        print("Runs per generation:", runs)
        print("Samples: {} \n".format(samples))

        # transpose so that each row is a weight combination
        combinations = combinations.transpose()
        best_comb = []
        best_avg_comb = []
        best_avg_score = 0
        max_tile = 0
        best_score = 0
        best_weights = []
        good_gen = -1
        good_avg_gen = -1

        # perform CMA-ES
        for generation in range(generations):
            # hash map of avg score which maps to weight combinations
            tracker = {}
            comb = 1  # track which combination we are on
            gen_avg_score = 0
            # Use weight combinations(for loop iterating for weight combinations)
            for combination in combinations:
                avg = 0
                weights = [heur_weight for heur_weight in combination]
                for run in range(runs):
                    # initialization
                    playerAI = PlayerAI(weights=weights)
                    computerAI = ComputerAI()
                    displayer = Displayer()
                    gameManager = GameManager(
                        4, playerAI, computerAI, displayer)
                    maxTile = gameManager.start_no_disp()
                    memo_dict = playerAI.memo

                    # track run results
                    curr_high_score = playerAI.high_score
                    avg += curr_high_score

                    # keep track of highest performance
                    if best_score < curr_high_score:
                        best_score = curr_high_score
                        max_tile = maxTile
                        best_weights = weights
                        good_gen = generation

                avg /= runs

                # best average tracking
                if avg > best_avg_score:
                    best_avg_score = avg
                    best_avg_comb = combination
                    good_avg_gen = generation

                # generation average score tracking
                gen_avg_score += avg

                tracker[avg] = weights
                print("Combination {} of {}".format(comb, samples))
                print("Weights:", weights)
                print("Average Score: {}\n".format(avg))
                comb += 1

            gen_avg_score /= samples

            # 3. take 25 % best average and generate gaussian distribution
            best = round(0.25 * samples)
            cov_matrix, new_means = CMAES.generate_next_generation_data(
                tracker, means, best_samples=best)

            # 4. take N new samples and repeat the process
            new_data = CMAES.generate_normal_distribution(
                means=new_means, cov_matrix=cov_matrix, samples=samples)

            # put new parameters in place
            means = new_means
            combinations = new_data

            # print results from generation
            print("\nGeneration {} of {} finished".format(
                generation + 1, generations))
            print("Generation average:", gen_avg_score)
            check = cov_matrix[0][0] + cov_matrix[1][1] + \
                cov_matrix[2][2] + cov_matrix[3][3]
            print("Current variance among weights: {} \n".format(check))

            # current best
            print("Current Best Results")
            print("Max tile: {}\nHigh Score: {}".format(
                max_tile, best_score))
            print("Best Weight Combination:", best_weights)
            print("Generation:", good_gen + 1)

            # current best average
            print("\nCurrent Best Average Results")
            print("Best average score", best_avg_score)
            print("Best Average Weight Combination:", best_avg_comb)
            print("Generation: {}\n".format(good_avg_gen + 1))

            # check for convergence
            if check < 5:
                print("Weights have converged")
                break

        print("Final Generation:\n {} ".format(combinations))
        CMAES.visualize_covariance_matrix(cov_matrix)
        print("\nCMA-ES finished")

    else:
        playerAI = PlayerAI()
        computerAI = ComputerAI()
        displayer = Displayer()
        gameManager = GameManager(4, playerAI, computerAI, displayer)
        maxTile = gameManager.start()
        print(maxTile)
    # '''
    # '''
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
    # '''


if __name__ == '__main__':
    main()
