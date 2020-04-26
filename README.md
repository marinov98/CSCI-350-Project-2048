# CSCI-350-Project-2048

## AI Final Project
- Aryan Bhatt
- Owen Kunhardt
- Marin Marinov

## Project Name: A Multifaceted Approach to 2048

### Project Description: 
Algorithmic project, exploring approaches to the game 2048 through the lens of different topics throughout AI. Improvements will be made to the algorithm used in Homework 2 and other algorithms will be tested. These methods will be compared and evaluated.

### Included Topics:  
Agents, Heuristic Search, Adversarial Search, Games, Reasoning under uncertainty, Neural Networks, Markov Decision Processes, Reinforcement Learning

### Real-world Application: 
Well performing modified search algorithms from this project could have use in other games or applications. This could also give us insight into how problems that would normally be restricted to one type of algorithm (adversarial search/gameplay) might respond to various other families of algorithms.

## Contributions:

### All Contribution:
- Modify provided files to better support different algorithms and time limits as well as save metrics of a run to train on 
- Write about own methods and results in report
- Analyze results
- Make video
- Work on final presentation
- Present part of presentation

### Aryan and Owen Combined Contribution:
- Markov decision processes, machine learning, and neural networks
- Find state representation so we can pose it as a Markov Decision Problem
- Get a set of heuristics that describe the general qualities of the board
- Run an autoencoder on the heuristics
- Transforms into a small set of numbers
- Discretize those and use them as the state
- Consider compression methods

### Aryan Contribution:
- Reinforcement learning and machine learning
- Apply policy iteration, value iteration, and Q learning, treating 2048 as an MDP
- Neural network (or possibly other similar regressors) as a heuristic
run trials at each state store state representation (or maybe just heuristic values)
at end of run, update heuristic weights depending on final score
train neural net to predict final score from heuristic values
- Implement iterative updating algorithms to modify heuristic values until they converge to the true utility function
initialize heuristic to a simple representation like sum or maxTile
on each max node, after you find the max heuristic value for the children of that node, update weights to make the heuristic value of that node closer to the max val of its children
same for min nodes
- Do the same thing, but with more advanced machine learning techniques

### Owen Contribution:
- Reinforcement learning, machine learning, and neural networks
- Apply deep reinforcement learning using Pytorch
- Train neural network to predict usual outcome based on state representation
- Transfer learning
Apply if time
- Incorporate previously used algorithms, using the score that an already established algorithm gets starting at a given state 
More specifically, for every state, run a different algorithm that worked well starting at that state 10 times and use the avg score as the true score of that state

### Marin Contribution:
- Heuristic Search, games, adversarial search, machine learning, and reasoning under uncertainty
- Improve move ordering by using previous iterations to motivation which nodes to explore first, making pruning more efficient
- Try to find a better heuristic by posing the static evaluation as a search problem.
- Find the shortest path that goes to each tile in decreasing value order
- For example, a path starts at 1024, then goes to 512, then 128, then 128, etc
- Alternatively, assume no more computer player
- Then find shortest number of moves/tile additions needed to upgrade to next maxTile value
- Evolutionary algorithms to optimize heuristic
- Use covariance matrix adaptation evolution strategy (CMA-ES) to optimize weights 
- Tweak heuristics using brute force or (very crude) gradient descent to further improve scores from 2048.

### Evaluation:
- Record various metrics such as time, ram usage, and max tile when game is over 
- Check for statistical significance between the performance metrics for each of the algorithms

### Deliverables:
- All code as well as report will be available on Github
- 30 page report will contain all documentation for using the program, describe all methods used, and evaluation of results using various metrics
- Video will show each method playing 2048 with voiceover explaining the method, the results, and also mention real-world applications
- Final presentation will talk about what motivated us to do this project, describe process, methods, results, evaluation of results, and show live demo of agents playing 2048 for each method

## Frame 2048 as an MDP
- But then we need a representation of each state
  - Get a set of heuristics that describe the general qualities of the board
  - Run an autoencoder on the heuristics
  - Transforms into a small set of numbers
  - Discretize those and use them as the state
- Alternatively, fourier transform to compress the board?(jpeg compression)
## Next idea
- initialize heuristic to a simple representation like sum or maxTile
- on each max node, after you find the max heuristic value for the children of that node, update weights to make the heuristic value of that node closer to the max val of its children
- same for min nodes
  - Do the same thing, but with machine learning maybe? 
## Next idea
- run trials
- at each state store state (or maybe just heuristic values)
- at end of run, update heuristic weights depending on final score
- alternatively, train neural net to predict final score from heuristic values
## Transfer learning
- Same as previous idea, but uses the score that an already established algorithm gets starting at a given state
  - ie for every state, run a better algorithm starting at that state 10 times and use the avg score as the true score of that state
## Heuristic as Search
- for heuristic, use a search problem
  - find the shortest path that goes to each tile in decreasing value order
  - for example, it starts at 1024, then goes to 512, then 128, then 128, etc
- Alternatively, assume no more computer player
- Then find shortest number of moves needed to upgrade to next maxTile value
## Next idea
- Small (or large) modification to depth/heuristic
  - Change it once you hit 1024 or 2048 or wtvr threshold
- Also have an “emergency mode” when you’re about to lose
  - Can drastically increase search depth if necessary or modify heuristic
## Monte Carlo methods
- As a heuristic, run N random trials from a state, then avg over total results
- Implement some kind of policy or value iteration maybe?
## Heuristic / State representation idea
- Assume no more computer
- Store big pieces and ignore the small ones
  - Run a few turns of player’s turn to combine small things (facilitates “ignoring small pieces”)
  - Or just count how many small pieces there are and store that

## Papers/Articles
- [Using Artifical Intelligent to solve the game of 2048](https://home.cse.ust.hk/~yqsong/teaching/comp3211/projects/2017Fall/G11.pdf)
- [AI plays 2048](http://cs229.stanford.edu/proj2016/report/NieHouAn-AIPlays2048-report.pdf)
- [Artificial Intelligence has crushed all human records in 2048](http://www.randalolson.com/2015/04/27/artificial-intelligence-has-crushed-all-human-records-in-2048-heres-how-the-ai-pulled-it-off/)
- [Deep Refrocement Learning for 2048](https://www.mit.edu/~adedieu/pdf/2048.pdf)
