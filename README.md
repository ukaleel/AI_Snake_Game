# AI_Snake_Game
An Approach to Optimizing Snake AI Using Genetic Algorithm and Neural Networks

Original Repository That I and Team Members Worked on for a Project:
https://github.com/GorillaDondon/AI_Snake_Game

The goal of the program is to create a snake AI that could beat the game of snake. Through the crossbreeding of many generations, the algorithm is supposed to get better at reaching the fruit without hitting itself and walls. 




> **Requirements**

Install Pygame and Numpy before running

> **Game Settings**

Snake speed represents how fast the visualization has the snake move.

Visualization count tells the visualizer how many of the generation you want to visualize.

> **Neural Network Structure**

14 inputs representing the game state variables, subject to change.

Hidden layers are configurable and represent the number of layers and neurons connecting each layer.

Output layer has 4 possible outputs, corresponding to the 4 possible movements. Subject to change, possibly 3 since a snake can only move 3 directtions.

> **Game Flow**

The program starts Pygame and opens up a game window. The initial population of neural networks is generated. As each neural network runs and a generation is completed, records of their results are saved in a text file as well as visualized in a graph. 

After all the weights and biases are generated for the neural networks of the generation, each is ran individually.

As they run, each move is dictated by the neural network. It passes the game state to the neural network every move, and the neural network then decides or "predicts" what move to make based off the currentt game state. 

Once the snake moves, the game state is updated and then passed again to the neural network until game over. 

The score of snake's neural network is determined as it runs, with its' score going up as it hits fruits but also if it makes moves that take it closer to the fruit. In the same vein, if it makes moves that are detrimental to it, like moving away from the fruit or hitting itself, it reduces its score. 

After all the snakes of a generation run, they are sorted by their score and then crossbred creating a new generation. This process repeats until it hits the number of generations desired by the program. 
