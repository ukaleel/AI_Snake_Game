# importing libraries
import pygame
import time
import random
import numpy as np
import os
import matplotlib.pyplot as plt

# Snake game settings
snake_speed = 1000  # Speed of the snake
window_x = 500
window_y = 500
population_size = 200  # Number of neural networks in each generation
initial_mutation_rate = 0.1  # Initial probability of mutation
num_generations = 500  # Number of generations
visualize_count = 1  # Number of neural networks to visualize from each generation
average_fruits_per_generation = []

black = pygame.Color(0, 0, 0)
white = pygame.Color(255, 255, 255)
red = pygame.Color(255, 0, 0)
green = pygame.Color(0, 255, 0)
blue = pygame.Color(0, 0, 255)


pygame.init()

# Initialise game window
pygame.display.set_caption('Snake AI Game with Genetic Algorithm')
game_window = pygame.display.set_mode((window_x, window_y))
fps = pygame.time.Clock()


# Neural Network class
# used for each snake AI generation
# will need modification for inputs and layers as we progress
class NeuralNetwork:
    def __init__(self, input_size=20, hidden_layers=[16, 32, 64, 32], output_size=4):
        # Neural Network Structure
        # Input size represents # of inputs given to the neural network using the get_game_state function
        # Hidden layer values represent the number of neurons in a hidden layer, optimizable
        # Output size represents the 4 directions the neural network can possibly output as a direction for the snake
        # to go to

        self.weights = []
        self.biases = []

        # Initializes the prev layer to input size as only "1" layer so far
        prev_layer_size = input_size

        # Generates the weights for each layer, connecting the neurons from layer to layer
        for hidden_size in hidden_layers:
            self.weights.append(np.random.rand(hidden_size, prev_layer_size) * 2 - 1)
            self.biases.append(np.zeros((hidden_size, 1)))
            prev_layer_size = hidden_size


        self.weights.append(np.random.rand(output_size, prev_layer_size) * 2 - 1)
        self.biases.append(np.zeros((output_size, 1)))

    # Relu and Sigmoid functions used to squash outputs
    def relu(self, x):
        return np.maximum(0, x)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, inputs):
        #Propagates the input information forward to the other hidden layers, and hidden layer to hidden layer,
        # and so on.
        self.inputs = np.array(inputs).reshape(-1, 1)
        self.hidden_outputs = []
        layer_output = self.inputs

        for i in range(len(self.weights) - 1):
            layer_output = self.relu(np.dot(self.weights[i], layer_output) + self.biases[i])
            self.hidden_outputs.append(layer_output)

        self.output = self.sigmoid(np.dot(self.weights[-1], layer_output) + self.biases[-1])
        return self.output

    def predict(self, inputs):
        output = self.forward(inputs)
        return np.argmax(output)

    # Crossbreeds 2 neural networks to create a child neural network
    def crossover(self, other):
        child = NeuralNetwork(
            input_size=len(self.weights[0][0]),
            hidden_layers=[layer.shape[0] for layer in self.weights[:-1]],
            output_size=self.weights[-1].shape[0]
        )

        for i in range(len(self.weights)):
            # Weight generation
            child_weights = np.zeros(self.weights[i].shape)
            random_weights = np.random.rand(*self.weights[i].shape)
            for row in range(self.weights[i].shape[0]):
                for col in range(self.weights[i].shape[1]):
                    if random_weights[row, col] > 0.5:
                        child_weights[row, col] = self.weights[i][row, col]
                    else:
                        child_weights[row, col] = other.weights[i][row, col]
            child.weights[i] = child_weights

            # Bias generation
            child_biases = np.zeros(self.biases[i].shape)
            random_biases = np.random.rand(*self.biases[i].shape)
            for j in range(self.biases[i].shape[0]):
                if random_biases[j] > 0.5:
                    child_biases[j] = self.biases[i][j]
                else:
                    child_biases[j] = other.biases[i][j]
            child.biases[i] = child_biases
        return child

    # Mutates the neural network biases/weights under a certain condition
    def mutate(self, mutation_rate):
        for i in range(len(self.weights)):
            random_weights = np.random.rand(*self.weights[i].shape)
            mutation_weight_flag = np.zeros(self.weights[i].shape)
            for row in range(self.weights[i].shape[0]):
                for col in range(self.weights[i].shape[1]):
                    if random_weights[row, col] < mutation_rate:
                        mutation_weight_flag[row, col] = 1
                    else:
                        mutation_weight_flag[row, col] = 0

            random_biases = np.random.rand(*self.biases[i].shape)
            mutation_bias_flag = np.zeros(self.biases[i].shape)
            for j in range(self.biases[i].shape[0]):
                if random_biases[j] < mutation_rate:
                    mutation_bias_flag[j] = 1
                else:
                    mutation_bias_flag[j] = 0

            weight_mutations = np.random.randn(*self.weights[i].shape)
            mutated_weights = mutation_weight_flag * weight_mutations

            bias_mutations = np.random.rand(*self.biases[i].shape)
            mutated_biases = mutation_bias_flag * bias_mutations

            self.weights[i] += mutated_weights
            self.biases[i] += mutated_biases

# Gives the relevant game state information to the neural network
# Which then allows it to make predictions
# Modifying as needed

def get_game_state(snake_position, fruit_position, snake_body, direction):
    #Relative position of the fruit to the snake
    relative_fruit_x = (fruit_position[0] - snake_position[0]) / window_x
    relative_fruit_y = (fruit_position[1] - snake_position[1]) / window_y

    # Determine relative direction of the fruit
    if relative_fruit_y < 0:
        fruit_direction_up = 1
    else:
        fruit_direction_up = 0
    if relative_fruit_y > 0:
        fruit_direction_down = 1
    else:
        fruit_direction_down = 0
    if relative_fruit_x < 0:
        fruit_direction_left = 1
    else:
        fruit_direction_left = 0
    if relative_fruit_x > 0:
        fruit_direction_right = 1
    else:
        fruit_direction_right = 0


    # What direction the snake is facing
    if direction == 'UP':
        direction_up = 1
    else:
        direction_up = 0

    if direction == 'DOWN':
        direction_down = 1
    else:
        direction_down = 0

    if direction == 'LEFT':
        direction_left = 1
    else:
        direction_left = 0

    if direction == 'RIGHT':
        direction_right = 1
    else:
        direction_right = 0

    # Danger detection - if there is a wall or the snakes body present in any direction
    if (snake_position[0] - 10, snake_position[1]) in snake_body or snake_position[0] - 10 < 0:
        danger_left = 1
    else:
        danger_left = 0

    if (snake_position[0] + 10, snake_position[1]) in snake_body or snake_position[0] + 10 >= window_x:
        danger_right = 1
    else:
        danger_right = 0

    if (snake_position[0], snake_position[1] - 10) in snake_body or snake_position[1] - 10 < 0:
        danger_up = 1
    else:
        danger_up = 0

    if (snake_position[0], snake_position[1] + 10) in snake_body or snake_position[1] + 10 >= window_y:
        danger_down = 1
    else:
        danger_down = 0

    # Tail position relative to the snake's head
    if len(snake_body) > 1:
        tail_position = snake_body[-1]
    else:
        tail_position = snake_position
    relative_tail_x = (tail_position[0] - snake_position[0]) / window_x
    relative_tail_y = (tail_position[1] - snake_position[1]) / window_y

    # Tail direction
    if len(snake_body) > 1:
        tail_direction = (snake_body[-2][0] - tail_position[0], snake_body[-2][1] - tail_position[1])
        if tail_direction == (0, -10):
            tail_up = 1
        else:
            tail_up = 0
        if tail_direction == (0, 10):
            tail_down = 1
        else:
            tail_down = 0
        if tail_direction == (-10, 0):
            tail_left = 1
        else:
            tail_left = 0
        if tail_direction == (-10, 0):
            tail_right = 1
        else:
            tail_right = 0
    else:
        tail_up = tail_down = tail_left = tail_right = 0

    return [
        relative_fruit_x, relative_fruit_y,
        danger_left, danger_right, danger_up, danger_down,
        direction_up, direction_down, direction_left, direction_right,
        fruit_direction_up, fruit_direction_down, fruit_direction_left, fruit_direction_right,
        relative_tail_x, relative_tail_y,
        tail_up, tail_down, tail_left, tail_right
    ]






def initialize_population(populationSize):
    listOfNeuralNetworks = []
    for i in range(population_size):
        newChild = NeuralNetwork()
        listOfNeuralNetworks.append(newChild)
    return listOfNeuralNetworks


# Fitness function to determine worth of a neural network
# Utilizes steps and distance from fruit to update score

def evaluate_fitness(nn, index, max_steps=10000, no_progress_steps=1000):
    snake_position = [100, 50]
    snake_body = [[100, 50], [90, 50], [80, 50], [70, 50]]
    fruit_position = [random.randrange(1, (window_x // 10)) * 10, random.randrange(1, (window_y // 10)) * 10]
    fruit_spawn = True
    direction = 'RIGHT'
    steps = 0
    score = 0
    fruits_eaten = 0
    last_distance = np.inf
    progress = False
    direction_change_count = 0


    move_history = []

    # Game loop, if it hits max steps, end the run
    while steps < max_steps:
        loop_count = 0
        # Gets the game state from the get function
        game_state = get_game_state(snake_position, fruit_position, snake_body, direction)
        # Gives the move that the neural network has decided to do based off game state and weights/biases
        prediction = nn.predict(game_state)
        direction_change = ['UP', 'DOWN', 'LEFT', 'RIGHT'][prediction]


        # a counter variable, not necessary
        if direction_change != direction:
            direction_change_count += 1

        # Changes the direction if needed based off prediction
        if direction_change == 'UP' and direction != 'DOWN':
            direction = 'UP'
        if direction_change == 'DOWN' and direction != 'UP':
            direction = 'DOWN'
        if direction_change == 'LEFT' and direction != 'RIGHT':
            direction = 'LEFT'
        if direction_change == 'RIGHT' and direction != 'LEFT':
            direction = 'RIGHT'

        #Changes position of snake based off direction
        if direction == 'UP':
            snake_position[1] -= 10
        if direction == 'DOWN':
            snake_position[1] += 10
        if direction == 'LEFT':
            snake_position[0] -= 10
        if direction == 'RIGHT':
            snake_position[0] += 10

        snake_body.insert(0, list(snake_position))

        # + 10 to the snakes score if it eats a fruit
        if snake_position[0] == fruit_position[0] and snake_position[1] == fruit_position[1]:
            score += 100  # High reward for eating fruit
            fruits_eaten += 1
            fruit_spawn = False
            progress = True
            max_steps+=2000
        else:
            snake_body.pop()

        # Spawns a fruit if its eaten
        if not fruit_spawn:
            fruit_position = [random.randrange(1, (window_x // 10)) * 10, random.randrange(1, (window_y // 10)) * 10]
        fruit_spawn = True


        # Ends the run if the snake hits its body or wall, lowers the score
        if (snake_position[0] < 0 or snake_position[0] >= window_x or
                snake_position[1] < 0 or snake_position[1] >= window_y):
            score -= 200

            break

        for block in snake_body[1:]:
            if snake_position[0] == block[0] and snake_position[1] == block[1]:
                score -= 200
                break

        # If the snake is getting closer to the food, it increases the score
        # Motivates over time for the snake to get closer to the food
        # Might be an unnecessary function

        distance_to_food = np.sqrt((snake_position[0] - fruit_position[0]) ** 2 +
                                   (snake_position[1] - fruit_position[1]) ** 2)
        if distance_to_food < last_distance:
            score += 1  # Increased reward for getting closer to food
            progress = True
        else:
            score -= 1  # Penalty for moving away from food

        last_distance = distance_to_food
        steps += 1

        # Functionality to punish snakes that just go in a circle without doing anything
        move_history.append(tuple(snake_position))
        if len(move_history) > no_progress_steps:
            move_history.pop(0)

        if steps > 50:
            pattern = move_history[:5]
            first_tuple = pattern[0]
            history = move_history[5:]
            for i, tuples in enumerate(history):
                if tuples == first_tuple:
                    look = history[i:i + 5]
                    if look == pattern:
                        loop_count += 1
            if loop_count > 5:
                print("loop", nn)
                score -= 1000
                break

      #  if direction_change_count < 3 and steps % 50 == 0:  # Penalize for lack of direction change
          #  score -= 200
    if index%20 == 0:
        with open("edittingtextfile.txt", "a") as file:
            file.write("\nNeural Network Index: "+ str(index) + ", Fruits Eaten: " + str(fruits_eaten) + "\n")



    return [score + steps/max_steps, fruits_eaten]


# Genetic Algorithm, will heavily modify

def genetic_algorithm(population, num_generations, initial_mutation_rate):
    mutation_rate = initial_mutation_rate
    for generation in range(num_generations):
        print("Generation ",generation+1)
        with open("edittingtextfile.txt", "a") as file:
            file.write("Generation " + str(generation + 1) + "\n")

        # Evaluates each neural network by running it, and then sorting it based off its' score
        fitness_scores = []
        fruits_eaten = []
        for i in range(len(population)):
            fitness = evaluate_fitness(population[i], i)
            fitness_scores.append(fitness[0])
            fruits_eaten.append(fitness[1])

        avg_fruits_eaten = np.mean(fruits_eaten)
        average_fruits_per_generation.append(avg_fruits_eaten)
        update_bar_graph(average_fruits_per_generation, generation+1)

        max_fitness = max(fitness_scores)
        min_fitness = min(fitness_scores)


        #scaled_fitness = [(score - min_fitness) / (max_fitness - min_fitness + 0.00000001) for score in fitness_scores]

        scaled_fitness = []
        for score in fitness_scores:
           scaled_fitness.append((score - min_fitness) / (max_fitness - min_fitness + 0.00000001))


        # Sort the population by scaled fitness scores
        fitness_cross_population = zip(scaled_fitness, population)
        sorted_scores = sorted(fitness_cross_population, key = lambda item: item[0], reverse=True)

        sorted_population = []
        for score in sorted_scores:
            single_score = score[1]
            sorted_population.append(single_score)


        #sorted_population = [x for _, x in sorted(zip(scaled_fitness, population), key=lambda item: item[0],
             #                                     reverse=True)]

        new_population = sorted_population[:5]  # Keep the top 5 neural networks intact

        # Generates the new population using the genetic algorithm
        top_count = 5
        second_count = 10
        while len(new_population) < population_size:
            for i in range(top_count, min(population_size, top_count + 5)):
                parent1 = sorted_population[random.randint(0, top_count - 1)]
               #parent2 = sorted_population[random.randint(0, i - 1)]#
                parent2 = sorted_population[random.randint(0, second_count - 1)]
                child = parent1.crossover(parent2)
                child.mutate(mutation_rate)
                new_population.append(child)
                if len(new_population) >= population_size:
                    break
            #top_count += 5

        population = new_population

        # Adjust mutation rate dynamically based on performance
        if max(scaled_fitness) < 0.5:
            mutation_rate = min(1.0, mutation_rate + 0.05)  # Increase mutation rate if performance is low
        else:
            mutation_rate = max(0.01, mutation_rate - 0.01)  # Decrease mutation rate if performance is good

        # Visualization of the top neural networks in the current generation
        for index, nn in enumerate(sorted_population[:visualize_count]):
            visualize_snake(nn, generation + 1, index + 1)
            #pygame.display.quit()

    return population



# Visualization function
def visualize_snake(nn, generation, nn_index, max_steps=5000):
    pygame.init()
    game_window = pygame.display.set_mode((window_x, window_y))
    pygame.display.set_caption(f'Snake AI Generation {generation}, NN {nn_index}')

    snake_position = [100, 50]
    snake_body = [[100, 50], [90, 50], [80, 50], [70, 50]]
    fruit_position = [random.randrange(1, (window_x // 10)) * 10, random.randrange(1, (window_y // 10)) * 10]
    fruit_spawn = True
    direction = 'RIGHT'
    steps = 0
    fruits_eaten = 0

    while steps<max_steps:
        game_window.fill(black)


        font = pygame.font.SysFont('arial', 24)
        generation_text = font.render(f'Generation: {generation}, NN: {nn_index} , Fruits Eaten: {fruits_eaten}' ,
                                      True, white)
        game_window.blit(generation_text, [10, 10])

        game_state = get_game_state(snake_position, fruit_position, snake_body, direction)
        prediction = nn.predict(game_state)


        if prediction == 0:
            direction_change = 'UP'
        elif prediction == 1:
            direction_change = 'DOWN'
        elif prediction == 2:
            direction_change = 'LEFT'
        else:
            direction_change = 'RIGHT'

        # Prevent snake from reversing direction
        if direction_change == 'UP' and direction != 'DOWN':
            direction = 'UP'
        if direction_change == 'DOWN' and direction != 'UP':
            direction = 'DOWN'
        if direction_change == 'LEFT' and direction != 'RIGHT':
            direction = 'LEFT'
        if direction_change == 'RIGHT' and direction != 'LEFT':
            direction = 'RIGHT'

        # Move the snake in the chosen direction
        if direction == 'UP':
            snake_position[1] -= 10
        if direction == 'DOWN':
            snake_position[1] += 10
        if direction == 'LEFT':
            snake_position[0] -= 10
        if direction == 'RIGHT':
            snake_position[0] += 10

        snake_body.insert(0, list(snake_position))

        # Check if the snake has eaten the fruit
        if snake_position[0] == fruit_position[0] and snake_position[1] == fruit_position[1]:
            fruit_spawn = False
            fruits_eaten+=1
            max_steps+=1000
        else:
            snake_body.pop()

        # Spawn a new fruit if one has been eaten
        if not fruit_spawn:
            fruit_position = [random.randrange(1, (window_x // 10)) * 10, random.randrange(1, (window_y // 10)) * 10]
        fruit_spawn = True

        # Draw the snake and fruit
        for pos in snake_body:
            pygame.draw.rect(game_window, green, pygame.Rect(pos[0], pos[1], 10, 10))
        pygame.draw.rect(game_window, red, pygame.Rect(fruit_position[0], fruit_position[1], 10, 10))

        # Check for collisions
        if (snake_position[0] < 0 or snake_position[0] >= window_x or
                snake_position[1] < 0 or snake_position[1] >= window_y or
                snake_body[1:].count(snake_position) > 0):
            if (generation+1) % 10==0 and nn_index == 1: # every 10 generations and the best ranked neural network
            # will output to the text file
                with open("experiment2.txt", "a") as file:
                    file.write("Generation "+ str(generation+1)+ ", Fruits Eaten: "+str(fruits_eaten) + "\n")

            break

        pygame.display.update()
        fps.tick(snake_speed)

        steps += 1  # Increment step counter


def update_bar_graph(average_fruits_per_generation, current_generation):
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, current_generation + 1), average_fruits_per_generation, color='skyblue')
    plt.xlabel('Generation')
    plt.ylabel('Average Fruits Eaten')
    plt.title('Average Fruits Eaten Per Generation')
    plt.tight_layout()

    # Save the graph to a file
    plt.savefig('average_apples.png')
    plt.close()


def main():
    population = initialize_population(population_size)
    genetic_algorithm(population, num_generations, initial_mutation_rate)
    pygame.quit()


if __name__ == "__main__":
    main()
