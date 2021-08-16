import numpy as np
import copy
import os
import math
import random
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from itertools import islice
import seaborn as sns
sns.set(context="talk", style="darkgrid", palette="hls", font="sans-serif", font_scale=1.05)


# Extract the input matrix from the input file:
def extract_input_matrix_from_input_file(file_name='input.txt'):
    '''
    inputs:
        file_name: String
            the name of the file that we want to read.
    ----------------------------------------------------
    this function will read the file and exctract the data of it and return it in a python list!
    -----------------------------------
    outputs:
        mat: list
            return the matrix as a python list!
    '''
    cwd = os.getcwd()  # Get the current working directory (cwd)
    files = os.listdir(cwd)  # Get all the files in that directory
    # print("Files in %r: %s" % (cwd, files))
    __location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(file_name)))
    # print("Location: ", __location__)
    with open(file_name, 'r') as fin:
        mat = []
        for line in islice(fin, 0, None): # to start reading the file from line 2 to the end
            line = line.strip()
            l = [int(num.rstrip().strip()) for num in line.split(' ')]
            mat.append(l)
    return mat


# Function to create the initialization population:
def Generate_Init(chromosome_size=20):
    '''
    inputs:
        chromosome_size: int
    --------------------------------------------------
    This function is generating for us a random initialization.
    --------------------------------------------------
    outputs:
        init: Numpy Array
    '''
    # Set up an initial array of all zeros
    init = np.zeros((chromosome_size), dtype=int)
    # Choose a random number of ones to create
    ones = random.randint(0, chromosome_size)
    # Change the required number of zeros to ones
    init[0:ones] = 1
    # Sfuffle row
    np.random.shuffle(init)
    return init


class SimulatedAnnealing:
    def __init__(self, temperature, alpha, stopping_temperature, stopping_iter, file_name, bit_change_rate):
        ''' animate the solution over time
            Parameters
            ----------
            temperature: float
                initial temperature
            alpha: float
                rate at which temperature decreases
            stopping_temperature: float
                temerature at which annealing process terminates
            stopping_iter: int
                interation at which annealing process terminates
            file_name: String
                name of file that the input data is stored in
            bit_change_rate: float
                the percent of the array that we whant each time to change!
        '''
        
        self.temperature = temperature
        self.alpha = alpha
        self.bit_change_rate = bit_change_rate
        self.stopping_temperature = stopping_temperature
        self.stopping_iter = stopping_iter
        self.iteration = 1

        self.input_mat = extract_input_matrix_from_input_file(file_name)
        self.column_number = len(self.input_mat[0])
        self.sample_size = self.column_number
        self.current_solution = Generate_Init(chromosome_size=self.sample_size)
        self.best_solution = self.current_solution

        self.solution_history = [self.current_solution]
        print("Sample Size: ", self.sample_size)
        print("Current Sol: ", self.current_solution)
        self.current_fitness = self.objective(instance=self.current_solution)
        self.initial_fitness = self.current_fitness
        self.min_fitness = self.current_fitness

        self.objective_list = [self.current_fitness]
        self.temperature_list = [self.temperature]

        print('Intial Fitness: ', self.current_fitness)

    def objective(self, instance):
        '''
        Calcuate Objective
        ------------------
        inputs:
            instance: the sample that we want to calculate it's fitness!
        ------------------
        outputs:
            fitness: returns the calculated fitness!
        '''
        instance_copy = copy.deepcopy(instance)
        score = 0
        # checks the type of the instance
        if not isinstance(instance_copy, (np.ndarray)):
            instance_copy = np.array(instance_copy)
        # checks the type of the input matrix
        if not isinstance(self.input_mat, (np.ndarray)):
            input_mat = np.array(self.input_mat)
        res_mat = input_mat.dot(instance_copy)
        total = len(res_mat)
        score = np.count_nonzero(res_mat==1)
        fitness = (score / total) #*100
        return fitness

    def acceptance_probability(self, candidate_fitness):
        '''
        Function that according to the Simmulated Annealing Algorithm, describes for us the 
        probability that an generated candidate if it's fitness is lower than maximum fitness,
        can be accepted or not!

        '''
        return math.exp(-abs(candidate_fitness - self.current_fitness) / self.temperature)

    def solution_acceptance(self, candidate):
        '''
        Accept with probability 1 if candidate solution is better than
        current solution, else accept with probability equal to the
        acceptance_probability()
        '''
        # print("Sample Size: ", self.sample_size)
        candidate_fitness = self.objective(instance=candidate)
        if candidate_fitness > self.current_fitness:
            self.current_fitness = candidate_fitness
            self.current_solution = candidate
            if candidate_fitness > self.min_fitness:
                self.min_fitness = candidate_fitness
                self.best_solution = candidate

        else:
            if random.random() < self.acceptance_probability(candidate_fitness):
                self.current_fitness = candidate_fitness
                self.current_solution = candidate

    def annealing(self):
        '''
        Annealing process:
        this function is going to ...
        -------------------------
        inputs:
            None
        ---------------------------
        outputs:
            None
        '''
        while self.temperature >= self.stopping_temperature and self.iteration < self.stopping_iter:
            candidate = copy.deepcopy(self.current_solution)
            change_positions_rate = np.random.random(size=(self.sample_size))
            chganging_positions = change_positions_rate <= self.bit_change_rate
            candidate[chganging_positions] = np.logical_not(candidate[chganging_positions])

            self.solution_acceptance(candidate)
            self.temperature *= self.alpha
            self.iteration += 1
            self.objective_list.append(self.current_fitness*100)
            self.solution_history.append(self.current_solution)
            self.temperature_list.append(self.temperature)

        print("==============================================================")
        print("| *************>>> Change Iterations : {0:d} <<<************* |".format(len(self.objective_list)))
        print('| Initial Fitness:===> {0:.2f}% |'.format(self.initial_fitness*100))
        print("|------------------------------------------------------------|")
        print('| Maximum (Optimum) Fitness:===> {0:.2f}% |'.format(self.min_fitness*100))
        print("|------------------------------------------------------------|")
        print("| Final Fitness:===> {0:.2f}% |".format(self.current_fitness*100))
        print("|------------------------------------------------------------|")
        print("| Solution:===> ", self.current_solution)
        print("|------------------------------------------------------------|")
        print("| Total changing Bits:===> {} |".format(self.sample_size))
        print("==============================================================")

    def plot_temperature(self):
        """
        ----------------
        This function will draw the temperature decrease plot for us!
        -----------------
        """
        plt.figure()
        # plt.subplot(121)
        plt.plot(self.temperature_list, 'r')
        # plt.title("States")
        plt.ylabel("Temperature")
        plt.xlabel("Iteration")
        plt.title("Temperature Decrease")
        plt.show()

    def plot_leaning_fitness(self):
        """
        -------------------
        This function will draw for us the fitness plot.
        -------------------
        """
        plt.plot([i for i in range(len(self.objective_list))], self.objective_list, c='dodgerblue')
        line_init = plt.axhline(y=self.initial_fitness*100, color='r', linestyle='--')
        line_min = plt.axhline(y=self.min_fitness*100, color='g', linestyle='--')
        plt.legend([line_init, line_min], ['Initial Fitness', 'Optimized Fitness'])
        plt.ylabel('Fitness (Percentage (%))')
        plt.xlabel('Iteration')
        plt.show()


def main():
    '''set the simulated annealing algorithm params'''
    temperature = 3000
    stopping_temperature = 0.00000000001
    alpha = 0.99995
    stopping_iter = 1000000000000

    '''Set the Rate that we change the Candidate Bites'''
    bit_change_rate = 0.02

    '''set the name of the file:'''
    input_file_name = "q1 input 41 40-2.txt"


    '''run simulated annealing algorithm'''
    # starting time
    start = time.time()

    # sleeping for 1 sec to get 10 sec runtime
    time.sleep(1)

    # program body starts
    sa = SimulatedAnnealing(temperature, alpha, stopping_temperature, stopping_iter, 
                            file_name=input_file_name, bit_change_rate=bit_change_rate)
    sa.annealing()
    # program body ends

    # end time
    end = time.time()

    # total time taken
            
    print("==============================================================")
    print("|------------------------------------------------------------|")
    print("| Runtime of the program is:===> {0:.4f} sec. |".format(end-start))
    print("|------------------------------------------------------------|")
    print("==============================================================")


    '''show the improvement over time'''
    sa.plot_leaning_fitness()
    sa.plot_temperature()


if __name__ == "__main__":
    main()





# instance = np.array([1, 1, 0, 1, 0])
# input_mat = extract_input_matrix_from_input_file('q1.txt')
# input_mat = np.array(input_mat)
# res_mat = input_mat.dot(instance)
# print(res_mat)
# x = np.array([1, 0, 0, 1, 0, 0, 0], dtype='int').reshape(-1)
# print("X: ", x)
# zero_arr = np.zeros(6, dtype='int').reshape(-1)
# one_arr = np.ones(6, dtype='int').reshape(-1)
# print(zero_arr)
# l = random.randint(0, 5)
# print("Random: ", l)
# zero_arr[l] = 1
# print("Zero: ", zero_arr)
# x[zero_arr] = np.logical_not(x[one_arr])
# print("Z: ", x)
# input_mat = extract_input_matrix_from_input_file('q1.txt')
# input_mat = np.array(input_mat)    
# random_mutation_array = np.random.random(size=(5))
# random_mutation_boolean = random_mutation_array <= 0.3
# print("Rand: ", random_mutation_boolean)
# instance = np.array([1, 1, 0, 1, 0])
# instance[random_mutation_boolean] = np.logical_not(instance[random_mutation_boolean])
# print("Inst: ", instance)