import pandas as pd
import numpy as np
import math
from GA_Body_Fat_Estimation_Util import Util
from sklearn.preprocessing import MinMaxScaler

#You need 2 python files to run this project. GA_Body_Fat_Estimation and GA_Body_Fat_Estimation_Util.
#Set Npop ,N and mutation rate
Npop = 500
N = 10
mutation_rate = 0.05

loopcount = 0
i = 10
ut: Util = Util()
scaler = MinMaxScaler()
y_hat_lst = []
weight_mat_lst = []
fitness_lst = []
max_fitness_lst = []
MSE_list=[]
max_fitness = -999999999999999


#Step 1: Import the dataset from the .csv file provided .
'''The file is present in the working directory hence no need to give the full path'''
data_set = pd.read_csv('BodyFatCsvData.csv')
print("Imported Dataset")
tr_data_set = ut.getTrainingDataset(data_set)
tst_data_set = ut.getTestingDataset(data_set)
#Normalise the training and testing dataset
nm_tr_data_set = ut.getNormalizedDataSet(tr_data_set)
nm_tst_data_set = ut.getNormalizedDataSet(tst_data_set)

#Since we have considered N=10,P(input Columns)=10, We have a Wpn Matrix of dim(10,5) generated randomly of range(-1 to 1) for Npop number of times
for N in range(1, Npop + 1):
    y_hat_lst.clear()
    #I have observed that if the randommatrix is n the range greater than(-3 to 3). Eg(-5 to 5) or (-10 to 10), I get a better fitness value. as compared to -1 to 1
    weight_mat = np.random.uniform(low=-1.0, high=1.0, size=(10, 5))
    weight_mat_lst.append(weight_mat)
    weight_mat_tr = np.transpose(weight_mat)
    #Calculate yhat for all rows and store that in the list (Equation 1)
    for n in range(len(nm_tr_data_set.index)):
        ip_detail = nm_tr_data_set.iloc[n, :5]
        ip_mat = ip_detail.values
        x_mat = np.matmul(ip_mat, weight_mat_tr)
        y_hat = 0
        for x in x_mat:
            y_hat = y_hat + (1 / (1 + math.exp(-x)))
        y_hat_lst.append(y_hat)

    # Compute Fitness values of all the Npop times and store it in a fitness_lst.
    diff = np.array(y_hat_lst) - nm_tr_data_set.iloc[:, 5].values
    summation = sum(diff ** 2)
    MSE_list.append(summation/int(len(tr_data_set)))
    fitness = (1 - summation / (int(len(tr_data_set)))) * 100
    fitness_lst.append(fitness)

print("The fitness values for Npop iterations is:", fitness_lst)
# Find the maximum fitness value and that will be our main parent(sire)
Wpn_mat_array = np.array(weight_mat_lst)
fitness_array = np.array(fitness_lst)
last_gen_max_fitness = np.max(fitness_array)
print('The best fitness value for this iteration:',last_gen_max_fitness)
# Finding the corresponding Wpn_mat of max fitness value. I have named that as parent_mat
index = np.where(fitness_array == np.max(fitness_array))
max_index = 0
for t in index:
    max_index = int(t)
sire = Wpn_mat_array[index]
parent_mat = sire[0]

other_parent_chromosome = []
max_fitness_lst.append(last_gen_max_fitness)
generation_count=1
# This is the main step where we perform iteration to get the max fitness value in our algorithm
while True:
    #Binarize the parent matrix
    parent_chromosome = ut.binarize_parent_chromosome(parent_mat, scaler)
    #Binarize all other parents
    other_chromosome = ut.binarize_other_chromosomes(Wpn_mat_array, scaler, max_index)
    # Create Crossover between main parent and other parent's chromosomes
    new_child_population = ut.create_crossover(parent_chromosome, other_chromosome)
    # Mutate all the child chromosomes
    mutated_child_pop = ut.mutate_child_population(new_child_population, mutation_rate)
    # Debinarise the mutated child matrix
    child_Wpn_mat = ut.debinarize_child_pop(mutated_child_pop, scaler)
    #Computing Fitness Value
    fitness_dict = ut.computeFitnessValue(child_Wpn_mat, nm_tr_data_set,MSE_list)
    # CrossCheckandRemove
    # fitness_dict=ut.addExistingParents(fitness_dict,weight_mat_array,fitness_array)

    # Eliminate Least Fit chromosomes
    fit_chrms_dict = ut.eliminate_least_fit_chromosomes(fitness_dict, Npop)
    # Get new parent from the above dictionary and replace the existing parent Wpn value (parent_mat) for iterating in
    # the loop
    parent_mat = ut.get_new_parent(fit_chrms_dict,last_gen_max_fitness,parent_mat)

    # Get new Fitness Value from the above dictionary. The process is similar to the above computation.
    current_iteration_max_fitness = ut.get_new_max_fitness_value(fit_chrms_dict,last_gen_max_fitness)
    print("Max fitness value for generation "+str(generation_count)+":"+str(current_iteration_max_fitness))
    # The current iteration max fitness value is stored in a list for scatterplotting
    max_fitness_lst.append(current_iteration_max_fitness)
    # This is calculated so as to pass that again in the while loop, so that the new values are reflected in the input.
    Wpn_mat_array = ut.get_other_parents(fit_chrms_dict)
    # This is calculated so as to pass that again in the while loop, so that the new values are reflected in the input.
    fitness_array = ut.get_other_parents_fitness(fit_chrms_dict)
    # This is calculated so as to pass that again in the while loop, so that the new values are reflected in the input.
    max_index = ut.get_max_index_for_computation(fit_chrms_dict)
    #This if-condition checks if the fitness value reaches the plateau, (if the value is same for 10 iterations),then
    #it breaks out of the loop.
    if round(current_iteration_max_fitness, ndigits=4) == round(last_gen_max_fitness, ndigits=4):
        loopcount = loopcount + 1
        if loopcount == 10:
            print('Plateau reached')
            break
    generation_count=generation_count+1
    last_gen_max_fitness = current_iteration_max_fitness

#MSE of the training dataset for each iteration can be viewed by uncommenting
#print(MSE_list)

#Will Scatterplot the fitness range with generation in the x axis and fitness value in the y axis
ut.scatterplot_fitness_range(max_fitness_lst)
y_hat_lst.clear()
#Compute the overall error. Closer to 0 is better
error_value_tst_dataset = ut.compute_error_for_testing_dataset(parent_mat,nm_tst_data_set)
print("The overall error for testing dataset:",error_value_tst_dataset)

# 3D scatterplot y and yhat for 2 input columns in testing dataset
ut.scatterplot_3d_y_yhat(parent_mat,nm_tst_data_set)