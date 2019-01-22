import pandas as pd
'''This class is created to contain all the utility methods '''
import numpy as np
import random
import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn import preprocessing

class Util:

    ''' Step 3: Filtering the training dataset . 5 columns as input, last column input'''
    def getTrainingDataset(self, data_set):
        return data_set.iloc[0:int(len(data_set.index) * 0.75), [0,1,2,3,4,13] ]

    # Filtering the testing dataset . 5 columns as input, last column input
    def getTestingDataset(self,data_set):
        return data_set.iloc[-int(len(data_set.index) * 0.25):, [0,1,2,3,4,13] ]

    #Function for normalizing
    def getNormalizedDataSet(self, tr_data_set):
        result = tr_data_set.copy()
        for feature_name in tr_data_set.columns:
            max_value = tr_data_set[feature_name].max()
            min_value = tr_data_set[feature_name].min()
            result[feature_name] = (tr_data_set[feature_name] - min_value) / (max_value - min_value)
        return result

    '''def getNormalizedTrDataSet(tr_data_set):
        return (tr_data_set - tr_data_set.min()) / (tr_data_set.max() - tr_data_set.min())'''

    '''def getNormalizedTrDataSet(self,tr_data_set,scaler):
        return pd.DataFrame(scaler.fit_transform(tr_data_set))'''

    # Create Crossover between main parent and other parents
    def create_crossover(self,parent_chromosome,other_chromosome_lst):
        child_chromosome_lst=[]
        for other_parent_chromosome in other_chromosome_lst:
            self.create_new_child_chromosomes(child_chromosome_lst,parent_chromosome,other_parent_chromosome)

        #Add parent also in the same list
        child_chromosome_lst.append(parent_chromosome)
        child_chromosome_lst.append(parent_chromosome)
        return child_chromosome_lst

    # Generates a random value and uses that value to get particular section from each of its parents
    def create_new_child_chromosomes(self,child_chromosome_lst,parent_chromosome,other_parent_chromosome):
        for i in range(2):
            c_point = np.random.random_integers(2, len(parent_chromosome) - 1)
            child_chromosome_lst.append(parent_chromosome[:c_point]+other_parent_chromosome[c_point:])

        return child_chromosome_lst

    # Mutate child population. Here,since the mutationrate is 5%, 25 bits will be changed.
    # So, I have fetched 25 random indexes and then flipped the values corresponding to that index
    def mutate_child_population(self,new_child_population,mutation_rate):
        no_of_mutated_bits = mutation_rate*500
        mutated_child_pop = []
        for child in new_child_population:
            for x in range(int(no_of_mutated_bits)):
                ran=random.randint(0, 499)
                if child[ran] == '0':
                    child=child[:ran]+'1'+child[ran+1:]
                if child[ran] == '1':
                    child = child[:ran] + '0' + child[ran + 1:]

            mutated_child_pop.append(child)
        return mutated_child_pop

    def binarize_parent_chromosomes(self,weight_mat_array,scaler,max_index,child_chromosome_lst):
        parent_chromosome=''
        count = 0
        for array in weight_mat_array:

            # print(np.max(array))
            # print(np.min(array))
            # array[:]=array-np.min(array)/(np.max(array)-np.min(array))
            # Verify this once

            # x_normed = (array - array.min(0)) / array.ptp(0)
            x_normed = scaler.fit_transform(array)
            x_normed[:] = x_normed * 1000
            chromosome = ''
            #print(x_normed)
            #print(count)
            for x in x_normed:
                #print('x_normed')
                #print(x)
                for i in x:
                    binary_x = bin(int(i))[2:].zfill(10)
                    # print(str(binary_x))
                    if count == max_index:
                        parent_chromosome = parent_chromosome + str(binary_x)
                    else:
                        chromosome = chromosome + str(binary_x)

            count = count + 1

            # print("Child:"+chromosome)
            if len(chromosome) != 0:
                child_chromosome_lst.append(chromosome)
        return parent_chromosome

    # Binarize the other chromosomes and create a long chromosome each of length 500 bits
    def binarize_other_chromosomes(self, weight_mat_array, scaler, max_index):
        parent_chromosome = ''
        count = 0
        child_chromosome_lst=[]
        for array in weight_mat_array:
            # Normalize each matrix
            x_normed = scaler.fit_transform(array)
            # Multiply by 1000
            x_normed[:] = x_normed * 1000
            # Convert that to binary and concatenate to form long 500 bit chromosome.
            chromosome = ''
            for x in x_normed:
                for i in x:
                    binary_x = bin(int(i))[2:].zfill(10)
                    #Eliminate parent from the resultant matrix
                    if count == max_index:
                        parent_chromosome = parent_chromosome + str(binary_x)
                    else:
                        chromosome = chromosome + str(binary_x)
            count = count + 1
            if len(chromosome) != 0:
                child_chromosome_lst.append(chromosome)
        return child_chromosome_lst

    # Debinarise the mutated child matrix
    def debinarize_child_pop(self, mutated_child_pop, scaler):

        child_weight_mat_array=[]
        for chromosome in mutated_child_pop:
            #Desegment Chromosome
            de_segmented_cr=self.desegment_chromosome(chromosome)

            decimal_lst=[int(cr, 2) for cr in de_segmented_cr]
            decimal_mat=np.array(decimal_lst).reshape(10,5)
            decimal_mat=decimal_mat/1000
            debinarized_mat= scaler.inverse_transform(decimal_mat)

            child_weight_mat_array.append(debinarized_mat)
        return np.array(child_weight_mat_array)


    # Desegment Chromosome in size 10 bits
    def desegment_chromosome(self,chromosome):
        lst = []
        chromosome_count = len(chromosome)
        while chromosome_count != 0:
            lst.append(chromosome[:10])
            chromosome = chromosome[10:]
            chromosome_count = chromosome_count - 10
        return lst

    # Compute Fitness Value using the same logic as above
    def computeFitnessValue(self,child_weight_mat_array,nm_tr_data_set,MSE_list):
        y_hat_lst = []
        fitness_weight_dict={}
        for matrix in child_weight_mat_array:
            y_hat_lst.clear()
            weight_mat_tr = np.transpose(matrix)
            data_size = len(nm_tr_data_set.index)
            for n in range(data_size):
                ip_detail = nm_tr_data_set.iloc[n, :5]
                ip_mat = ip_detail.values
                x_mat = np.matmul(ip_mat, weight_mat_tr)
                y_hat = 0
                for x in x_mat:
                    y_hat = y_hat + (1 / (1 + math.exp(-x)))
                y_hat_lst.append(y_hat)

            diff = np.array(y_hat_lst) - nm_tr_data_set.iloc[:, 5].values
            summation = sum(diff ** 2)
            MSE_list.append(summation/data_size)
            fitness = (1 - summation / data_size) * 100
            fitness_weight_dict[fitness]=matrix

        return fitness_weight_dict

    # Sort all chromosomes by fitness value
    # Eliminate Half of them and store them in a dictionary with key as fitness value and value is the corresponding Wpn Matrix.
    def eliminate_least_fit_chromosomes(self,fitness_dict,Npop):
        length=len(fitness_dict)
        asc_sorted = sorted(fitness_dict)
        list=asc_sorted[-(Npop):]
        new_dict={}
        for x in list:
            new_dict[x]=fitness_dict[x]
        return new_dict

    # Get new parent from the above dictionary
    # If current calculated fitness value is less than the previous iteration fitness value, then the
    # previous parent is returned
    def get_new_parent(self,fit_chrms_dict,max_fitness,max_fitness_Wpn):
        current_iteration_max_fitness=sorted(fit_chrms_dict.keys())[-1]
        if(current_iteration_max_fitness<max_fitness):
            return max_fitness_Wpn
        else:
            return fit_chrms_dict.get(current_iteration_max_fitness)

    def get_new_max_fitness_value(self,fit_chrms_dict,last_gen_max_fitness):
        current_iteration_max_fitness = sorted(fit_chrms_dict.keys())[-1]
        if (current_iteration_max_fitness < last_gen_max_fitness):
            return last_gen_max_fitness
        else:
            return current_iteration_max_fitness

    # Binarize the parent chromosome and create a long chromosome of 500 bits
    def binarize_parent_chromosome(self,parent_mat,scaler):
        # Normalize Parent Matrix
        x_normed = scaler.fit_transform(parent_mat)
        # Multiply by 1000
        x_normed[:] = x_normed * 1000
        # Convert into Binary and Concatenate Strings in loop
        chromosome = ''
        for x in x_normed:
            for i in x:
                binary_x = bin(int(i))[2:].zfill(10)
                chromosome = chromosome + str(binary_x)
        return chromosome

    def get_other_parents(self,fit_chrms_dict):
        weight_list=[]
        for element in fit_chrms_dict:
            weight_list.append(fit_chrms_dict[element])
        return np.array(weight_list)

    def get_other_parents_fitness(self,fit_chrms_dict):
        weight_list=[]
        for element in fit_chrms_dict:
            #print(element)
            #print(fit_chrms_dict[element])
            weight_list.append(element)
        return np.array(weight_list)

    def get_max_index_for_computation(self,fit_chrms_dict):
        return len(fit_chrms_dict)-1

    def addExistingParents(self,fitness_dict,weight_mat_array,fitness_array):
        for fitness in fitness_array:
            for weight in weight_mat_array:
                fitness_dict[fitness]=weight
        return fitness_dict

    def scatterplot_fitness_range(self,max_fitness_lst):
        x=range(len(max_fitness_lst))
        plt.scatter(x,max_fitness_lst, label='Fitness Value')
        plt.xlabel('Generation')
        plt.ylabel('Fitness Value')
        plt.title('The highest fitness value for each iteration')
        plt.legend()
        plt.show()

    def compute_error_for_testing_dataset(self,parent_mat,tst_data_set):
        y_hat_lst = []
        weight_mat_tr = np.transpose(parent_mat)
        data_size = len(tst_data_set.index)
        for n in range(data_size):
            ip_detail = tst_data_set.iloc[n, :5]
            ip_mat = ip_detail.values
            x_mat = np.matmul(ip_mat, weight_mat_tr)
            y_hat = 0

            for x in x_mat:
                y_hat = y_hat + (1 / (1 + math.exp(-x)))

            y_hat_lst.append(y_hat)
        diff = np.array(y_hat_lst) - tst_data_set.iloc[:, 5].values
        summation = sum(diff ** 2)
        return summation/data_size

    def scatterplot_3d_y_yhat(self,parent_mat,nm_tst_data_set):
        y_hat_lst = []
        parent_mat=self.split_parent_mat(parent_mat)
        weight_mat_tr=np.transpose(parent_mat)
        data_size = len(nm_tst_data_set.index)
        for n in range(data_size):
            ip_detail = nm_tst_data_set.iloc[n, :2]
            ip_mat = ip_detail.values
            x_mat = np.matmul(ip_mat, weight_mat_tr)
            y_hat = 0

            for x in x_mat:
                y_hat = y_hat + (1 / (1 + math.exp(-x)))

            y_hat_lst.append(y_hat)
        self.plot_3d(y_hat_lst,nm_tst_data_set)





    def split_parent_mat(self,parent_mat):
        return np.delete(parent_mat, np.s_[2:5], axis=1)

    def plot_3d(self,y_hat_lst,nm_tst_data_set):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        z1 = y_hat_lst
        z2 = nm_tst_data_set.iloc[:, 5].values
        x = nm_tst_data_set.iloc[:, 0].values
        y = nm_tst_data_set.iloc[:, 1].values
        ax.scatter(x, y, z1, c='r', marker='o', label='y_hat')
        ax.scatter(x, y, z2, c='b', marker='^', label='y')
        ax.set_title('3D scatterplot of y and yhat')
        ax.set_xlabel('Weight (Normalised)')
        ax.set_ylabel('Height (Normalised)')
        ax.set_zlabel('Y hat and Y(normalised)')
        ax.legend()
        plt.show()