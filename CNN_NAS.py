import os
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import datasets
from sklearn.datasets import load_digits
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#########################
#                       #
#     Dataset Class     #
#                       #
#########################

class DigitsDataset(Dataset):
    '''
    Digit database from scikit-learn
    '''

    def __init__(self, mode = "train", transforms = None):
        digits = load_digits()

        #select the first 1000 datapoints as training set
        if mode == "train": 
            self.data = digits.data[:1000].astype(np.float32)
            self.targets = digits.target[:1000]

        #select 350 datapoints as training set
        elif mode == "val": 
            self.data = digits.data[1000:1350].astype(np.float32)
            self.targets = digits.target[1000:1350]
        
        #select the remaining datapoints as test set
        else: 
            self.data = digits.data[1350:].astype(np.float32)
            self.targets = digits.target[1350:]

        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    #get data from the dataset
    def __getitem__(self, idx):
        sample_x = self.data[idx]       
        #reshape datapoints from vector to matrix
        sample_x = sample_x.reshape(1, 8, 8) 
        sample_y = self.targets[idx]
        
        #move on gpu if possible
        if torch.cuda.is_available(): 
          sample_x = torch.from_numpy(sample_x).to("cuda")
          sample_y = torch.from_numpy(np.array(sample_y)).to("cuda")

        return (sample_x, sample_y)
    

#########################
#                       #
#       CNN Class       #
#                       #
#########################

class CNN(nn.Module):
    '''
    following the structure of:
      Conv2d → f(.) → Pooling → Flatten → Linear 1 → f(.) → Linear 2 → Softmax

    ● Conv2d:
        ○ Number of filters: 8, 16, 32
        ○ kernel=(3,3), stride=1, padding=1 OR kernel=(5,5), stride=1, padding=2
    ● f(.):
        ○ ReLU OR sigmoid OR tanh OR softplus OR ELU
    ● Pooling:
        ○ 2x2 OR Identity
        ○ Average OR Maximum
    ● Linear 1:
        ○ Number of neurons: 10, 20, 30, 40, 50, 60, 70, 80, 90, 100

    '''
        
    def __init__(self, 
                 conv_out = 8, # Number of filters: 8, 16, 32
                 kernel_params = 0, #0 OR 1
                      #0=> kernel=(3,3), stride=1, padding=1
                      #1=> kernel=(5,5), stride=1, padding=2
                 f1 = "ReLU", #possible activation functions:
                      #ReLU OR Sigmoid OR Tanh OR Softplus OR ELU
                 f2 = "ReLU", #possible activation functions:
                      #ReLU OR Sigmoid OR Tanh OR Softplus OR ELU                 
                 pool_size = 2,  #2, identity
                 pooling = "avg", # avg, max
                 linear_out = 10 #10,20,30,40,50,60,70,80,90,100
                 ):
        super(CNN, self).__init__()


        #adjust the parameters depending on the input
        self.conv_out = conv_out 
        if kernel_params == 0: #kernel=(3,3), stride=1, padding=1
            self.kernel_size = 3 
            self.padding = 1 
        else:
            self.kernel_size = 5 
            self.padding = 2

        if pool_size == 2:
            self.pool_size = 2
        else:
            self.pool_size = 1

        self.linear_out = linear_out # Number of neurons

        #find the size of the tensor before entering the fully connected layer
        if True:
          #find the size after the convolutional layers
          size = 8 - self.kernel_size + 2 * self.padding + 1
          #find the size after the pooling layer
          size = (size - self.pool_size) / (self.pool_size) + 1
          #find the size after flattening
          size = int(size**2 * self.conv_out)


        #convolutional layers
        self.cnn =  nn.Conv2d(in_channels = 1, out_channels = conv_out, kernel_size = self.kernel_size, stride = 1, padding = self.padding)

        #activation function 1
        if f1 == "ReLU":    
            self.activation1 = nn.ReLU()
        elif f1 == "Sigmoid":
            self.activation1 = nn.Sigmoid()
        if f1 == "Tanh":    
            self.activation1 = nn.Tanh() 
        elif f1 == "Softplus":
            self.activation1 = nn.Softplus()
        elif f1 == "ELU":
            self.activation1 = nn.ELU()

        #pooling
        if pooling == "avg":
            self.pool = nn.AvgPool2d(kernel_size = pool_size)
        else:
            self.pool = nn.MaxPool2d(kernel_size = pool_size)
       
        #fully connected layer 1
        self.linear1 = nn.Linear(in_features = size, out_features = self.linear_out)

        #activation function 2
        if f2 == "ReLU":    
            self.activation2 = nn.ReLU()
        elif f2 == "Sigmoid":
            self.activation2 = nn.Sigmoid()
        if f2 == "Tanh":    
            self.activation2 = nn.Tanh() 
        elif f2 == "Softplus":
            self.activation2 = nn.Softplus()
        elif f2 == "ELU":
            self.activation2 = nn.ELU()

        #fully connected layer 1
        self.linear2 = nn.Linear(in_features = self.linear_out, out_features = 10)
        #softmax
        self.softmax = nn.LogSoftmax(dim=1)

        #loss funtion
        self.nll = nn.NLLLoss(reduction="none") 
    
    #classify method to find the prediccion of a datapoint
    def classify(self, x):
        #input goes through all the layers
        x = self.cnn(x)
        x = self.activation1(x)
        x = self.pool(x)
        #get flattened before fully conneted layers
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        x = self.activation2(x)
        x = self.linear2(x)
        log_prob = self.softmax(x)
        #find prediction
        y_pred = torch.argmax(log_prob, dim = 1).long()        
        return y_pred

    #foward method for the foward pass
    def forward(self, x, y, reduction="avg"):
        #input goes through all the layers
        x = self.cnn(x)
        x = self.activation1(x)
        x = self.pool(x)
        #get flattened before fully conneted layers
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        x = self.activation2(x)
        x = self.linear2(x)
        log_prob = self.softmax(x)
        
        #find the loss (it has as input the log_prob and targets)
        loss = self.nll(log_prob, y)
        #reduction for loss over a batch (either summ or mean)
        if reduction == "sum":
            return loss.sum()
        else:
            return loss.mean()

    #find how many classifying mistakes were made
    def missclassified_datapoints(self, predictions, targets):
        #number of differences in the two vectors
        return torch.sum(predictions != targets)
    

#########################
#                       #
# Train and Evaluation  #
# Loops (one epoch)     #
# and Plot Function     #
#                       #
#########################

def train_loop(dataloader, model, optimizer, verbose = True):

    #counters for train loss and missclassified and total datapoints
    loss_counter = 0
    missclass_counter = 0
    total_data_points = 0

    #get model on train mode
    model.train()

    #for datapoint and target in dataloader
    for (X, y) in dataloader:

        #get batch size
        batch_size = X.size(0)

        #find the loss by giving a prediction regarding the datapoint
        loss = model(X, y, reduction = "sum")
        #add loss to counter
        loss_counter += loss

        #classify each datapoint and count missclassified points
        predictions = model.classify(X)
        missclassified_datapoints = model.missclassified_datapoints(predictions, y)
        #add missclassified datapoints to counter
        missclass_counter += missclassified_datapoints


        #backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        #update total
        total_data_points += batch_size

    
    #find average loss and average classification error over the epoch 
    average_loss = float(loss_counter / total_data_points)
    average_ce = float(missclass_counter / total_data_points)
    if verbose == True:
      #print average loss and average classification error 
      print("\033[95mTRAINING\033[0m")
      print(f"        average train loss => {average_loss}")
      print(f"        average classification error => {average_ce}")
      print("")
      
    return average_loss, average_ce

def evaluation_loop(dataloader, model, mode = "validation", verbose = True):

    #counters for train loss and missclassified datapoints
    loss_counter = 0
    missclass_counter = 0

    #get model to eval mode
    model.eval()


    #to not save the gradients
    with torch.no_grad():
      #for datapoint and target in dataloader
      for (X, y) in dataloader:

        #find the loss by giving a prediction regarding the datapoint
        loss = model(X, y, reduction = "sum")
        #add loss to counter
        loss_counter += loss

        #classify each datapoint and count missclassified points
        predictions = model.classify(X)
        missclassified_datapoints = model.missclassified_datapoints(predictions, y)
        #add missclassified datapoints to counter
        missclass_counter += missclassified_datapoints

    #find average loss and average classification error over the epoch 
    dataloader_size = len(dataloader.dataset)
    average_loss = float(loss_counter / dataloader_size)
    average_ce = float(missclass_counter / dataloader_size)
    
    if verbose == True:
      #print average loss and average classification error 
      if mode == "validation":
        print("\033[95mVALIDATION\033[0m")
      else:
        print("\033[92mTESTING\033[0m")

      print(f"        average {mode} loss => {average_loss}")
      print(f"        average {mode} classification error => {average_ce}")
      print("")

    return average_loss, average_ce

#function to plot
def plot_results(train_loss_list, train_ce_list, val_loss_list, val_ce_list, test_loss_list = None, test_ce_list = None):
    #number of epoches
    n_epochs = len(train_loss_list)
    #plotting train loss
    plt.plot(range(1, n_epochs + 1), train_loss_list, label="Train Loss")

    #plotting train classification error
    plt.plot(range(1, n_epochs + 1), train_ce_list, label="Train Classification Error")

    #plotting validation loss
    plt.plot(range(1, n_epochs + 1), val_loss_list, label="Validation Loss")

    #plotting validation classification error
    plt.plot(range(1, n_epochs + 1), val_ce_list, label="Validation Classification Error")

    if test_loss_list != None and test_ce_list != None:
      #plotting test loss
      plt.plot(range(1, n_epochs + 1), [test_loss_list] * n_epochs, label="Test Loss", linestyle="--")

      #plotting test classification error
      plt.plot(range(1, n_epochs + 1), [test_ce_list] * n_epochs, label="Test Classification Error", linestyle="--")

    #labels and title
    plt.xlabel("Epochs")
    plt.ylabel("Loss / Classification Error")
    plt.title("Training and Evaluation Results")
    plt.legend()

    #show the plot
    plt.show()

#########################
#                       #
# Dataloaders and Full  #
# Training and Testing  #
#                       #
#########################

#training, validation and test sets.
train_set = DigitsDataset(mode="train")
val_set = DigitsDataset(mode="val")
test_set = DigitsDataset(mode="test")

#data loaders.
training_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = DataLoader(val_set, batch_size=64, shuffle=False)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

#full loopfor training and testing 
def full_cnn_loop(model, number_epoch = 20, testing = True, training_loader = None, val_loader = None, test_loader = None, verbose = True):

    #optimizer
    for param in model.parameters():
      param.requires_grad = True
    opt = torch.optim.Adamax(model.parameters(), lr=1e-3, weight_decay=1e-5)

    #initialize empty lists
    train_loss_list = []
    train_ce_list = []
    val_loss_list =[]
    val_ce_list =[]
    test_loss_list =[]
    test_ce_list =[]

    #progress bar if verbose
    if verbose:
      loop = tqdm(range(number_epoch))
    else:
      loop = range(number_epoch)
      
    #for number of epoches
    for i in loop:
      
      if verbose == True:
        print(f"\033[96mEPOCH NUMBER {i}\033[0m")

      #train loop
      train_loss, train_ce =  train_loop(dataloader = training_loader, model = model, optimizer = opt, verbose = verbose)
      train_loss_list.append(train_loss)
      train_ce_list.append(train_ce)

      #validation loop
      val_loss, val_ce = evaluation_loop(dataloader = val_loader, model = model, verbose = verbose)
      val_loss_list.append(val_loss)
      val_ce_list.append(val_ce)

    #testing loop (only if in testing mode)
    if testing == True:
      test_loss, test_ce = evaluation_loop(dataloader = test_loader, model = model, mode = "test", verbose = verbose)
      test_loss_list.append(test_loss)
      test_ce_list.append(test_ce)
    else:
      test_loss_list = None
      test_ce_list = None

    #plot graphs
    if verbose == True:
       plot_results(train_loss_list, train_ce_list, val_loss_list, val_ce_list, test_loss_list, test_ce_list)

    if testing == True: #if in testing mode
      return min(test_loss_list), max(test_ce_list)
    else: #if in validation mode
      return  min(val_loss_list), max(val_ce_list)
    
'''
#example of a full loop (no specified params for the architecture)
Random_model = CNN()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Random_model.to(device)
_,_ = full_cnn_loop(model = Random_model, number_epoch = 20, testing = True, training_loader = training_loader, val_loader = val_loader, test_loader = test_loader)
'''

#########################
#                       #
#   Neuroevolutionary   # 
#   Algorithm Class     #
#                       #
#########################

class EA(object):
    def __init__(self, number_epoch = 10, pop_size = 10, p_parents = 0.8, nr_parents = 5, nr_candidate_child = 12):

        self.number_epoch = number_epoch #number epoch for each evaluation
        self.pop_size = pop_size #size of a population
        self.p_parents = p_parents #percentage of top indivuals from old generation taken as possible parents
        self.nr_parents = nr_parents #parents taken from each each generation
        self.nr_candidate_child = nr_candidate_child #candidate children after each recombination

        #max value of each entry (min is always 0)
        self.conv_out_max = 2
        self.kernel_params_max = 1
        self.f1_max = 4
        self.f2_max = 4
        self.pool_size_max = 1
        self.pooling_max = 2
        self.linear_out = 9


        #training, validation and test sets
        self.train_set = DigitsDataset(mode="train")
        self.val_set = DigitsDataset(mode="val")
        #data loaders
        self.training_loader = DataLoader(self.train_set, batch_size=64, shuffle=True)
        self.val_loader = DataLoader(self.val_set, batch_size=64, shuffle=False)
       
      
    def parent_selection(self, x_old, f_old):

        #sort parents from best to worst fistness
        sorted_index = np.argsort(f_old)
        sorted_x_parents = np.array(x_old)[sorted_index]
        sorted_f_parents = np.array(f_old)[sorted_index]
        
        p = float(self.p_parents)

        #the number of individuals in p percent
        range_p = int(self.pop_size * p)
        parents_list=[]
        parent_fitness =[]

        #for self.nr_parents times, sample with replacement (from the 100p best individuals from the old generaton)
        for i in range(self.nr_parents):
          #random index within the top range
          top_index = np.random.choice(range(range_p), size=1)
          #get individual at the randomly selected index from the top range
          random_top_ind = sorted_x_parents[top_index]
          random_top_ind_fitness = sorted_f_parents[top_index]
          #add the new parent to the list
          parents_list.append(random_top_ind)
          parent_fitness.append(random_top_ind_fitness)
        #turn list into numpy array
        array_parents = np.stack(parents_list).reshape((self.nr_parents, 7))    

        return array_parents, parent_fitness



    def recombination(self, x_parents, f_parents):
        candidate_children = []
        pop_size = len(x_parents)
        #since each loop 2 children are made => nr_candidate_child/2
        for i in range(int(self.nr_candidate_child/2)):

          #pick two random parents
          idx1, idx2 = np.random.choice(pop_size, size=2, replace=False)
          parent1 = x_parents[idx1]
          parent2 = x_parents[idx2]

          #pick a random number from 1 to 5
          random_number = np.random.randint(1, 6)

          #split the parents into two halves
          half1_1 = x_parents[idx1][:random_number]
          half1_2 = x_parents[idx1][random_number:]
          half2_1 = x_parents[idx2][:random_number]
          half2_2 = x_parents[idx2][random_number:]

          #make the children as the cobination of two (un-paired) halves
          child1 = np.concatenate((half1_1, half2_2))
          child2 = np.concatenate((half2_1, half1_2))
          candidate_children.append(child1)
          candidate_children.append(child2)

        #turn list into numpy array of the right size 
        candidate_children = np.vstack(candidate_children)

        return candidate_children



    def mutation(self, x_children):    
        mutated_children = []
        for x_child in x_children:
            ex = [self.conv_out_max, self.kernel_params_max, self.f1_max, self.f2_max,self.pool_size_max, self.pooling_max,self.linear_out ]
            #pick a random number from 0 to 6
            random_number = np.random.randint(0, 7)
            
            #if the value of the entry picked is 0, add one
            if x_child[random_number] == 0:
              x_child[random_number] = x_child[random_number] + 1
            #if the value of the entry picked is the max, subtract 1
            elif x_child[random_number] == ex[random_number]:
              x_child[random_number] = x_child[random_number] - 1
            #if the value of the entry picked is not the max nor the min, either add or subtract 1
            else:
              random_choice = np.random.choice([-1, 1])
              x_child[random_number] = x_child[random_number] + random_choice
            mutated_children.append(x_child)

        #reshape array
        mutated_children = np.vstack(mutated_children)
        return mutated_children 


    def survivor_selection(self, x_old, x_children, f_old, f_children):
        #combine parent and children populations
        x = np.vstack([x_old, x_children])
        f = np.concatenate([f_old, f_children])
        
        #select the best individuals based on their fitness
        idx_sorted = np.argsort(f)
        x_sorted = x[idx_sorted]
        
        #keep the top individuals as the new population
        x_survivors = x_sorted[:self.pop_size]
        f_survivors = f[idx_sorted][:self.pop_size]

        return x_survivors, f_survivors


    def evaluate(self, x_pop):
        f = []

        #for each architecture in the population
        for x in x_pop:
          #convert genotype into usable paarameters
          y = self.nn_from_genotype(x)
          #initialize cnn and move it to CUDA if available
          cnn_to_evaluate =  CNN(conv_out = y[0], 
                                kernel_params = y[1], 
                                f1 = y[2],
                                f2 = y[3],               
                                pool_size = y[4],
                                pooling = y[5],
                                linear_out = y[6] )
          cnn_to_evaluate.to(device)

          _, validation_ce = full_cnn_loop(model = cnn_to_evaluate, number_epoch = self.number_epoch, testing = False, training_loader = self.training_loader, val_loader = self.val_loader, verbose = False)                 
          
          #count number of parameters in the CNN
          num_parameters = sum(p.numel() for p in cnn_to_evaluate.parameters())
          num_params_max = 206742
          #Objective = ClassError + λ * Np/Nmax
          fitness_x = validation_ce + 0.01*(num_parameters/num_params_max)
          f.append(fitness_x)

        return f

    @staticmethod
    def nn_from_genotype(genotype):
        '''
        transform genotype into parameters usable to create the CNN.
        it follows the schema described above
        '''

        y = []
        #conv_out = 8, 16, 32
        if genotype[0] == 0:
          y.append(8)
        elif genotype[0] == 1:
          y.append(16)
        else:
          y.append(32)

        #kernel params= 0 or 1
        y.append(genotype[1])

        #f1= ReLU OR Sigmoid OR Tanh OR Softplus OR ELU
        if genotype[2] == 0:
          y.append("ReLU")
        elif genotype[2] == 1:
          y.append("Sigmoid")
        elif genotype[2] == 2:
          y.append("Tanh")
        elif genotype[2] == 3:
          y.append("Softplus")
        else:
          y.append("ELU")

        #f1= ReLU OR Sigmoid OR Tanh OR Softplus OR ELU
        if genotype[3] == 0:
          y.append("ReLU")
        elif genotype[3] == 1:
          y.append("Sigmoid")
        elif genotype[3] == 2:
          y.append("Tanh")
        elif genotype[3] == 3:
          y.append("Softplus")
        else:
          y.append("ELU")

        #pool_size = 2 or identity
        if genotype[3] == 0:
          y.append(2)
        else: 
          y.append(1)

        #pooling = avg, max
        if genotype[5] == 0:
          y.append("avg")
        else:
          y.append("max")
        
        #linear_out = 10,20,30,40,50,60,70,80,90,100
        a =  int(genotype[6])
        b = (a+1)*10
        y.append(b)
        
        #return the parameters 
        return y

    def step(self, x_old, f_old):

        x_parents, f_parents = self.parent_selection(x_old, f_old) # return a subset of the old population/parents and how fit they are

        x_children = self.recombination(x_parents, f_parents) # compute a set of INITIAL candidate solutions

        x_children = self.mutation(x_children)  # compute a set of FINAL candidate solutions
        
        f_children = self.evaluate(x_children) # compute the fitness of the FINAL candidate solutions

        x, f = self.survivor_selection(x_old, x_children, f_old, f_children) # select the new population from the old population and the FINAL candidate solutions (or solely from the FINAL candidate solutions)<= commenting the code before implementing

        return x, f # return the new population and their fitness
    

#########################
#                       #
#   Running Loop        #
#                       #
#########################

#function tu create a random genotype
def create_random_idividual():

  ''' Given the following:
        conv_out_max = 2
        kernel_params_max = 1
        f1_max = 4
        f2_max = 4
        pool_size_max = 1
        pooling_max = 2
        linear_out = 9
  '''

  individual = []
  
  #define the range for each entry
  entry_ranges = [
      2,    # Range for entry 1
      1,    # Range for entry 2
      4,    # Range for entry 3
      4,    # Range for entry 4
      1,    # Range for entry 5
      2,    # Range for entry 6
      9     # Range for entry 7
  ]
    
  #generate a random integer within each range and append to the vector
  for entry_range in entry_ranges:
      entry = random.randint(0, entry_range)
      individual.append(entry)
    
  return individual
     
#hyperparameters
num_generations = 50
population_size = 30

#initialize EA
ea = EA(number_epoch = 6, pop_size = population_size, p_parents = 0.4, nr_parents = 20, nr_candidate_child = 40)

#init the population
x = [create_random_idividual() for _ in range(population_size)]
f = ea.evaluate(x)

#gather populations and values of the best candidates to further analyze the algorithm.
populations = []
populations.append(x)
f = np.array(f)
f_best = [f.min()]
best_individual = [x[np.argmin(f)]]

#run the EA.
for i in tqdm(range(num_generations)):
    if i % int(num_generations * 0.1) == 0:
        print("Generation: {}, best fitness: {:.2f}".format(i, f.min()))
    x, f = ea.step(x, f)
    populations.append(x)
    #store best individual of each generation
    if f.min() < f_best[-1]:
        f_best.append(f.min())
        best_individual.append(x[np.argmin(f)])
    else:
        f_best.append(f_best[-1])
        best_individual.append(best_individual[-1])

print("FINISHED!")

#testing the best models
test_loss_for_each_generation = []
test_ce_for_each_generation = []

#for each model, train it and test it
for model in best_individual:
  #from genotype to phenotype
  params = EA.nn_from_genotype(model)

  best_model = CNN(conv_out = params[0],
                   kernel_params = params[1], 
                   f1 = params[2],
                   f2 = params[3],                 
                   pool_size = params[4],
                   pooling = params[5],
                   linear_out = params[6])
  best_model.to(device)
  #properly train and test the model
  #dataloaders are initialized before
  test_loss, test_ce = full_cnn_loop(model = best_model, number_epoch = 10, testing = True, 
                                               training_loader = training_loader, 
                                               val_loader = val_loader, 
                                               test_loader = test_loader, 
                                               verbose = True)
  
  test_loss_for_each_generation.append(test_loss)
  test_ce_for_each_generation.append(test_ce)

#########################
#                       #
#     Results           #
#                       #
#########################

#plotting test loss and test ce over generations
plt.plot(test_loss_for_each_generation, label='Test Loss')
plt.plot(test_ce_for_each_generation, label='Test CE')

#adding labels and title
plt.xlabel('Generation')
plt.ylabel('Loss/CE')
plt.title('Test Loss and Test CE over Generations')
plt.legend()

#display
plt.show()


#results
overall_best_ce =  min(test_ce_for_each_generation)
best_generation = test_ce_for_each_generation.index(overall_best_ce)

overall_best_geno = best_individual[best_generation]

print(f"\033[92mOVERALL RESULTS:\033[0m")
print("")
print(f"The best generation was generation number {best_generation}")
print(f"The best architecture genotype was:  {overall_best_geno}")
print(f"Which achieved a test Classification Error of:  {overall_best_ce}")