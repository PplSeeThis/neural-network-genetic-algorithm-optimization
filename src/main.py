import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from deap import base, creator, tools, algorithms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# --- Configuration ---
# NOTE: Download the dataset from https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data
# and update this path.
DATA_PATH = "../input/heart-disease-data/heart.csv"

# Genetic Algorithm Parameters
POPULATION_SIZE = 20
NUM_GENERATIONS = 30
CXPB = 0.6  # Crossover probability
MUTPB = 0.3  # Mutation probability

# --- 1. Data Preparation ---
def get_data_loaders():
    """Loads and preprocesses the Heart Disease dataset."""
    data = pd.read_csv(DATA_PATH)
    
    X = data.drop('target', axis=1).values
    y = data['target'].values
    
    # Split and scale data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.FloatTensor(y_test).view(-1, 1)
    
    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor

# --- 2. Neural Network Definition ---
class HeartDiseaseNN(nn.Module):
    """A simple neural network for classification."""
    def __init__(self, n1, n2, activation_fn):
        super(HeartDiseaseNN, self).__init__()
        self.layer1 = nn.Linear(X_train.shape[1], n1)
        self.layer2 = nn.Linear(n1, n2)
        self.output = nn.Linear(n2, 1)
        self.activation = activation_fn

    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        x = torch.sigmoid(self.output(x))
        return x

# --- 3. Genetic Algorithm Fitness Function ---
# Global variables for data tensors to avoid passing them repeatedly
X_train, y_train, X_test, y_test = get_data_loaders()

def evaluate(individual):
    """
    Fitness function for the GA. Trains and evaluates a neural network
    based on the hyperparameters in the 'individual'.
    """
    # Decode the individual's genes into hyperparameters
    n1, n2, lr_log, act_idx = individual
    n1 = int(n1)
    n2 = int(n2)
    lr = 10**lr_log # Log-uniform learning rate
    activation_functions = [nn.ReLU(), torch.tanh, nn.Sigmoid()]
    activation_fn = activation_functions[int(act_idx)]

    # Create and train the model
    model = HeartDiseaseNN(n1, n2, activation_fn)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Simple training loop for 5 epochs
    for _ in range(5):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
    # Evaluate the model
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        predicted = (outputs > 0.5).float()
        accuracy = (predicted == y_test).float().mean()
        
    return (accuracy.item(),) # DEAP requires a tuple

# --- 4. Genetic Algorithm Setup ---
def setup_ga():
    """Sets up the DEAP toolbox for the genetic algorithm."""
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # Define attributes (genes) for an individual
    # n1, n2: number of neurons (integer)
    # lr: learning rate (float, log scale from 1e-4 to 1e-1)
    # act_idx: activation function index (integer)
    toolbox.register("attr_n", random.randint, 8, 128)
    toolbox.register("attr_lr", random.uniform, -4, -1) 
    toolbox.register("attr_act", random.randint, 0, 2)

    # Define the structure of an individual
    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_n, toolbox.attr_n, toolbox.attr_lr, toolbox.attr_act), n=1)

    # Define the population
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Register GA operators
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    return toolbox

# --- 5. Main Execution ---
if __name__ == "__main__":
    toolbox = setup_ga()
    pop = toolbox.population(n=POPULATION_SIZE)
    
    # Statistics to keep track of the evolution
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    # Run the genetic algorithm
    print("\n--- Starting Genetic Algorithm Optimization ---")
    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=CXPB, mutpb=MUTPB, ngen=NUM_GENERATIONS, 
                                       stats=stats, verbose=True)
    print("--- Finished Optimization ---\n")

    # Get the best individual
    best_ind = tools.selBest(pop, 1)[0]
    best_fitness = best_ind.fitness.values[0]

    print("--- Best Individual Found ---")
    print(f"Hyperparameters: {best_ind}")
    print(f"  - Layer 1 Neurons: {int(best_ind[0])}")
    print(f"  - Layer 2 Neurons: {int(best_ind[1])}")
    print(f"  - Learning Rate: {10**best_ind[2]:.5f}")
    print(f"  - Activation Index: {int(best_ind[3])} (0:ReLU, 1:Tanh, 2:Sigmoid)")
    print(f"Best Accuracy: {best_fitness*100:.2f}%")

    # Plotting the results
    gen = logbook.select("gen")
    max_fitness = logbook.select("max")
    avg_fitness = logbook.select("avg")

    fig, ax1 = plt.subplots()
    line1 = ax1.plot(gen, max_fitness, "b-", label="Maximum Fitness")
    line2 = ax1.plot(gen, avg_fitness, "r-", label="Average Fitness")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness (Accuracy)")
    ax1.set_title("Evolution of Fitness Over Generations")
    ax1.grid(True)
    
    lns = line1 + line2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="best")
    
    plt.show()
