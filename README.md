# Neural Network Optimization with a Genetic Algorithm

This project showcases the use of a Genetic Algorithm (GA) to optimize the hyperparameters of a neural network. The goal was to find the best combination of neuron counts, learning rate, and activation functions to maximize classification accuracy on the Heart Disease dataset.

## 📜 Project Overview

A baseline neural network was first created and trained, achieving an accuracy of **82.44%**. Subsequently, a Genetic Algorithm was implemented to evolve a population of neural network architectures over several generations, seeking to improve this performance.

The GA was configured to optimize the following hyperparameters:
* Number of neurons in two hidden layers.
* The learning rate for the Adam optimizer.
* The choice of activation function (ReLU, Tanh, or Sigmoid).

## 🛠️ Technologies & Libraries

* **Language:** Python
* **Core Libraries:**
    * PyTorch (for building the neural network)
    * DEAP (for implementing the Genetic Algorithm)
    * Scikit-learn (for data preprocessing)
    * Pandas & NumPy

## 📈 Results

The Genetic Algorithm successfully identified a superior set of hyperparameters, significantly boosting the model's performance.
* **Base Model Accuracy:** **82.44%**
* **Optimized Model Accuracy:** **98.54%**
* **Improvement:** **+16.10%**

**Optimal Hyperparameters Found:**
* **Layer 1 Neurons:** 102
* **Layer 2 Neurons:** 104
* **Learning Rate:** ~0.0074
* **Activation Function:** ReLU

The evolution of the best accuracy across generations is shown below.

![Evolution of Best Accuracy](https://storage.googleapis.com/agent-tools-public-bucket/hosted_tools_images/635d0383-7c06-47b2-bd74-32f2c8d28e75.png)

## 🚀 How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/PplSeeThis/neural-network-genetic-algorithm-optimization.git](https://github.com/PplSeeThis/neural-network-genetic-algorithm-optimization.git)
    cd neural-network-genetic-algorithm-optimization
    ```
2.  **Create and activate a virtual environment.**
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run the optimization script:**
    ```bash
    python src/main.py
    ```
