# NN-EXP-2-Implementation-of-Perceptron

## AIM:
To implement a perceptron for classification using Python

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:
A Perceptron is a basic learning algorithm invented in 1959 by Frank Rosenblatt. It is meant to mimic the working logic of a biological neuron. The human brain is basically a collection of many interconnected neurons. Each one receives a set of inputs, applies some sort of computation on them and propagates the result to other neurons.
A Perceptron is an algorithm used for supervised learning of binary classifiers.Given a sample, the neuron classifies it by assigning a weight to its features. To accomplish this a Perceptron undergoes two phases: training and testing. During training phase weights are initialized to an arbitrary value. Perceptron is then asked to evaluate a sample and compare its decision with the actual class of the sample.If the algorithm chose the wrong class weights are adjusted to better match that particular sample. This process is repeated over and over to finely optimize the biases. After that, the algorithm is ready to be tested against a new set of completely unknown samples to evaluate if the trained model is general enough to cope with real-world samples.
The important Key points to be focused to implement a perceptron:
Models have to be trained with a high number of already classified samples. It is difficult to know a priori this number: a few dozen may be enough in very simple cases while in others thousands or more are needed.
Data is almost never perfect: a preprocessing phase has to take care of missing features, uncorrelated data and, as we are going to see soon, scaling.
Perceptron requires linearly separable samples to achieve convergence.
The math of Perceptron
If we represent samples as vectors of size n, where ‘n’ is the number of its features, a Perceptron can be modeled through the composition of two functions. The first one 
f(x) maps the input features  ‘x’  vector to a scalar value, shifted by a bias ‘b’

A threshold function, usually Heaviside or sign functions, maps the scalar value to a binary output:

Indeed if the neuron output is exactly zero it cannot be assumed that the sample belongs to the first sample since it lies on the boundary between the two classes. Nonetheless for the sake of simplicity,ignore this situation.

## ALGORITHM:
* Importing the libraries
* Importing the dataset
* Plot the data to verify the linear separable dataset and consider only two classes
* Convert the data set to scale the data to uniform range by using Feature scaling
* Split the dataset for training and testing
* Define the input vector ‘X’ from the training dataset
* Define the desired output vector ‘Y’ scaled to +1 or -1 for two classes C1 and C2
* Assign Initial Weight vector ‘W’ as 0 as the dimension of ‘X’
* Assign the learning rate
* For ‘N ‘ iterations ,do the following:
        v(i) = w(i)*x(i)
        W (i+i)= W(i) + learning_rate*(y(i)-t(i))*x(i)
* Plot the error for each iteration 
* Print the accuracy


 ## PROGRAM:
 ```
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv("/content/drive/MyDrive/Neural Networks Dataset/IRIS.csv")

# Preprocess the data
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Encode the target variable (species)
le = LabelEncoder()
y = le.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the Perceptron class
class Perceptron:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
    
    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])
        self.bias = 0
        
        for _ in range(self.n_iterations):
            for xi, target in zip(X, y):
                prediction = self.predict(xi)
                update = self.learning_rate * (target - prediction)
                self.weights += update * xi
                self.bias += update
    
    def predict(self, X):
        return np.where(np.dot(X, self.weights) + self.bias > 0, 1, 0)

# Initialize and train the Perceptron
perceptron = Perceptron(learning_rate=0.1, n_iterations=1000)
perceptron.fit(X_train, y_train)

# Make predictions on the test set
y_pred = perceptron.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
```
## OUTPUT:
Accuracy: 30.00%

## RESULT:
Thus, the implementation of perceptron for classification using Python has been executed successfully.
