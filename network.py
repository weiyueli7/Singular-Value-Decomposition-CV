# import statements
import numpy as np
import data
import time
from tqdm import tqdm

"""
NOTE
----
Start by implementing your methods in non-vectorized format - use loops and other basic programming constructs.
Once you're sure everything works, use NumPy's vector operations (dot products, etc.) to speed up your network.
"""

def sigmoid(a):
    """
    Compute the sigmoid function.
    f(x) = 1 / (1 + e ^ (-x))
    Parameters
    ----------
    a
        The internal value while a pattern goes through the network.
    Returns
    -------
    Array
       Value after applying sigmoid to each image.
    """
    # Sigmoid equation given by the write up
    return 1 / (1 + np.exp(-1 * a))

def softmax(a):
    """
    Compute the softmax function.
    f(x) = (e^x) / Σ (e^x)
    Parameters
    ----------
    a
        The internal value while a pattern goes through the network.
    Returns
    -------
    2-D Array
       Value after applying softmax.
       Each row will represent a image.
       Each column will reveal the chance an image is c class.
    """
    # Softmax equation given by the write up
    return np.exp(a) / np.sum(np.exp(a), axis = 1).reshape(-1,1)

def binary_cross_entropy(y, t):
    """
    Compute binary cross entropy.
    L(x) = t*ln(y) + (1-t)*ln(1-y)
    Parameters
    ----------
    y
        The network's predictions
    t
        The corresponding targets
    Returns
    -------
    Array 
        Binary cross entropy loss value for each image prediction according to above definition.
    """
    # loss function for logistic regression
    return -1 * (t * np.log(y + 1e-20) + (1 - t) * np.log(1 - y + 1e-20))

def multiclass_cross_entropy(y, t):
    """
    Compute multiclass cross entropy.
    L(x) = - Σ (t*ln(y))
    Parameters
    ----------
    y
        The network's predictions
    t
        The corresponding targets
    Returns
    -------
    float 
        multiclass cross entropy loss value according to above definition.
    """
    # loss function for softmax regression
    return -1 * np.sum(t * np.log(y))

class Network:
    def __init__(self, hyperparameters, activation, loss, out_dim):
        """
        Perform required setup for the network.
        Initialize the weight matrix, set the activation function, save hyperparameters.
        You may want to create arrays to save the loss values during training.
        Parameters
        ----------
        hyperparameters
            A Namespace object from `argparse` containing the hyperparameters.
        activation
            The non-linear activation function to use for the network.
        loss
            The loss function to use while training and testing.
        """
        # initial class parameters
        self.hyperparameters = hyperparameters
        self.activation = activation
        self.loss = loss
        self.out_dim = out_dim
        self.weights = np.zeros((28*28+1, self.out_dim))
        
        # initial arrays that store average train / validation loss for each fold
        self.fold_train_loss_history = np.array([])
        self.fold_val_loss_history = np.array([]) 
        
        # initial array that store average validation accuracy for each fold
        self.fold_val_accuracy_history = np.array([])
        
        # initial array that store average weights for each fold
        self.best_weights = np.array([])

        # initial array that store the train / validation loss for each epoch among all folds
        self.epoch_val_loss_record = np.zeros((self.hyperparameters[1] + 1)).reshape(-1,1)
        self.epoch_train_loss_record = np.zeros((self.hyperparameters[1] + 1)).reshape(-1,1)
        
        # initial array that store the number of fold each epoch has been called
        self.index_count = np.zeros((self.hyperparameters[1] + 1, 1))

    def forward(self, X):
        """
        Apply the model to the given patterns
        Use `self.weights` and `self.activation` to compute the network's output
        f(x) = σ(w*x)
            where
                σ = non-linear activation function
                w = weight matrix
        Make sure you are using matrix multiplication when you vectorize your code!
        Parameters
        ----------
        X
            Patterns to create outputs for
        """
        # call activation function with parameter: multiplication of design matrix and weights
        return self.activation(X @ self.weights)
        
    def __call__(self, X):
        return self.forward(X)
    
    def gradient(self, t, X):
        # predict the label based on input designed matrix
        predicted_label = self(X)
        
        # calculate the gradient through matrix multiplication
        if self.activation == sigmoid:
            return -1 * X.T @ (t - (predicted_label.T)).T
        return -1 * X.T @ (t - predicted_label)
        
    def train(self, minibatch):
        """
        Train the network on the given minibatch
        Use `self.weights` and `self.activation` to compute the network's output
        Use `self.loss` and the gradient defined in the slides to update the network.
        Parameters
        ----------
        minibatch
            The minibatch to iterate over
        Returns
        -------
        tuple containing:
            average loss over minibatch
            accuracy over minibatch
        """
        X, y = minibatch
        network_output = self(X)
        
        # update the network based on the gradient
        self.weights = self.weights - (self.hyperparameters[2] * self.gradient(y, X) / len(X))
        if self.activation == sigmoid: 
            # replace the predicted output with 0 and 1
            apply_threshold_train = np.where(np.round(network_output,5) >= 0.5, 1, network_output)
            apply_threshold_train_2 = np.where(np.round(apply_threshold_train,5) < 0.5, 0, apply_threshold_train)
            
            # average loss / accuracy based on model predictions and target label
            avg_loss = np.mean(self.loss(self(X), y.reshape(-1,1)))
            avg_accuracy = np.mean(apply_threshold_train_2.T == y)
        else:
            # average loss / accuracy based on model predictions and target label
            avg_loss = self.loss(self(X), y) / len(X)
            avg_accuracy = np.mean(np.argmax(self(X), axis = 1) == data.onehot_decode(y))
        return avg_loss, avg_accuracy
    
    def test(self, minibatch):
        """
        Test the network on the given minibatch
        Use `self.weights` and `self.activation` to compute the network's output
        Use `self.loss` to compute the loss.
        Do NOT update the weights in this method!
        Parameters
        ----------
        minibatch
            The minibatch to iterate over
        Returns
        -------
            tuple containing:
                average loss over minibatch
                accuracy over minibatch
        """
        X, y = minibatch
        network_output = self(X)
        if self.activation == sigmoid: 
            # replace the predicted output with 0 and 1
            apply_threshold_train = np.where(np.round(network_output,5) >= 0.5, 1, network_output)
            apply_threshold_train_2 = np.where(np.round(apply_threshold_train,5) < 0.5, 0, apply_threshold_train)
            
            # average loss / accuracy based on model predictions and target label
            avg_accuracy = np.mean(apply_threshold_train_2.T == y)
            avg_loss = np.mean(self.loss(self(X), y.reshape(-1,1)))
        else:
            # average loss / accuracy based on model predictions and target label
            avg_loss = self.loss(self(X), y) / len(X)
            avg_accuracy = np.mean(np.argmax(self(X), axis = 1) == data.onehot_decode(y))
        return avg_loss, avg_accuracy
    
    def predict(self, dataset):
        # initialize fold generator
        fold_generator = data.generate_k_fold_set(dataset, k = self.hyperparameters[4])
        
        # loop through each fold
        for train_set, val_set in tqdm(fold_generator):
            
            # initial arrays to store the information for each epoch
            epoch_val_accuracy = np.array([])
            epoch_best_weight = np.array([])
            epoch_train_loss = np.array([])
            epoch_val_loss = np.array([])
            self.weights = np.zeros((28*28+1, self.out_dim))
            epoch_counter = 0
            
            # loop through each epoch in current fold
            for epoch in np.arange(self.hyperparameters[1]):
                
                # initial arrays to store the information for each batch
                train_set = data.shuffle(train_set)
                batch_train_loss = np.array([])
                
                # generate the minibatch for training set
                for batch_train in data.generate_minibatches(train_set, batch_size=self.hyperparameters[0]):
                    
                    # shuffle training set
                    batch_train = data.shuffle(batch_train)
                    
                    # train current minibatch
                    train_result = self.train(batch_train)
                    
                    # append train loss for current minibatch
                    batch_train_loss = np.append(batch_train_loss, train_result[0])
                
                # Test the validation loss / accuracy
                current_val_loss = self.test(val_set)[0]
                current_val_accuracy = self.test(val_set)[1]
                
                # apply early stop if the average validation loss begin to increase
                if len(epoch_val_loss) > 0 and current_val_loss > epoch_val_loss[-1]:
                    break
                else:
                    epoch_counter += 1
                    
                    # record the current weight, epoch training set / validation set loss
                    epoch_best_weight = np.append(epoch_best_weight, self.weights)
                    self.epoch_val_loss_record[epoch_counter] += current_val_loss
                    self.epoch_train_loss_record[epoch_counter] += np.mean(batch_train_loss)
                    
                    # temporary record the current weight, epoch training set / validation set loss
                    epoch_train_loss = np.append(epoch_train_loss, np.mean(batch_train_loss))
                    epoch_val_loss = np.append(epoch_val_loss, current_val_loss)
                    epoch_val_accuracy = np.append(epoch_val_accuracy, current_val_accuracy)
                    
                    # mark current epoch
                    self.index_count[epoch_counter] += 1
                    
                    # record the current validation accuracy
                    epoch_val_accuracy = np.append(epoch_val_accuracy, current_val_accuracy)
            
            # reshape array to store weight in correct format
            epoch_best_weight = epoch_best_weight.reshape(-1, len(self.weights), len(self.weights.T))
            
            # save the best weights of current fold
            self.best_weights = np.append(self.best_weights, epoch_best_weight[-1])
            
            # save the fold train / validation loss and validation accuracy
            self.fold_train_loss_history = np.append(self.fold_train_loss_history, np.mean(epoch_train_loss))
            self.fold_val_loss_history = np.append(self.fold_val_loss_history, np.mean(epoch_val_loss))
            self.fold_val_accuracy_history = np.append(self.fold_val_accuracy_history, np.mean(epoch_val_accuracy))
        
        # reshape array to store weight in correct format
        self.best_weights = self.best_weights.reshape(-1, len(self.weights), len(self.weights.T))
        
        # best weight index
        best_weight_idx = np.where(self.fold_val_loss_history == np.min(self.fold_val_loss_history))[0][0]
        
        # update weights
        self.weights = self.best_weights[best_weight_idx]
        return self.weights
