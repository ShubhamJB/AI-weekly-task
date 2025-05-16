from approvedimports import *

def make_xor_reliability_plot(train_x, train_y):
    """ Insert code below to  complete this cell according to the instructions in the activity descriptor.
    Finally it should return the fig and axs objects of the plots created.

    Parameters:
    -----------
    train_x: numpy.ndarray
        feature values

    train_y: numpy array
        labels

    Returns:
    --------
    fig: matplotlib.figure.Figure
        figure object
    
    ax: matplotlib.axes.Axes
        axis
    """
    
    # ====> insert your code below here
    hidden_layer_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] # Define the hidden layer sizes to test

    success_counts = [0] * 10 # Initialize a list to keep track of successful trials for each hidden layer size
    epoch_list = [[0 for _ in range(10)] for _ in range(10)] # Initialize a list to keep track of epochs for each trial

    for size_index in  range(len(hidden_layer_sizes)):  # Iterate over each hidden layer size
        current_size = hidden_layer_sizes[size_index]

        for trial in range(10): # Perform 10 trials for each hidden layer size
            mlp_model = MLPClassifier( 
                hidden_layer_sizes = (current_size,),  # Set the current hidden layer size
                max_iter = 1000,  # Set the maximum number of iterations
                alpha = 0.0001,  # Set the regularization parameter
                solver = 'sgd',  # Use stochastic gradient descent as the solver
                learning_rate_init = 0.1,   # Set the initial learning rate
                random_state = trial  # Set the random state for reproducibility
            )

            mlp_model.fit(train_x, train_y)  # Train the model

            accuracy = mlp_model.score(train_x, train_y) * 100  # Calculate the accuracy

            if accuracy == 100:  # Check if the accuracy is 100%
                success_counts[size_index] = success_counts[size_index] + 1
                epoch_list[size_index][trial] = mlp_model.n_iter_

    average_epochs = [0] * 10  # Initialize a list to keep track of the average epochs for each hidden layer size
    for size_index in range(10):  # Iterate over each hidden layer size
        total_epochs = 0
        successful_runs = 0

        for trial in range(10):  # Iterate over each trial
            if epoch_list[size_index][trial] > 0:  # Check if the trial was successful
                total_epochs = total_epochs + epoch_list[size_index][trial]
                successful_runs = successful_runs + 1

            if successful_runs > 0:  # Check if there were any successful runs
                average_epochs[size_index] = total_epochs / successful_runs
            else:
                average_epochs[size_index] = 1000
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))  # Create a figure with two subplots

    ax[0].plot(hidden_layer_sizes, success_counts, marker='o')  # Plot the success counts
    ax[0].set_title("Reliability")  # Set the title of the first subplot
    ax[0].set_xlabel("Hidden Layer Width")  # Set the x-axis label of the first subplot
    ax[0].set_ylabel("Success Rate")  # Set the y-axis label of the first subplot
    ax[0].set_xticks(hidden_layer_sizes)  # Set the x-ticks of the first subplot
    
    ax[1].plot(hidden_layer_sizes, average_epochs, marker='o')  # Plot the average epochs
    ax[1].set_title("Efficiency")  # Set the title of the second subplot
    ax[1].set_xlabel("Hidden Layer Width")  # Set the x-axis label of the second subplot
    ax[1].set_ylabel("Mean Epochs")  # Set the y-axis label of the second subplot
    ax[1].set_xticks(hidden_layer_sizes)  # Set the x-ticks of the second subplot

    plt.tight_layout()  # Adjust the layout of the subplots
    # <==== insert your code above here

    return fig, ax

# make sure you have the packages needed
from approvedimports import *

#this is the class to complete where indicated
class MLComparisonWorkflow:
    """ class to implement a basic comparison of supervised learning algorithms on a dataset """ 
    
    def __init__(self, datafilename:str, labelfilename:str):
        """ Method to load the feature data and labels from files with given names,
        and store them in arrays called data_x and data_y.
        
        You may assume that the features in the input examples are all continuous variables
        and that the labels are categorical, encoded by integers.
        The two files should have the same number of rows.
        Each row corresponding to the feature values and label
        for a specific training item.
        """
        # Define the dictionaries to store the models, and the best performing model/index for each algorithm
        self.stored_models:dict = {"KNN":[], "DecisionTree":[], "MLP":[]}
        self.best_model_index:dict = {"KNN":0, "DecisionTree":0, "MLP":0}
        self.best_accuracy:dict = {"KNN":0, "DecisionTree":0, "MLP":0}

        # Load the data and labels
        # ====> insert your code below here
        self.data_x = np.loadtxt(datafilename, delimiter=",")
        self.data_y = np.loadtxt(labelfilename, delimiter=",")
        # <==== insert your code above here

    def preprocess(self):
        """ Method to 
           - separate it into train and test splits (using a 70:30 division)
           - apply the preprocessing you think suitable to the data
           - create one-hot versions of the labels for the MLP if ther are more than 2 classes
 
           Remember to set random_state = 12345 if you use train_test_split()
        """
        # ====> insert your code below here
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(
            self.data_x, self.data_y, test_size=0.3, random_state=12345, stratify=self.data_y
        )  # Split the data into training and test sets with a 70:30 ratio

        # Normalize features
        min_vals = []  # List to store minimum values for each feature
        max_vals = []  # List to store maximum values for each feature

        # Find min and max for each feature
        for i in range(self.data_x.shape[1]):  # Iterate over each feature
            feature_values = self.data_x[:, i]
            min_vals.append(min(feature_values))
            max_vals.append(max(feature_values))

        # Normalize training data
        train_norm = []  # List to store normalized training data
        for row in self.train_x:
            norm_row = []
            for i in range(len(row)):
                # Normalize each feature value
                if max_vals[i] == min_vals[i]:
                    norm_row.append(0)
                else:
                    norm_row.append((row[i] - min_vals[i]) / (max_vals[i] - min_vals[i]))
            train_norm.append(norm_row)

        # Normalize test data
        test_norm = []
        for row in self.test_x:
            norm_row = []
            for i in range(len(row)):
                # Normalize each feature value
                if max_vals[i] == min_vals[i]:
                    norm_row.append(0)
                else:
                    norm_row.append((row[i] - min_vals[i]) / (max_vals[i] - min_vals[i]))
            test_norm.append(norm_row)

         # Convert normalized data to arrays
        self.train_x = np.array(train_norm)
        self.test_x = np.array(test_norm)

        # Create one-hot encoded labels if needed
        unique_classes = list(set(self.data_y))
        num_classes = len(unique_classes)

        # Check if we need one-hot encoding
        if num_classes > 2:
            # Prepare one-hot training labels
            train_onehot = []
            for label in self.train_y:
                onehot = [0] * num_classes
                class_index = unique_classes.index(label)
                onehot[class_index] = 1
                train_onehot.append(onehot)

            # Prepare one-hot test labels
            test_onehot = []
            for label in self.test_y:
                onehot = [0] * num_classes
                class_index = unique_classes.index(label)
                onehot[class_index] = 1
                test_onehot.append(onehot)

            # Store one-hot labels
            self.train_y_onehot = np.array(train_onehot)
            self.test_y_onehot = np.array(test_onehot)
        else:
            # Binary classification - no need for one-hot
            self.train_y_onehot = self.train_y
            self.test_y_onehot = self.test_y
        # <==== insert your code above here
    
    def run_comparison(self):
        """ Method to perform a fair comparison of three supervised machine learning algorithms.
        Should be extendable to include more algorithms later.
        
        For each of the algorithms KNearest Neighbour, DecisionTreeClassifer and MultiLayerPerceptron
        - Applies hyper-parameter tuning to find the best combination of relevant values for the algorithm
         -- creating and fitting model for each combination, 
            then storing it in the relevant list in a dictionary called self.stored_models
            which has the algorithm names as the keys and  lists of stored models as the values
         -- measuring the accuracy of each model on the test set
         -- keeping track of the best performing model for each algorithm, and its index in the relevant list so it can be retrieved.
        
        """
        # ====> insert your code below here
        # KNN parameter tuning
        k_values = [1, 3, 5, 7, 9]  # List of k values to try
        for i, k in enumerate(k_values):
            # Create and train KNN model with the current k value
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(self.train_x, self.train_y)
            self.stored_models["KNN"].append(knn) # Store the trained model

            # Test accuracy of the model
            predictions = knn.predict(self.test_x)
            correct = 0
            for j in range(len(self.test_y)):
                if predictions[j] == self.test_y[j]:
                    correct += 1
            accuracy = (correct / len(self.test_y)) * 100

            # Update best model if the current one is better
            if accuracy > self.best_accuracy["KNN"]:
                self.best_accuracy["KNN"] = accuracy
                self.best_model_index["KNN"] = i

        # Decision Tree parameter tuning
        depths = [1, 3, 5]  # List of max depths to try
        splits = [2, 5, 10]  # List of min samples split to try
        leafs = [1, 5, 10]  # List of min samples leaf to try

        dt_index = 0  # Index to keep track of the best Decision Tree model
        for depth in depths:
            for split in splits:
                for leaf in leafs:
                    # Create and train Decision Tree with the current parameters
                    dt = DecisionTreeClassifier(
                        max_depth=depth,
                        min_samples_split=split,
                        min_samples_leaf=leaf,
                        random_state=12345
                    )
                    dt.fit(self.train_x, self.train_y)
                    self.stored_models["DecisionTree"].append(dt)  # Store the trained model

                    # Test accuracy of the model
                    predictions = dt.predict(self.test_x)
                    correct = 0
                    for j in range(len(self.test_y)):
                        if predictions[j] == self.test_y[j]:
                            correct += 1
                    accuracy = (correct / len(self.test_y)) * 100

                    # Update best model if the current one is better
                    if accuracy > self.best_accuracy["DecisionTree"]:
                        self.best_accuracy["DecisionTree"] = accuracy
                        self.best_model_index["DecisionTree"] = dt_index

                    dt_index += 1

        # MLP parameter tuning
        first_layers = [2, 5, 10] # List of sizes for the first hidden laye
        second_layers = [0, 2, 5]  # List of sizes for the second hidden layer (0 means no second layer)
        activations = ["logistic", "relu"]  # List of activation functions to try

        mlp_index = 0  # Index to keep track of the best MLP model
        for first in first_layers:
            for second in second_layers:
                for activation in activations:
                    # Set up layer sizes
                    if second == 0:
                        layers = (first,)
                    else:
                        layers = (first, second)

                    # Create and train MLP with the current parameters
                    mlp = MLPClassifier(
                        hidden_layer_sizes=layers,
                        activation=activation,
                        max_iter=1000,
                        random_state=12345
                    )
                    mlp.fit(self.train_x, self.train_y_onehot)
                    self.stored_models["MLP"].append(mlp)  # Store the trained model

                    # Test accuracy of the model
                    predictions = mlp.predict(self.test_x)
                    correct = 0

                    # Check if multiclass
                    if len(set(self.data_y)) > 2:
                        for j in range(len(self.test_y)):
                            # Find predicted class
                            pred_class = 0
                            max_val = predictions[j][0]
                            for k in range(1, len(predictions[j])):
                                if predictions[j][k] > max_val:
                                    max_val = predictions[j][k]
                                    pred_class = k

                            # Find true class
                            true_class = 0
                            max_val = self.test_y_onehot[j][0]
                            for k in range(1, len(self.test_y_onehot[j])):
                                if self.test_y_onehot[j][k] > max_val:
                                    max_val = self.test_y_onehot[j][k]
                                    true_class = k

                            # Check if correct
                            if pred_class == true_class:
                                correct += 1
                    else:
                        # Binary classification
                        for j in range(len(self.test_y)):
                            if predictions[j] == self.test_y[j]:
                                correct += 1

                    accuracy = (correct / len(self.test_y)) * 100

                    # Update best model if the current one is better
                    if accuracy > self.best_accuracy["MLP"]:
                        self.best_accuracy["MLP"] = accuracy
                        self.best_model_index["MLP"] = mlp_index

                    mlp_index += 1 
        # <==== insert your code above here
    
    def report_best(self) :
        """Method to analyse results.

        Returns
        -------
        accuracy: float
            the accuracy of the best performing model

        algorithm: str
            one of "KNN","DecisionTree" or "MLP"
        
        model: fitted model of relevant type
            the actual fitted model to be interrogated by marking code.
        """
        # ====> insert your code below here
        best_algo = ""
        best_acc = 0

        # Check each algorithm
        algos = ["KNN", "DecisionTree", "MLP"]
        for algo in algos:
            if self.best_accuracy[algo] > best_acc:
                best_acc = self.best_accuracy[algo]
                best_algo = algo

        # Get the best model
        best_model = self.stored_models[best_algo][self.best_model_index[best_algo]]

        # Return results
        return best_acc, best_algo, best_model
        # <==== insert your code above here
