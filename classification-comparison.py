import os
import sys
import pandas as pd
import numpy as np

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        Trying to Find the Best Performance Classifier
        for the Sonar Mines vs. Rocks UCI Berkeley Dataset
        @Author: Thomas @Date: 27/11/2018 @Version: 1.0
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

# List of built-in classifiers
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier 


from sklearn.metrics.classification import accuracy_score
from sklearn.metrics.classification import confusion_matrix
from sklearn.metrics.classification import classification_report


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        CHANGE TO THE PATH OF YOUR DOWNLOAD LOCATION AS NECESSARY
Make sure you download sonar.all-data.csv to the same directory as well
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Set the Working Directory to read sonar.all-data.csv
# (Same as project directory)
	
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                Acquiring Data and Data Preprocessing
                Data Set Selection and Pre-Processing.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Have chosen the sonar dataset
# http://archive.ics.uci.edu/ml/datasets/connectionist+bench+(sonar,+mines+vs.+rocks)
# For details and more information go to the site above.

# Store this file in the same directory as the program
csv_filename = "sonar.all-data.csv"

# Using a Pandas DataFrame
dataframe = pd.read_csv(csv_filename)

# Check out the top 5 Rows
dataframe.head()

# Dimensions of the input data array
dataframe.describe()

# Inputs (data values) sonar readings from an underground submarine. Cool!
X = dataframe.values[:, 0:-1].astype(float)

# Convert classes M (Mine) and R (Rock) to numbers,
# since they're categorical values.
le = LabelEncoder()

# Classification target
target = dataframe.R

# Do conversion
le = LabelEncoder.fit(le, y=["R", "M"])
y = le.transform(target)

# Data Normalizing and Scaling not required for this dataset

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        Configuring and Setting up the Scikit-Learn Classifiers:
                Hard-Coding as Much as Possible
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# Names of the classifiers that we will compare with each other
names = ["K Nearest Neighbors",
         "RBF SVM",
         "Gaussian Process",
         "Decision Tree"
         "Random Forest",
         "Neural Network",
         "AdaBoost",
         "Gradient Boosting Classifier",
         "Extra Trees Classifier"
         ]

# Code implementations in the scikit-learn library
classifiers = [
    KNeighborsClassifier(2),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=10),
    RandomForestClassifier(max_depth=25, n_estimators=50),
    MLPClassifier(max_iter=3000, alpha=1),
    AdaBoostClassifier(),
    GradientBoostingClassifier(n_estimators=200),
    ExtraTreesClassifier()
]   
num_classifiers = len(classifiers)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                Training a Number of Scikit-Learn Classifiers
                Finding the Best Performing Classifier
Printing the Confusion Matrix and Classification Report for Each Classifier
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# Divide into train and test data, set random_state to get reproducible output
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state = 42)

# Set Up basic Parameters
max_iter = 3
scores = {}
classifiers_range = np.arange(0, num_classifiers)

# Run every classifier and store the results in scores array of 2-element lists. 
# Print out confusion matrix and classification report for every classifier
for iter in np.arange(0, max_iter):
    for classifier, num in zip(classifiers, classifiers_range):
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        cv_score = max(cross_val_score(classifier, X, y, cv=5))
        print("\n" + names[num])
        print("\nAccuracy: " + str(accuracy))
        print("\nCross-Validation Score: " + str(cv_score))
        print("\n " + str(confusion_matrix(y_test, y_pred)))
        print("\n " + str(classification_report(y_test, y_pred)))
        scores[iter, num] = [accuracy, cv_score]

# Print out the best classifier with the highest cross-validation score
best_classifier = ""
max_mean_cv_score = 0.0

# Find the classifier with the highest cross_val_score()
for iter in range(0, max_iter):
    for i in range(0, num_classifiers):
        if max_mean_cv_score < (scores[iter, i])[1]:
            max_mean_cv_score = (scores[iter, i])[1]
            best_classifier = names[i]
				
# Final Best Classifier	- will vary depending on random_state in train_test_split			
print("\nBest Performance: " + best_classifier, max_mean_cv_score, "\n")

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Logging of all Classifiers used and their Performance cross_val_score metrics
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Print scores of all the classifiers
for i in range(0, max_iter):
    print("\nIteration "+str(i+1))
    for j in classifiers_range:
        print("\n " + names[j] + ": " + str((scores[i, j])[1]))

		   
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

--------------------------------E--O--F--------------------------------------

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""