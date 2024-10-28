# workshop---mediapipe
resources docs link:
https://docs.google.com/document/d/1fOdxSnMz_m_EC0rMcg6iA6Qe2_hR3lgJygnGYv1kWdo/edit




**Code Overview**
Import Libraries:

Imports necessary libraries such as pandas, os, PIL, MediaPipe, numpy, sklearn, etc.
**Load and Display the Dataset:**

Loads a dataset from a CSV file containing image attributes and displays the first few rows.
Filters the dataset to only the first 500 rows.
**Prepare Image Files:**

Lists image files from a specified directory and sorts them to ensure a consistent order.
**Define Target Blendshapes:**

Specifies a list of blendshape categories related to smiling and squinting.
**Initialize MediaPipe FaceLandmarker:**

Initializes a custom FaceLandmarker using a provided MediaPipe model file.
**Extract Blendshapes from Images:**

Defines a function to extract blendshape values from an image.
Processes each image in the dataset to extract blendshape values.
Tracks and removes any images where extraction fails.
**Prepare Data for Grid Search:**

Converts the extracted blendshape values to a NumPy array.
Converts the blendshape array to a DataFrame.
Extracts the true labels (whether the person is smiling) from the dataset.
Prepares the feature matrix X and label vector y.
**Define Custom Classifier and Scorer:**

Implements a custom ThresholdClassifier that predicts based on whether blendshape scores exceed certain thresholds.
Defines a custom F1 scorer function for evaluating the classifier.
**Perform Grid Search:**

Defines a parameter grid of possible threshold values.
Uses GridSearchCV to search for the best threshold values that maximize the F1 score.
Fits the grid search on the data.
**Output Best Parameters and Score:**

Prints the best threshold values and the best F1 score found by the grid search.
Expected Results
**Best Parameters:**

The best threshold values for the blendshape scores that maximize the F1 score. These thresholds indicate the optimal cut-off points for deciding whether a person is smiling based on the blendshape scores.
**Best F1 Score:**

The highest F1 score achieved using the best threshold values during the grid search. The F1 score is a measure of a test's accuracy, considering both precision and recall.
