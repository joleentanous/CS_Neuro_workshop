import pandas as pd
import cv2
import os
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score

# Load the dataset
dataset_path = 'list_attr_celeba.csv'  # Update with the dataset path
df = pd.read_csv(dataset_path)

folder_path = 'C:\Users\jolee\OneDrive\Desktop\שנה ג\סמסטר ב\workshop\codingWithMP\grid_search\img_align_celeba'
image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
image_files.sort()  # Sort the filenames to ensure a consistent order


# Extract columns for image paths and labels
image_paths = df['image_path'].tolist()  # Update 'image_path' based on the column name in df
true_labels = df['Smiling'].values()  # Update 'label' based on the column name in df
true_labels = np.where(true_labels == -1, 0, true_labels)

# Define the target blendshapes
target_blendshapes = [
    'mouthSmileLeft',
    'mouthSmileRight',
    'mouthPressLeft',
    'mouthPressRight',
    'eyeSquintLeft',
    'eyeSquintRight'
]

# Initialize the custom FaceLandmarker with the custom model
base_options = python.BaseOptions(model_asset_path="C:/Users/jolee/OneDrive/Desktop/odays_code/face_landmarker_v2_with_blendshapes.task")
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

# Function to extract blendshapes
def extract_blendshapes(image):
    results = detector.detect(image)
    if results.multi_face_landmarks:
        # Extract blendshape values
        blendshape_values = [blendshape.score for blendshape in results.multi_face_blendshapes[0] if blendshape.category_name in target_blendshapes]
        return blendshape_values
    return None

# Process each image in the dataset
all_blendshapes_data = []

for image_file in image_files:
    image_path = os.path.join(folder_path, image_file)
    image = cv2.imread(image_path)
    blendshapes = extract_blendshapes(image)
    
    if blendshapes is not None:
        all_blendshapes_data.append(blendshapes)

# Convert all blendshapes data to a NumPy array
blendshapes_array = np.array(all_blendshapes_data)

# Convert the blendshapes array to a DataFrame
blendshapes_df = pd.DataFrame(blendshapes_array)

# # Save the DataFrame to a CSV file (optional)
# csv_filename = 'blendshapes_data.csv'
# blendshapes_df.to_csv(csv_filename, index=False)

# Prepare data for GridSearch
X = blendshapes_array
y = true_labels # 1 for smile , 0 otherwise

# Define a scorer function
def threshold_classifier(X, thresholds):
    return np.all(X > thresholds, axis=1).astype(int)

# Define a scorer function for GridSearchCV
def f1_scorer(estimator, X, y):
    y_pred = threshold_classifier(X, estimator['thresholds'])
    return f1_score(y, y_pred)

# Define parameter grid
param_grid = {
    'thresholds': [np.full(X.shape[1], t) for t in np.arange(0, 1, 0.01)]
}

# Perform grid search
grid_search = GridSearchCV(
    estimator={'thresholds': None},
    param_grid=param_grid,
    scoring=make_scorer(f1_scorer),
    cv=5,
    refit=True
)
grid_search.fit(X, y)

# Get the best parameters and score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Best Parameters:", best_params)
print("Best F1 Score:", best_score)
