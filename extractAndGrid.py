import pandas as pd
import cv2
import os
from PIL import Image
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score

# Load the dataset
dataset_path = 'C:/Users/jolee/OneDrive/Desktop/שנה ג/סמסטר ב/workshop/codingWithMP/grid_search/list_attr_celeba.csv'  # Update with the dataset path
df = pd.read_csv(dataset_path)

folder_path = 'C:/Users/jolee/OneDrive/Desktop/שנה ג/סמסטר ב/workshop/codingWithMP/grid_search/images_trial'
image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
image_files.sort()  # Sort the filenames to ensure a consistent order
# print(f"Total images found: {len(image_files)}")
# print("First few image filenames:", image_files[:10])


# Extract columns for image paths and labels

true_labels = df['Smiling'].values  # Update 'label' based on the column name in df
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
    image_np = np.array(image)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np)
    results = detector.detect(mp_image)
    if results.face_landmarks:
        # Extract blendshape values
        blendshape_values = [blendshape.score for blendshape in results.face_blendshapes[0] if blendshape.category_name in target_blendshapes]
        print(blendshape_values)
        return blendshape_values
    return None

# Process each image in the dataset
all_blendshapes_data = []

print(len(image_files))
for image_file in image_files:
    image_path = os.path.join(folder_path, image_file)
    if not os.path.exists(image_path):
        print(f"File does not exist: {image_path}")
        continue

    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Failed to load image: {image_path} with error: {e}")
        continue
    blendshapes = extract_blendshapes(image)
    if blendshapes is not None:
        all_blendshapes_data.append(blendshapes)

# Convert all blendshapes data to a NumPy array
blendshapes_array = np.array(all_blendshapes_data)



print(f"Blendshapes array shape: {blendshapes_array.shape}")

# Check if blendshapes_array is empty
if blendshapes_array.size == 0:
    print("No blendshapes data extracted.")

# Convert the blendshapes array to a DataFrame
blendshapes_df = pd.DataFrame(blendshapes_array)



# Prepare data for GridSearch
X = blendshapes_array
y = true_labels

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
