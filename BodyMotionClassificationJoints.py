# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 19:26:14 2024
@author: seyed.mousavi
"""

import numpy as np
import os
from bvh import Bvh
from scipy.interpolate import interp1d
import warnings
import pandas as pd
from scipy import stats
import scipy.stats
import time

# Record the start time
start_time = time.time()
# Suppress warnings
warnings.filterwarnings("ignore")
def read_bvh(filename):
    """Reads a BVH file and returns a Bvh object."""
    with open(filename) as f:
        mocap = Bvh(f.read())
    return mocap

def interpolate_frames(mocap, target_frame_count):
    """Interpolates BVH frames to match a target frame count."""
    original_frame_count = len(mocap.frames)
    original_time = np.linspace(0, 1, original_frame_count)
    target_time = np.linspace(0, 1, target_frame_count)
    interpolated_frames = []
    for frame in np.array(mocap.frames).T:
        interpolator = interp1d(original_time, frame.astype(float), kind='linear')
        interpolated_frame = interpolator(target_time)
        interpolated_frames.append(interpolated_frame)
    return np.array(interpolated_frames).T

def find_max_frames(folder_path):
    """Finds the maximum number of frames among all BVH files in a folder."""
    max_frames = 0
    for filename in os.listdir(folder_path):
        if filename.endswith('.bvh'):
            mocap = read_bvh(os.path.join(folder_path, filename))
            max_frames = max(max_frames, len(mocap.frames))
    return max_frames

def extract_motion_features(mocap):
    """Extracts motion features from interpolated BVH data."""
    motion_data = mocap.frames
    channels_per_joint = 3
    num_joints = len(motion_data[0]) // channels_per_joint

    motion_features = {
        f'joint_{i}': {
            'rotations': [], 'velocity': [], 'acceleration': [], 'range_of_motion': [],
            'average_rotation': [], 'spatial_path': [], 'harmonics': [], 'symmetry': [],
            'frequency_analysis': [], 'joint_distance': [], 'angular_velocity': [], 'angular_acceleration': []
        } for i in range(num_joints)
    }

    for frame in motion_data:
        for i in range(num_joints):
            start_idx = i * channels_per_joint
            end_idx = start_idx + channels_per_joint
            joint_data = [float(value) for value in frame[start_idx:end_idx]]
            motion_features[f'joint_{i}']['rotations'].append(joint_data)


    return motion_features

def process_bvh_files(folder_path, max_frames):
    """Processes each BVH file in the folder after interpolating to the same number of frames."""
    all_features = {}
    processed_files_count = 0
    for filename in os.listdir(folder_path):
        if filename.endswith('.bvh'):
            print(f"Processing file: {filename}")
            full_path = os.path.join(folder_path, filename)
            mocap = read_bvh(full_path)
            interpolated_frames = interpolate_frames(mocap, max_frames)
            mocap.frames = interpolated_frames
            # Assume extract_motion_features function is defined elsewhere
            motion_features = extract_motion_features(mocap)
            all_features[filename] = motion_features
            processed_files_count += 1
            print(f"Processed {processed_files_count} files.")
    return all_features

# Main workflow
train_folder_path = 'SmallDatasetTrain/'

# Find maximum frame size in the training data
max_frames_train = find_max_frames(train_folder_path)

# Process training and test BVH files using the training max frame size
all_features_train = process_bvh_files(train_folder_path, max_frames_train)
# Process the training features...

def recursive_flatten(input_item):
    """Recursively flattens nested lists or lists of lists into a flat list."""
    if isinstance(input_item, dict):
        return [sub_item for value in input_item.values() for sub_item in recursive_flatten(value)]
    elif isinstance(input_item, list):
        return [element for item in input_item for element in recursive_flatten(item)]
    else:
        return [input_item]

# Initialize an empty list to hold all flattened features from all samples
all_samples_flattened_features = []

# Iterate over all_features to flatten each sample's features and stack them
for filename, features in all_features_train.items():
    flattened_features = recursive_flatten(features)
    all_samples_flattened_features.append(flattened_features)

# Now, all_samples_flattened_features is a list where each item is the flattened feature list of a sample
# Optional: Convert the list of lists into a NumPy array for numerical processing
all_samples_flattened_features_array = np.array(all_samples_flattened_features, dtype=object)
array_of_lists = [list(row) for row in all_samples_flattened_features_array]
array_of_float64 = np.array(array_of_lists, dtype='float64')
flattened_data = array_of_float64

# Labels
C_Walk = [0] * 9
C_Run = [1] * 9
C_Jump = [2] * 9
C_Punch = [3] * 9
C_Kick = [4] * 9

# Concatenate the two lists
Labels = C_Walk + C_Run + C_Jump + C_Punch + C_Kick
Labels_int32= np.array(Labels, dtype=np.int32)

Xtr=flattened_data
Ytr=Labels_int32

# ------------------------------------------------------------------
# Train ------------------------------------------------------------------
# ------------------------------------------------------------------
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
# XGB ------------------------------------------------------------------
import xgboost as xgb
# XGBoost classifier with verbosity to show training progress
xgb_classifier = xgb.XGBClassifier(
    n_estimators=100,  # Number of gradient boosted trees. Equivalent to number of boosting rounds.
    max_depth=6,  # Maximum tree depth for base learners.
    learning_rate=0.4,  # Boosting learning rate (xgb's "eta")
    verbosity=1,  # Default is 1, showing warnings and training progress
    objective='binary:logistic',  # Specify the learning task and the corresponding learning objective or a custom objective function to use for training.
    random_state=42,  # Random number seed. 
    subsample=0.2,  # Subsample ratio of the training instances.
    colsample_bytree=0.4,  # Subsample ratio of columns when constructing each tree.
    gamma=0,  # Minimum loss reduction required to make a further partition on a leaf node of the tree.
    reg_alpha=0,  # L1 regularization term on weights
    reg_lambda=1  # L2 regularization term on weights
)  

xgb_classifier.fit(Xtr, Ytr, verbose=True)  # Additionally, setting verbose=True in fit method for progress
xgb_tr_acc = xgb_classifier.score(Xtr, Ytr)
# Print the training accuracy
print('XGBoost Train Accuracy is:')
print(xgb_tr_acc)

# Generate a classification report
ypredXGBtr = xgb_classifier.predict(Xtr)
class_reportXGBtr = classification_report(Ytr, ypredXGBtr)
print("XGBoost Train Classification Report:")
print(class_reportXGBtr)

# Generate the confusion matrix
cmxgb = confusion_matrix(Ytr, ypredXGBtr)
print("XGBoost Regression Train Confusion Matrix (Samples):")
print(cmxgb)
# Normalize the confusion matrix
cmxgb_normalized = cmxgb.astype('float') / cmxgb.sum(axis=1)[:, np.newaxis]
print("XGBoost Regression Train Confusion Matrix (Percentage):")
print(cmxgb_normalized)

#------------------------------------------------------
# Test
#------------------------------------------------------
test_folder_path = 'Test/'
all_features_test = process_bvh_files(test_folder_path, max_frames_train)

def recursive_flatten(input_item):
    """Recursively flattens nested lists or lists of lists into a flat list."""
    if isinstance(input_item, dict):
        return [sub_item for value in input_item.values() for sub_item in recursive_flatten(value)]
    elif isinstance(input_item, list):
        return [element for item in input_item for element in recursive_flatten(item)]
    else:
        return [input_item]

# Initialize an empty list to hold all flattened features from all samples
all_samples_flattened_features = []

# Iterate over all_features to flatten each sample's features and stack them
for filename, features in all_features_test.items():
    flattened_features = recursive_flatten(features)
    all_samples_flattened_features.append(flattened_features)

# Now, all_samples_flattened_features is a list where each item is the flattened feature list of a sample
# Optional: Convert the list of lists into a NumPy array for numerical processing
all_samples_flattened_features_array = np.array(all_samples_flattened_features, dtype=object)
array_of_lists = [list(row) for row in all_samples_flattened_features_array]
array_of_float64 = np.array(array_of_lists, dtype='float64')
flattened_data = array_of_float64


# Labels
C_Walk = [0] * 3
C_Run = [1] * 3
C_Jump = [2] * 3
C_Punch = [3] * 3
C_Kick = [4] * 3

# Concatenate the two lists
Labels = C_Walk + C_Run + C_Jump + C_Punch + C_Kick
Labels_int32= np.array(Labels, dtype=np.int32)
#FinalFeatures = [sorted(row) for row in FinalFeatures]
Xte=flattened_data
Yte=Labels_int32

xgb_te_acc = xgb_classifier.score(Xte, Yte)
print('XGBoost Train Accuracy is:')
print(xgb_tr_acc)
print('XGBoost Test Accuracy is:')
print(xgb_te_acc)
# Calculate accuracy on the test data
ypredXGB =  xgb_classifier.predict(Xte)
class_reportXGB = classification_report(Yte, ypredXGB)
# Print the classification report
print("XGBoost Classification Report:")
print(class_reportXGB)
# Generate the confusion matrix
cmxgbte = confusion_matrix(Yte, ypredXGB)
print("XGBoost Test Confusion Matrix (Samples):")
print(cmxgbte)
# Normalize the confusion matrix
cmxgbte_normalized = cmxgbte.astype('float') / cmxgbte.sum(axis=1)[:, np.newaxis]
print("XGBoost test Confusion Matrix (Percentage):")
print(cmxgbte_normalized)

# Violin Plot
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# Convert test data and predictions to a DataFrame for easier manipulation
df_test = pd.DataFrame(Xte)
df_test['True_Label'] = Yte
df_test['Predicted_Label'] = ypredXGB
# Select a feature to visualize. Here, choosing the first feature for demonstration.
feature_index = 0
feature_name = f"Feature_{feature_index}"
df_test[feature_name] = df_test[feature_index]
# Determine the number of unique classes for plotting
unique_classes = np.union1d(df_test['True_Label'].unique(), df_test['Predicted_Label'].unique())
# Setup the subplot grid
n_classes = len(unique_classes)
n_cols = 3  # Adjust based on preference
n_rows = n_classes // n_cols + (n_classes % n_cols > 0)
# Set bold fonts
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
# Prepare a list of colors for the violin plots
colors = sns.color_palette("hsv", n_classes)
plt.figure(figsize=(n_cols * 5, n_rows * 4))  # Adjust figure size as needed
for i, cls in enumerate(unique_classes, start=1):
    plt.subplot(n_rows, n_cols, i)
    # Filter the data for the current class
    current_class_data = df_test[df_test['Predicted_Label'] == cls]
    sns.violinplot(x='Predicted_Label', y=feature_name, data=current_class_data,
                   palette=[colors[i - 1]])  # Use a unique color for each plot
    plt.title(f'Class {cls}', fontweight='bold')
    plt.xlabel('Predicted Class', fontweight='bold')
    plt.ylabel('Feature Value', fontweight='bold')

plt.tight_layout()
plt.show()
# Reset font weights back to default to avoid affecting other plots
plt.rcParams['font.weight'] = 'normal'
plt.rcParams['axes.labelweight'] = 'normal'
plt.rcParams['axes.titleweight'] = 'normal'

# ROC Curve
from sklearn.preprocessing import label_binarize
import numpy as np
# Assuming 'Yte' is array of test labels and contains class labels as integers
n_classes = len(np.unique(Yte))  # Determine the number of unique classes
Yte_bin = label_binarize(Yte, classes=np.arange(n_classes))
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle
Y = label_binarize(Labels_int32, classes=np.arange(n_classes))
n_classes = Y.shape[1]
# Predict probabilities for ROC curve
y_score = xgb_classifier.predict_proba(Xte)
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(Yte_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
# Create a subplot for each class
n_cols = 2  # Adjust the number of columns as per your preference
n_rows = int(np.ceil(n_classes / n_cols))
fig, axs = plt.subplots(n_rows, n_cols, figsize=(10, n_rows * 5))
fig.subplots_adjust(hspace=0.4, wspace=0.4)
# Color and font settings
color = 'blue'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
# Plot ROC curves
for i in range(n_classes):
    row, col = divmod(i, n_cols)
    ax = axs[row, col] if n_classes > 1 else axs
    ax.plot(fpr[i], tpr[i], color=color, lw=2,
            label='ROC curve (area = {0:0.2f})'.format(roc_auc[i]))
    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Class {0}'.format(i), fontweight='bold')
    ax.legend(loc="lower right")
# Adjust for any empty subplots
if n_classes % n_cols != 0:
    for i in range(n_classes, n_rows * n_cols):
        row, col = divmod(i, n_cols)
        axs[row, col].axis('off')
plt.show()
# Reset to default font weights after plotting
plt.rcParams['font.weight'] = 'normal'
plt.rcParams['axes.labelweight'] = 'normal'
plt.rcParams['axes.titleweight'] = 'normal'

# precision and recall plot
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import numpy as np
# Assuming Yte, y_score are defined as in previous examples
# Binarize the output labels if not already done
n_classes = len(np.unique(Yte))
Yte_binarized = label_binarize(Yte, classes=np.arange(n_classes))
# Calculate precision and recall for each class
precision = dict()
recall = dict()
average_precision = dict()
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(Yte_binarized[:, i], y_score[:, i])
    average_precision[i] = average_precision_score(Yte_binarized[:, i], y_score[:, i])
# Plot Precision-Recall curve for each class
plt.figure(figsize=(7, n_classes*5))
for i in range(n_classes):
    plt.subplot(n_classes, 1, i+1)
    plt.step(recall[i], precision[i], color='b', alpha=0.2, where='post')
    plt.fill_between(recall[i], precision[i], step='post', alpha=0.2, color='b')
    plt.xlabel('Recall', fontweight='bold')
    plt.ylabel('Precision', fontweight='bold')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(f'Precision-Recall curve of Class {i} (AP={average_precision[i]:0.2f})', fontweight='bold')
plt.tight_layout()
plt.show()

# Confusion Matrix Heatmap
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cmxgbte, annot=True, fmt='g', cmap='Blues', cbar=True)
plt.xlabel('Predicted labels', fontweight='bold')
plt.ylabel('True labels', fontweight='bold')
plt.title('Confusion Matrix Heatmap', fontweight='bold')
plt.show()

# Cumulative Gain Curve
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
# Sort samples by their predicted probability
sorted_indices = np.argsort(y_score[:, 0])[::-1]
sorted_true_labels = Yte_binarized[:, 0][sorted_indices]
# Calculate the cumulative sum of true positives
cumulative_gains = np.cumsum(sorted_true_labels) / np.sum(sorted_true_labels)
cumulative_gains = np.insert(cumulative_gains, 0, 0)  # Insert 0 at the beginning
# Calculate the baseline (random model's performance)
baseline = np.linspace(0, 1, len(cumulative_gains))
# Plotting
plt.figure(figsize=(8, 6))
plt.plot(cumulative_gains, label='Cumulative Gains Curve', color='blue')
plt.plot(baseline, label='Baseline', linestyle='--', color='red')
plt.title('Cumulative Gains Curve', fontweight='bold')
plt.xlabel('Percentage of Sample', fontweight='bold')
plt.ylabel('Cumulative Gain', fontweight='bold')
plt.legend(loc='lower right')
plt.show()

# Print Acc
print ('XGBoost Train Accuracy is :')
print (xgb_tr_acc)
print ("---------------------------")
print ('XGBoost Test Original Accuracy is :')
print (xgb_te_acc)
print ("---------------------------")
print ("---------------------------")

# Record the end time
end_time = time.time()
# Calculate the runtime
runtime_seconds = end_time - start_time
runtime_minutes = runtime_seconds / 60
# Print the runtime
print(f"Runtime: {runtime_seconds:.2f} seconds ({runtime_minutes:.2f} minutes)")