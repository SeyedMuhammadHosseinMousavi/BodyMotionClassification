import os
import numpy as np
from scipy.spatial.distance import pdist, squareform
from bvh import Bvh
import time
start_time = time.time()
def load_bvh(file_path):
    with open(file_path) as f:
        mocap = Bvh(f.read())

    joint_names = [
        "Hips", "LHipJoint", "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase",
        "RHipJoint", "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase",
        "LowerBack", "Spine", "Spine1", "Neck", "Neck1", "Head", "LeftShoulder",
        "LeftArm", "LeftForeArm", "LeftHand", "LeftFingerBase", "LeftHandIndex1",
        "LThumb", "RightShoulder", "RightArm", "RightForeArm", "RightHand",
        "RightFingerBase", "RightHandIndex1", "RThumb"
    ]

    motion_data = []
    for frame_number in range(mocap.nframes):
        frame = []
        for joint_name in joint_names:
            channels = []
            for channel in ['Xposition', 'Yposition', 'Zposition', 'Xrotation', 'Yrotation', 'Zrotation']:
                if channel in mocap.joint_channels(joint_name):
                    channels.append(channel)
            if channels:
                frame.extend(mocap.frame_joint_channels(frame_number, joint_name, channels))
        motion_data.append(frame)
    return motion_data

def normalize_frame_count(motion_data_list, target_frame_count):
    normalized_data = []
    for data in motion_data_list:
        if len(data) > target_frame_count:
            normalized_data.append(data[:target_frame_count])
        else:
            last_frame = data[-1]
            frames_to_add = target_frame_count - len(data)
            normalized_data.append(data + [last_frame for _ in range(frames_to_add)])
    return normalized_data

def calculate_diversity(motion_data):
    flattened_data = [np.array(frame).flatten() for data in motion_data for frame in data]
    distances = pdist(flattened_data, metric='euclidean')
    return np.mean(distances)

def load_folder(folder_path):
    all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.bvh')]
    all_motion_data = []
    for i, file in enumerate(all_files):
        print(f"Loading file {i+1}/{len(all_files)}: {file}")
        all_motion_data.append(load_bvh(file))
    return all_motion_data

# Load BVH files from folders
original_folder = 'Original'
synthetic_folder = 'Synthetic'

print("Loading original motion data...")
original_motion_data = load_folder(original_folder)
print("Loading synthetic motion data...")
synthetic_motion_data = load_folder(synthetic_folder)

# Normalize frame counts
print("Normalizing frame counts...")
min_frame_count = min([len(data) for data in original_motion_data + synthetic_motion_data])
original_motion_data_normalized = normalize_frame_count(original_motion_data, min_frame_count)
synthetic_motion_data_normalized = normalize_frame_count(synthetic_motion_data, min_frame_count)

# Calculate diversity
print("Calculating diversity within original samples...")
diversity_original = calculate_diversity(original_motion_data_normalized)
print("Calculating diversity within synthetic samples...")
diversity_synthetic = calculate_diversity(synthetic_motion_data_normalized)
print("Calculating diversity between original and synthetic samples...")
diversity_between = calculate_diversity(original_motion_data_normalized + synthetic_motion_data_normalized)

print("Diversity within Originals:", diversity_original)
print("Diversity within Synthetics:", diversity_synthetic)
print("Diversity between Originals and Synthetics:", diversity_between)

end_time = time.time()
print(f"The code took {end_time - start_time} seconds to run")