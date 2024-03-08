# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 21:32:08 2024

@author: S.M.H Mousavi
"""

# MPJPE (Mean Per Joint Position Error) = The lower the better
# Average Euclidean distance between the predicted and the true positions of the joints

import os
import numpy as np
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
            for channel in ['Xposition', 'Yposition', 'Zposition']:
                if channel in mocap.joint_channels(joint_name):
                    channels.append(channel)
            if channels:
                frame.extend(mocap.frame_joint_channels(frame_number, joint_name, channels))
        motion_data.append(frame)
    return motion_data

def calculate_mpjpe(motion_data1, motion_data2):
    mpjpe_values = []
    for frame1, frame2 in zip(motion_data1, motion_data2):
        joint_errors = []
        for i in range(0, len(frame1), 3):
            pos1 = np.array(frame1[i:i+3])
            pos2 = np.array(frame2[i:i+3])
            joint_errors.append(np.linalg.norm(pos1 - pos2))
        frame_error = np.mean(joint_errors)
        mpjpe_values.append(frame_error)
    return np.mean(mpjpe_values)

def load_folder(folder_path):
    all_files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.bvh')])
    all_motion_data = []
    for i, file in enumerate(all_files):
        print(f"Loading file {i+1}/{len(all_files)} from {folder_path}: {file}")
        all_motion_data.append(load_bvh(file))
    return all_motion_data

# Load BVH files from folders
original_folder = 'Original'
synthetic_folder = 'Synthetic'

print("Loading original motion data...")
original_motion_data = load_folder(original_folder)
print("Loading synthetic motion data...")
synthetic_motion_data = load_folder(synthetic_folder)

# Calculate MPJPE for each pair of original and synthetic files
print("Calculating MPJPE for each pair...")
mpjpe_scores = []
for i, (orig_data, synth_data) in enumerate(zip(original_motion_data, synthetic_motion_data)):
    mpjpe_score = calculate_mpjpe(orig_data, synth_data)
    mpjpe_scores.append(mpjpe_score)
    print(f"Calculated MPJPE for pair {i+1}: {mpjpe_score}")

# Calculate the average MPJPE across all pairs
average_mpjpe = np.mean(mpjpe_scores)
print("Average MPJPE across all pairs:", average_mpjpe)

end_time = time.time()
print(f"The code took {end_time - start_time} seconds to run")