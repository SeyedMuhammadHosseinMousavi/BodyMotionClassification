# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 22:18:09 2024
@author: S.M.H Mousavi
"""
'''
Dynamic Time Warping (DTW) represents a method for measuring the similarity 
between two motion sequences, even if they vary in timing or speed. 
'''
import os
import numpy as np
from bvh import Bvh
from dtaidistance import dtw

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

def calculate_dtw(motion_data1, motion_data2):
    motion_data1_np = np.array(motion_data1)
    motion_data2_np = np.array(motion_data2)
    
    total_dtw_distance = 0
    joint_count = motion_data1_np.shape[1] // 3

    for joint_idx in range(joint_count):
        for dim in range(3):  # X, Y, Z dimensions
            idx = joint_idx * 3 + dim
            dtw_distance = dtw.distance(motion_data1_np[:, idx], motion_data2_np[:, idx])
            total_dtw_distance += dtw_distance

    avg_dtw_distance = total_dtw_distance / (joint_count * 3)
    return avg_dtw_distance


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

# Calculate DTW for each pair of original and synthetic files
print("Calculating DTW for each pair...")
dtw_scores = []
for i, (orig_data, synth_data) in enumerate(zip(original_motion_data, synthetic_motion_data)):
    dtw_score = calculate_dtw(orig_data, synth_data)
    dtw_scores.append(dtw_score)
    print(f"Calculated DTW for pair {i+1}: {dtw_score}")

# Calculate the average DTW across all pairs
average_dtw = np.mean(dtw_scores)
print("Average DTW across all pairs:", average_dtw)

end_time = time.time()
print(f"The code took {end_time - start_time} seconds to run")