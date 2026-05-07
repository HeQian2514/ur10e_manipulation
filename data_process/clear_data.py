import os
import pickle
import numpy as np
from tqdm import tqdm


def clean_demo_data(demo_folder, pose_threshold=1e-4):
    """
    Clean up demo data. Delete continuous frames with identical pose and gripper, and overwrite original files.
    
    Args:
        demo_folder: Folder path containing demo files
        pose_threshold: Threshold to determine if two poses are identical, default is 1e-6
    """
    # Get all demo files
    demo_files = [f for f in os.listdir(demo_folder) if f.startswith("demo_") and f.endswith(".pkl")]
    
    # Count total data amount
    total_frames_before = 0
    total_frames_after = 0
    
    print(f"Found {len(demo_files)} demo files")
    
    # Process each demo file
    for demo_file in tqdm(demo_files, desc="Processing demo files"):
        print(f"File {demo_file}")
        demo_path = os.path.join(demo_folder, demo_file)
        
        # Load demo data
        with open(demo_path, 'rb') as f:
            demo_data = pickle.load(f)
        
        # Data consistency check
        assert all(len(demo_data[key]) == len(demo_data['pose']) for key in demo_data.keys()), \
            f"Data lengths in demo file {demo_file} are inconsistent"
        
        # Record original frame count
        original_length = len(demo_data['pose'])
        total_frames_before += original_length
        
        # Create a mask for kept frames, initially all True
        keep_mask = np.ones(original_length, dtype=bool)
        
        # Iterate and mark frames to delete
        for i in range(1, original_length):
            # If current frame has identical pose and gripper as the previous one, mark for deletion
            pose_same = np.allclose(demo_data['pose'][i], demo_data['pose'][i-1], atol=pose_threshold)
            gripper_same = demo_data['gripper'][i] == demo_data['gripper'][i-1]
            
            if pose_same and gripper_same:
                keep_mask[i] = False
        
        # Create cleaned data
        cleaned_data = {}
        for key in demo_data.keys():
            cleaned_data[key] = [demo_data[key][i] for i in range(original_length) if keep_mask[i]]

        # Record cleaned frame count
        cleaned_length = len(cleaned_data['pose'])
        total_frames_after += cleaned_length
        
        # Save cleaned data, directly overwriting original file
        with open(demo_path, 'wb') as f:
            pickle.dump(cleaned_data, f)
            
        print(f"File {demo_file}: Original frames {original_length} -> Cleaned frames {cleaned_length}")
    
    # Print overall statistics
    print(f"\nCleaning complete! Total frames: {total_frames_before} -> {total_frames_after}")
    print(f"Deleted {total_frames_before - total_frames_after} frames ({(total_frames_before - total_frames_after) / total_frames_before * 100:.2f}%)")

if __name__ == "__main__":
    # Directly specify data folder path
    demo_folder = 'data/expert_demos'
    
    # You can modify this path as needed
    # demo_folder = input("Please enter demo data folder path: ")
    
    clean_demo_data(demo_folder)