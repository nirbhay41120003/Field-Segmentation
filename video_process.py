import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
import os
import torch

def process_video(input_video_path, output_video_path, model_path, conf_threshold=0.2):
    """
    Process a video file for segmentation
    Args:
        input_video_path: Path to input video file
        output_video_path: Path to save the processed video
        model_path: Path to the YOLO model weights
        conf_threshold: Confidence threshold for predictions
    """
    # Load the model with custom settings
    torch.set_warn_always(False)  # Suppress warnings
    
    # Load the model
    model = YOLO(model_path)
    
    # Open the video file
    cap = cv2.VideoCapture(input_video_path)
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create video writer object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    # Process each frame
    with tqdm(total=total_frames, desc="Processing video") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Perform prediction with segmentation
            results = model(frame, conf=conf_threshold, verbose=False)
            
            # Get the plotted frame with segmentation
            plotted_frame = results[0].plot()
            
            # Convert from RGB to BGR for OpenCV
            plotted_frame = cv2.cvtColor(plotted_frame, cv2.COLOR_RGB2BGR)
            
            # Write the frame to output video
            out.write(plotted_frame)
            
            # Update progress bar
            pbar.update(1)
    
    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"Video processing complete. Output saved to: {output_video_path}")

def extract_frames_for_analysis(video_path, output_dir, frame_interval=30):
    """
    Extract frames from video for detailed analysis
    Args:
        video_path: Path to input video file
        output_dir: Directory to save extracted frames
        frame_interval: Extract every nth frame
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    frame_count = 0
    frame_number = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_interval == 0:
            # Save frame
            frame_path = os.path.join(output_dir, f'frame_{frame_number:04d}.jpg')
            cv2.imwrite(frame_path, frame)
            frame_number += 1
            
        frame_count += 1
    
    cap.release()
    print(f"Extracted {frame_number} frames to {output_dir}")

def analyze_video_frames(frames_dir, model_path, output_csv_path):
    """
    Analyze extracted frames and save results to CSV
    Args:
        frames_dir: Directory containing extracted frames
        model_path: Path to the YOLO model weights
        output_csv_path: Path to save the analysis results
    """
    import pandas as pd
    from PIL import Image
    
    # Load the model
    model = YOLO(model_path)
    
    # Initialize lists to store results
    results = []
    
    # Process each frame
    for frame_name in tqdm(os.listdir(frames_dir), desc="Analyzing frames"):
        if not frame_name.endswith(('.jpg', '.jpeg', '.png')):
            continue
            
        frame_path = os.path.join(frames_dir, frame_name)
        frame = Image.open(frame_path)
        
        # Perform prediction
        pred = model(frame, conf=0.2, verbose=False)
        
        # Extract results
        for r in pred:
            if hasattr(r, 'masks'):
                for mask in r.masks:
                    mask_data = mask.data.cpu().numpy()
                    results.append({
                        'frame': frame_name,
                        'mask_data': mask_data.tolist()
                    })
    
    # Convert results to DataFrame and save to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv_path, index=False)
    print(f"Analysis results saved to: {output_csv_path}")

if __name__ == "__main__":
    # Example usage
    model_path = "best.pt"
    input_video = "path/to/input/video.mp4"
    output_video = "path/to/output/video.mp4"
    frames_dir = "path/to/output/frames"
    output_csv = "path/to/output/analysis.csv"
    
    # Process video
    process_video(input_video, output_video, model_path)
    
    # Extract frames for analysis
    extract_frames_for_analysis(input_video, frames_dir)
    
    # Analyze frames
    analyze_video_frames(frames_dir, model_path, output_csv) 