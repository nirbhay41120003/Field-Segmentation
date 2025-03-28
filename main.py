from video_process import process_video, extract_frames_for_analysis, analyze_video_frames

# Define the input and output paths
# Define paths for video processing
input_video = r"C:\Users\nirbh\OneDrive\Desktop\det\input_videos\test_video.mp4"  # Update this path to your video file
output_video = r"C:\Users\nirbh\OneDrive\Desktop\det\processed_video\processed_video.mp4"

model_path = r"C:\Users\nirbh\OneDrive\Desktop\det\last.pt"

# Process the video
process_video(
    input_video_path=input_video,
    output_video_path=output_video,
    model_path=model_path,
    conf_threshold=0.2
)
