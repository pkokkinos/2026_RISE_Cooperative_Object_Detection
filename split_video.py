import cv2
import os
import math

def split_video(input_path, num_parts, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    cap = cv2.VideoCapture(input_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    frames_per_part = math.ceil(total_frames / num_parts)
    print(f"Splitting {input_path} into {num_parts} parts ({frames_per_part} frames each)...")
    
    for i in range(num_parts):
        output_path = os.path.join(output_folder, f"drone_{i}.mp4")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for _ in range(frames_per_part):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        
        out.release()
        print(f"Created {output_path}")
        
    cap.release()

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python split_video.py <input_file> <num_parts>")
    else:
        split_video(sys.argv[1], int(sys.argv[2]), "drone_feeds")
