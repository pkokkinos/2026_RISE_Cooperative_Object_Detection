import cv2
import os

def rescale_video(input_path, output_path, scale_factor, target_fps):
    print(f"Rescaling {input_path} -> {output_path} (Scale: {scale_factor}, FPS: {target_fps})")
    cap = cv2.VideoCapture(input_path)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * scale_factor)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale_factor)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    
    # We maintain the original codec (MP4V or similar)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, target_fps, (width, height))
    
    frame_idx = 0
    # Logic to drop frames to match target_fps
    # e.g. if original is 30 and target is 15, we take every 2nd frame
    step = int(original_fps / target_fps) if original_fps > target_fps else 1
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx % step == 0:
            resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            out.write(resized)
            
        frame_idx += 1
        
    cap.release()
    out.release()
    print("Done.")

if __name__ == "__main__":
    INPUT = "input.mp4"
    
    # Medium Tier (720p equivalent for 1080p source)
    # 1080 * 0.66 ~= 720
    rescale_video(INPUT, "input_medium.mp4", 0.66, 15)
    
    # Low Tier (360p equivalent)
    # 1080 * 0.33 ~= 360
    rescale_video(INPUT, "input_low.mp4", 0.33, 10)
