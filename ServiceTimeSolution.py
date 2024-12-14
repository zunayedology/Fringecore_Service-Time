import cv2
import numpy as np
from tqdm import tqdm

def get_pixel_color(frame):
    return frame[404, 1034] # A pixel where the customer stands

def is_darker(prev_frame, frame, threshold=50):
    prev_color = np.array(get_pixel_color(prev_frame))
    current_color = np.array(get_pixel_color(frame))
    return np.mean(current_color) < np.mean(prev_color) - threshold

def process_video(video):
    cap = cv2.VideoCapture(video)
    total_duration = 0
    frame_count = 0

    prev_frame = None
    color_change_start = None

    for _ in tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if prev_frame is None:
            prev_frame = frame
            continue

        if is_darker(prev_frame, frame):
            if color_change_start is None:
                color_change_start = frame_count
            elif frame_count - color_change_start >= 30:
                total_duration += frame_count - color_change_start
                color_change_start = None

        prev_frame = frame

    avg = total_duration / (frame_count // 30) if frame_count > 60 else 0
    cap.release()
    return avg

if __name__ == "__main__":
    video_path = "fringestorez.mp4"
    avg_duration = process_video(video_path)
    print(f"Average: {avg_duration:.2f} seconds")

