import os
import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

def download_sample_video():
    """دانلود یک ویدیو نمونه برای تست"""
    video_path = os.path.join(DATA_DIR, "sample_video.mp4")
    
    if not os.path.exists(video_path):
        print("Creating synthetic video for demo...")
        # ساخت ویدیو مصنوعی با مربع‌های متحرک
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 30.0, (640, 480))
        
        for frame_idx in range(300):  # ۱۰ ثانیه ویدیو
            frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
            
            # دو شی متحرک
            x1 = 50 + frame_idx * 1
            y1 = 100
            cv2.rectangle(frame, (x1, y1), (x1 + 50, y1 + 50), (0, 255, 0), -1)
            
            x2 = 400 - frame_idx * 0.8
            y2 = 300
            cv2.rectangle(frame, (int(x2), y2), (int(x2 + 60), y2 + 60), (255, 0, 0), -1)
            
            out.write(frame)
        
        out.release()
    
    return video_path
