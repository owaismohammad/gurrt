import cv2
from utils.utils import frame_sampling

path = './Example-Video.mp4'

sampled_video_path = frame_sampling(path=path)