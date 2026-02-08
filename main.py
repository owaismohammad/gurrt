import cv2
from utils.utils import uniform_frame_sampling, mapping_frame_with_timestamp, using_clip
from scripts.sampling_frames import uniform_frame_sampling
path = './This Integral Breaks Math.mp4'

# sampled_video_path, sampled_timestamp = uniform_frame_sampling(path=path)

# w = mapping_frame_with_timestamp(video_path='./outputs/output.mp4',
#                                  timestamp=sampled_timestamp)

# f = using_clip(w, "person explains limit of x to the power h minus 1  divided by h")
# print(f)

w = uniform_frame_sampling(path=path,
                           prompt="person explains limit of x to the power h minus 1  divided by h")
print(w)