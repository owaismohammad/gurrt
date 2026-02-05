import cv2
import moviepy as mp
INPUT_VIDEO = '../This Integral Breaks Math.mp4'
def frame_sampling(path: str):
    cap= cv2.VideoCapture(path)
    out_path = '../outputs/output.mp4'
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename=out_path, fourcc=fourcc, fps= 1.0, frameSize=(width, height), isColor= True)
    frame_no = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_no % int(round(fps)) == 0:
            out.write(frame)
        frame_no +=1
    
    return out_path

def audio_extraction(path: str):
    audio_file = '../outputs/audio_file.mp3'
    video = mp.VideoFileClip(path)
    audio = video.audio
    
    audio.write_audiofile(audio_file)
    
    audio.close()
    video.close()
    
    return audio_file
b = frame_sampling(INPUT_VIDEO)

c = audio_extraction(INPUT_VIDEO)