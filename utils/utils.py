import cv2
import moviepy as mp
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def mapping_frame_with_timestamp(video_path: str, timestamp: list):
    
    cap = cv2.VideoCapture(video_path)
    frame_list = []
    
    while cap.isOpened():
        ret, frame= cap.read()
        if not ret:
            break
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_list.append(frame)
        cv2.waitKey(0)
    dic = {}
    for i,j in enumerate(timestamp):
        dic[j] = frame_list[i]
        
    return dic

def using_clip(frame_timestamp: dict, prompt: str):
    dic = {}
    for i,j in frame_timestamp.items():
        inputs = processor(text = prompt, images = j, return_tensors = 'pt', padding = True)
        outputs = model(**inputs)
    
        # similarity of each text description to the image
        logits_per_image = outputs.logits_per_image

        # convert to normalized probabilities
        probs = logits_per_image.softmax(dim=1)

        dic[i] = probs
    return dic
def uniform_frame_sampling(path: str):
    cap= cv2.VideoCapture(path)
    out_path = '../outputs/output.mp4'
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename=out_path,
                          fourcc=fourcc,
                          fps= 1.0,
                          frameSize=(width, height),
                          isColor= True)
    frame_no = 0
    sampled_timestamps = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_no % int(round(fps)) == 0:
            timestamp_sec = cap.get(cv2.CAP_PROP_POS_MSEC) 
            sampled_timestamps.append(timestamp_sec)
            out.write(frame)
        frame_no +=1
        
    return out_path, sampled_timestamps

def audio_extraction(path: str):
    audio_file = '../outputs/audio_file.mp3'
    video = mp.VideoFileClip(path)
    audio = video.audio
    
    audio.write_audiofile(audio_file)
    
    audio.close()
    video.close()
    
    return audio_file
