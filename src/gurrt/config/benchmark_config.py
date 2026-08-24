from pathlib import Path
# VIDEO_PATH=Path(rf"C:\Users\fareh\Downloads\gurrt_benchmark\Video_ID_20\Video_ID_20.mp4")
# OUTPUT_PATH=Path(rf"C:\Users\fareh\Downloads\gurrt_benchmark\Video_ID_20")
# MANIFEST_PATH=Path(r"C:\Users\fareh\Downloads\gurrt_benchmark\Video_ID_20\manifest.json")
# CAPTION_PATH=Path(r"C:\Users\fareh\Downloads\gurrt_benchmark\Video_ID_20\captions.json")
# DEFAULT_OUT = Path(r"C:\Users\fareh\Downloads\gurrt_benchmark\Video_ID_20") / "prompts.json"

ID = 16
VIDEO_PATH=Path(rf"/workspace/gurrt/experiment/Video_ID_{ID}.mp4")
OUTPUT_PATH=Path(rf"/workspace/llama_bench/Video_ID_{ID}")
MANIFEST_PATH = Path(rf"/workspace/llama_bench/Video_ID_{ID}/manifest.json")
CAPTION_PATH = Path(rf"/workspace/llama_bench/Video_ID_{ID}/captions.json")
DEFAULT_OUT = Path(rf"/workspace/llama_bench/Video_ID_{ID}") / "prompts.json"
RESPONSE_PATH = Path(rf"/workspace/llama_bench/Video_ID_{ID}") / "response.csv"

QUESTIONS = [
    "How does the concept of rectangles relate to the foundational understanding of integrals?"
]

