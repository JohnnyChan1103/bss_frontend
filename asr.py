import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# 1. We first create the model
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=256,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)

def asr(wav_file):
    return pipe(wav_file, generate_kwargs={"language": "english"})["text"]

# result = pipe("/n/work1/juchen/BSS/scripts/fastmnmf_source1.wav")
# print(result["text"])
# result = pipe("/n/work1/juchen/BSS/scripts/fastmnmf_source2.wav")
# print(result["text"])
# result = pipe("/n/work1/juchen/BSS/scripts/489c0407.wav")
# print(result["text"])