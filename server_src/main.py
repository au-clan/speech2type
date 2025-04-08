from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from datasets import load_dataset
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch
import soundfile as sf
import numpy as np
import io


# Configure device and precision
# device = "cuda:0" if torch.cuda.is_available(
# ) else "mps" if torch.backends.mps.is_available() else "cpu"

# doesn't work for quantized model
do_compile = False

if do_compile:
    device = "cpu"
    torch_dtype = torch.float32
    attn_implementation = "sdpa"
else:
    device = "mps"
    torch_dtype = torch.float16
    attn_implementation = "sdpa"

# Update to use Whisper Turbo model
model_id = "openai/whisper-large-v3-turbo"
# model_id = "Yehor/whisper-large-v3-turbo-quantized-uk"

# Load model
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True,
    attn_implementation=attn_implementation, use_safetensors=True)
model.to(device)

if do_compile:
    # Enable static cache and compile the forward pass
    model.generation_config.cache_implementation = "static"
    model.generation_config.max_new_tokens = 400
    model.forward = torch.compile(
        model.forward, mode="reduce-overhead", fullgraph=True)

# Load processor
processor = AutoProcessor.from_pretrained(model_id)

# Create pipeline
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

# Run a test
dataset = load_dataset("distil-whisper/librispeech_long",
                       "clean", split="validation")
sample = dataset[0]["audio"]
result = pipe(sample, return_timestamps=True,
              generate_kwargs={"language": "english"})
print(result["text"])

# Set up server to provide hosted transcription
app = FastAPI()


@app.post("/speechtotext")
async def speech_to_text(audio_file: UploadFile = File(...)):
    try:
        # Read audio file
        contents = await audio_file.read()
        bytes_io = io.BytesIO(contents)
        audio, sample_rate = sf.read(bytes_io, dtype="float32")
        assert sample_rate == 16000, "Whisper is trained on 16kHz audio. It might resample internally but let's not rely on that."

        # Ensure audio is in mono
        if len(audio.shape) == 2:
            audio = np.mean(audio, axis=1)

        # Process audio
        result = pipe(audio, return_timestamps=True,
                      generate_kwargs={"language": "english"})

        return JSONResponse(content={"transcription": result["text"]})

    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"An error occurred: {str(e)}"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
