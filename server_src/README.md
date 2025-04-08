# nohands

### Dependencies
```bash
pip install uv

# Create a new virtual environment
uv venv .venv

# Activate the virtual environment
# On Windows:
# .venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

Now install dependencies:
```bash
# Install dependencies from requirements.txt
uv pip install -r requirements.txt
```

If you install new dependencies, update requirements.txt with `pip freeze`. For your reference, what I've done as initial setup is this:

```bash
# Install all required packages using uv
uv pip install numpy scipy sounddevice transformers fastapi uvicorn soundfile requests torch

# Generate requirements.txt file
uv pip freeze > requirements.txt
```

### Running the code
Simple demo (read output and wait until it tells you it records audio)
```bash
python record_and_transcribe_demo.py
```

Server
```bash
python main.py
```