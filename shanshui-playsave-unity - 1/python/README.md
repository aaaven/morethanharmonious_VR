# More than Harmonious with VR Project

## Requirements

### Hardware
- NVIDIA GPU (RTX 30 series or newer recommended)
- 8GB+ VRAM
- 16GB+ system RAM

### Software
- Python 3.8+
- CUDA 11.7+
- cuDNN 8.0+

## Installation

1. Clone the repository
```bash
git clone <repository-url>
cd <project-directory>
```

## Create virtual environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

## Install dependencies

```bash
pip install -r requirements.txt
Dependencies
text
torch==2.0.0+cu117
torchvision==0.15.1+cu117
opencv-python==4.7.0.72
numpy==1.24.3
Pillow==9.5.0
moviepy==1.0.3
keyboard==0.13.5
diffusers==0.16.0
transformers==4.29.2
```

## Running the System
Start the main server:

```bash
python main.py
```

### The system listens on these ports:

- 12345: Main image processing

- 12346: Client connections

- 8080: HMD status monitoring

- 13000: Prompt index service

## Usage
- Ensure all dependencies are installed

- Run the main program

- Connect client devices (e.g., VR headset)

- System processes inputs in real-time

- Press R to reset the system

- Generated videos save to history/ directory