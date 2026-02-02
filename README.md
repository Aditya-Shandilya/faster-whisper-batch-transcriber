# Faster-Whisper Batch Transcriber

A command-line tool for efficiently batch-transcribing audio files using the `faster-whisper` library. Optimized for processing multiple files such as lectures, interviews, podcasts, or meetings with high performance and accuracy.

## ✨ Features

- **⚡ High Performance**: Uses `faster-whisper`, a reimplementation of OpenAI's Whisper that is up to 4x faster and uses 2x less memory than the original
- **📦 Batch Processing**: Transcribe entire folders of audio files with a single command
- **🎯 Smart Formatting**: Automatically adds paragraph breaks based on silence detection (2+ seconds)
- **🔍 Voice Activity Detection (VAD)**: Integrates Silero VAD to filter out non-speech segments for cleaner transcripts
- **🌍 Automatic Language Detection**: Detects spoken language automatically with confidence scores
- **⚙️ Flexible Configuration**: Customizable model size, device (CPU/GPU), quantization, and beam size
- **📊 Progress Tracking**: Real-time progress bars for each transcription job
- **🎵 Broad Format Support**: Works with `.mp3`, `.wav`, `.m4a`, `.flac`, `.ogg`, `.aac`, and more
- **🖥️ Device Flexibility**: Run on CPU or GPU (CUDA) with INT8 or FP16 quantization

## 📋 Requirements

- Python 3.8 or higher
- For GPU acceleration: NVIDIA GPU with CUDA 12 and cuDNN 9 (or CUDA 11/cuDNN 8 with ctranslate2==3.24.0)

> **Note:** Unlike OpenAI's original Whisper, `faster-whisper` does not require FFmpeg to be installed separately. Audio decoding is handled by PyAV, which bundles FFmpeg libraries internally.

## 🚀 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Aditya-Shandilya/faster-whisper-batch-transcriber.git
   cd faster-whisper-batch-transcriber
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   > **Note for GPU users:** If using CUDA 11 or older versions, you may need to downgrade ctranslate2:
   > ```bash
   > pip install --force-reinstall ctranslate2==3.24.0
   > ```

3. **Create your audio folder:**
   ```bash
   mkdir audios
   ```
   Place your audio files in the `audios` folder before running the script.

## 📖 Usage

### Quick Start (CPU)

The simplest way to get started - transcribes all audio files in the `audios` folder using the medium model on CPU:

```bash
python transcribe.py
```

Transcripts will be saved to the `transcripts` folder (created automatically).

### GPU Usage (Recommended for Speed)

For best performance with a CUDA-capable GPU:

```bash
python transcribe.py --model turbo --device cuda --compute_type float16 --beam_size 5
```

Or for maximum accuracy:

```bash
python transcribe.py --model large-v3 --device cuda --compute_type float16 --beam_size 5
```

### Custom Directories

Specify custom input and output folders:

```bash
python transcribe.py --input_dir path/to/audio/files --output_dir path/to/save/transcripts
```

### Fast CPU Transcription

For faster CPU processing with slightly lower accuracy:

```bash
python transcribe.py --model small --beam_size 1
```

### High Accuracy Transcription

For maximum accuracy (slower):

```bash
python transcribe.py --model large-v3 --device cuda --compute_type float16 --beam_size 10
```

For great accuracy with much better speed (recommended):

```bash
python transcribe.py --model turbo --device cuda --compute_type float16 --beam_size 5
```

## ⚙️ Command-Line Arguments

| Argument         | Default       | Description                                                                                      |
|------------------|---------------|--------------------------------------------------------------------------------------------------|
| `--input_dir`    | `audios`      | Folder containing audio files to transcribe                                                      |
| `--output_dir`   | `transcripts` | Folder where output `.txt` files will be saved                                                   |
| `--model`        | `medium`      | Whisper model size: `tiny`, `base`, `small`, `medium`, `large-v2`, `large-v3`, `turbo`, `distil-large-v3` |
| `--device`       | `cpu`         | Processing device: `cpu` or `cuda` (for NVIDIA GPUs)                                            |
| `--compute_type` | `int8`        | Quantization type: `int8`, `float16`, `int8_float16`                                            |
| `--beam_size`    | `3`           | Beam search size (1-10). Higher = more accurate but slower. Use 1 for fastest results          |

## 🎯 Model Selection Guide

| Model Size       | Speed    | Accuracy | VRAM (GPU) | RAM (CPU) | Best For                          |
|------------------|----------|----------|------------|-----------|-----------------------------------|
| `tiny`           | Fastest  | Low      | ~1 GB      | ~1 GB     | Quick drafts, testing             |
| `base`           | Fast     | Good     | ~1 GB      | ~1 GB     | General use, fast turnaround      |
| `small`          | Medium   | Better   | ~2 GB      | ~2 GB     | Balance of speed and quality      |
| `medium`         | Slower   | High     | ~5 GB      | ~5 GB     | **Recommended default**           |
| `large-v2`       | Slow     | Higher   | ~10 GB     | ~10 GB    | High accuracy needs               |
| `large-v3`       | Slow     | Highest  | ~10 GB     | ~10 GB    | Best quality, latest improvements |
| `turbo`          | Fast     | High     | ~6 GB      | ~6 GB     | Speed + accuracy balance*         |
| `distil-large-v3`| Medium   | High     | ~6 GB      | ~6 GB     | Distilled model, good speed/accuracy|

> **Tip**: `large-v3` offers 10-20% error reduction compared to `large-v2` and supports 128 Mel bins for better quality.

> **Note on turbo model**: The `turbo` model (also known as `large-v3-turbo`) is a pruned version of `large-v3` with 4 decoder layers instead of 32, making it significantly faster (~8x speed) with only minor accuracy loss. However, it **does not support translation** - it can only transcribe audio in its original language.

## 🔧 Advanced Features

### Voice Activity Detection (VAD)

The script automatically uses VAD filtering (`vad_filter=True`) to:
- Remove silent segments from transcription
- Improve accuracy by focusing on speech
- Reduce processing time
- Create cleaner, more readable transcripts

### Smart Paragraph Formatting

Transcripts automatically include paragraph breaks when silence exceeds 2 seconds, creating natural reading flow without manual formatting.

### Automatic Language Detection

The script detects the spoken language automatically and displays it with confidence probability for each file.

## 💡 Tips & Best Practices

1. **CPU Performance Warning**: If using CPU with `beam_size > 3`, expect slower performance. Consider using `--beam_size 1` or `--beam_size 2` for faster CPU transcription.

2. **GPU Memory**: If you encounter out-of-memory errors on GPU, try:
   - Using a smaller model size
   - Switching to `int8` or `int8_float16` compute type
   - Reducing the beam size

3. **Accuracy vs Speed**: 
   - For quick drafts: `tiny` or `base` model with `beam_size 1`
   - For production: `medium` or `large-v3` with `beam_size 5`

4. **Supported Audio Formats**: `.mp3`, `.wav`, `.m4a`, `.flac`, `.ogg`, `.aac`

## 📁 Project Structure

After setup, your directory should look like:

```
faster-whisper-batch-transcriber/
├── audios/                  # Place your audio files here
│   ├── lecture1.mp3
│   ├── meeting_notes.wav
│   └── podcast_episode.m4a
├── transcripts/             # Generated automatically - transcripts saved here
│   ├── lecture1.txt
│   ├── meeting_notes.txt
│   └── podcast_episode.txt
├── requirements.txt
├── transcribe.py
└── README.md
```

## 🔬 How It Works

1. **Model Loading**: Downloads and loads the specified Whisper model (cached locally for future use)
2. **Audio Detection**: Scans input directory for supported audio formats
3. **Transcription**: For each file:
   - Applies VAD filtering to remove silence
   - Detects language automatically
   - Transcribes with specified beam size
   - Formats output with smart paragraph breaks
4. **Output**: Saves clean, formatted transcripts as `.txt` files

## 🐛 Troubleshooting

**"Input directory not found"**
- Ensure the `audios` folder exists or specify a valid path with `--input_dir`

**"Failed to load model"**
- Check your internet connection (models download on first use)
- Verify CUDA installation if using `--device cuda`
- Try a smaller model if encountering memory issues

**Slow CPU performance**
- Use smaller models (`tiny`, `base`, `small`)
- Reduce beam size to 1 or 2
- Consider using GPU if available

**CUDA/cuDNN errors**
- Ensure CUDA 12 + cuDNN 9 are installed (or downgrade ctranslate2 for CUDA 11)
- Verify GPU drivers are up to date

## 📄 License

This project uses the `faster-whisper` library which is based on OpenAI's Whisper model. Please refer to the respective licenses for usage terms.

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## 🙏 Acknowledgments

- Built with [faster-whisper](https://github.com/SYSTRAN/faster-whisper) by SYSTRAN
- Based on OpenAI's [Whisper](https://github.com/openai/whisper) model
- Uses [CTranslate2](https://github.com/OpenNMT/CTranslate2) for optimized inference

---

Made with ❤️ for the transcription community