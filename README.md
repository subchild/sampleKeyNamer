# Sample Folder Finalizer

Finalizes audio samples for distribution by:
1. Normalizing audio to -1dB peak (preserves sample rate and bit depth)
2. Detecting the musical key and appending it to the filename
3. Detecting BPM for files whose names start with `loop` and appending it to the filename

## Usage

```bash
./finalize_samples.sh /path/to/folder
# or for a single file:
./finalize_samples.sh audio.wav
```

## Requirements

- ffmpeg (install with `brew install ffmpeg`)
- Python 3 with librosa and numpy (`pip install librosa numpy`)

## Example

```bash
./finalize_samples.sh my_samples/
```

Will normalize and rename:
- `synth.wav` → `synth-Cm.wav` (C minor, normalized to -1dB)
- `loop-bass.wav` → `loop-bass-G-131bpm.wav` (G major, 131 BPM, normalized to -1dB)
