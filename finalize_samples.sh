#!/usr/bin/env bash
# Finalize audio samples: normalize to -1dB and add key to filename
# Usage: ./finalize_samples.sh /path/to/folder OR ./finalize_samples.sh file.wav

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if a path argument was provided
if [[ $# -eq 0 ]]; then
    echo "Usage: $0 /path/to/folder OR $0 file.wav"
    echo "Normalizes audio to -1dB and adds key to filename for all WAV files."
    exit 1
fi

PATH_ARG="$1"

# Check if the path exists
if [[ ! -e "$PATH_ARG" ]]; then
    echo "Error: Path '$PATH_ARG' does not exist."
    exit 1
fi

# Check if ffmpeg is installed
if ! command -v ffmpeg &> /dev/null; then
    echo "Error: ffmpeg is not installed. Install it with: brew install ffmpeg"
    exit 1
fi

# Check if Python script exists
if [[ ! -f "$SCRIPT_DIR/detect_and_rename_keys.py" ]]; then
    echo "Error: detect_and_rename_keys.py not found in $SCRIPT_DIR"
    exit 1
fi

echo "=========================================="
echo "Step 1: Normalizing audio to -1dB peak"
echo "=========================================="

# If it's a file, process just that file; if it's a directory, process recursively
if [[ -f "$PATH_ARG" ]]; then
    # Single file mode
    FILES_TO_PROCESS=("$PATH_ARG")
else
    # Directory mode - find all WAV files recursively
    FILES_TO_PROCESS=()
    while IFS= read -r -d '' wav_file; do
        FILES_TO_PROCESS+=("$wav_file")
    done < <(find "$PATH_ARG" -name "*.wav" -print0)
fi

# Counter for processed files
processed=0
failed=0

# Process each file
for wav_file in "${FILES_TO_PROCESS[@]}"; do
    echo "Processing: $wav_file"
    
    # Create a temporary file for the normalized audio
    temp_file="${wav_file}.normalized.wav"
    
    # Get original sample rate and bit depth
    original_sr=$(ffprobe -v error -select_streams a:0 -show_entries stream=sample_rate -of default=noprint_wrappers=1:nokey=1 "$wav_file")
    original_bd=$(ffprobe -v error -select_streams a:0 -show_entries stream=bits_per_raw_sample -of default=noprint_wrappers=1:nokey=1 "$wav_file")
    
    # Determine PCM codec based on bit depth
    if [[ "$original_bd" == "24" ]]; then
        pcm_codec="pcm_s24le"
    else
        pcm_codec="pcm_s16le"  # Default to 16-bit
    fi
    
    # Normalize to -1dB peak while preserving original sample rate and bit depth
    # First pass: detect peak level, second pass: scale to -1dB
    peak_db=$(ffmpeg -i "$wav_file" -af "volumedetect" -f null - 2>&1 | grep max_volume | awk '{print $5}')
    gain_db=$(awk "BEGIN {print -1 - $peak_db}")
    
    if ffmpeg -i "$wav_file" -c:a "$pcm_codec" -ar "$original_sr" -af "volume=${gain_db}dB" -y "$temp_file" 2>/dev/null; then
        # Replace original with normalized version
        mv "$temp_file" "$wav_file"
        echo "  ✓ Normalized successfully"
        ((processed++))
    else
        echo "  ✗ Failed to normalize"
        rm -f "$temp_file"
        ((failed++))
    fi
done

echo ""
echo "=========================================="
echo "Normalization complete!"
echo "Processed: $processed file(s)"
if [[ $failed -gt 0 ]]; then
    echo "Failed: $failed file(s)"
fi
echo "=========================================="
echo ""

# Step 2: Detect keys and rename files
echo "=========================================="
echo "Step 2: Detecting keys and renaming files"
echo "=========================================="

python3 "$SCRIPT_DIR/detect_and_rename_keys.py" "$PATH_ARG"

echo ""
echo "=========================================="
echo "All done! Samples finalized."
echo "=========================================="

if [[ $failed -gt 0 ]]; then
    exit 1
fi
