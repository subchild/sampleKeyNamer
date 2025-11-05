#!/usr/bin/env python3
"""
Detects the musical key of WAV files and renames them to include the key.
Example: audio.wav -> audio_Cm.wav (for C minor)
"""

import os
import sys
import librosa
import numpy as np
from pathlib import Path

# Key mappings
KEYS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
KEY_PROFILES = {
    'major': [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
    'minor': [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
}


def detect_key(audio_path):
    """Detect the musical key of an audio file using the Krumhansl-Schmuckler algorithm."""
    # Load audio file
    y, sr = librosa.load(audio_path)
    
    # Get chromagram
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    
    # Average over time to get key profile
    chroma_vals = np.mean(chroma, axis=1)
    
    # Normalize
    chroma_vals = chroma_vals / np.sum(chroma_vals)
    
    # Compare with key profiles
    max_correlation = -1
    detected_key = None
    detected_mode = None
    
    for mode in ['major', 'minor']:
        profile = np.array(KEY_PROFILES[mode])
        profile = profile / np.sum(profile)
        
        for i in range(12):
            # Rotate profile for each key
            rotated_profile = np.roll(profile, i)
            
            # Calculate correlation
            correlation = np.corrcoef(chroma_vals, rotated_profile)[0, 1]
            
            if correlation > max_correlation:
                max_correlation = correlation
                detected_key = KEYS[i]
                detected_mode = mode
    
    # Format key name (e.g., "Cm" for C minor, "C" for C major)
    if detected_mode == 'minor':
        key_name = detected_key + 'm'
    else:
        key_name = detected_key
    
    return key_name


def rename_files_with_keys(directory='.'):
    """Process all WAV files in directory and rename them with detected keys."""
    directory = Path(directory)
    wav_files = list(directory.rglob('*.wav')) + list(directory.rglob('*.WAV'))
    
    if not wav_files:
        print(f"No WAV files found in {directory}")
        return
    
    print(f"Found {len(wav_files)} WAV file(s)")
    
    for wav_file in wav_files:
        try:
            print(f"\nProcessing: {wav_file.name}")
            
            # Detect key
            key = detect_key(str(wav_file))
            print(f"  Detected key: {key}")
            
            # Create new filename
            stem = wav_file.stem
            suffix = wav_file.suffix
            
            # Check if key is already in filename
            if key not in stem:
                new_name = f"{stem}_{key}{suffix}"
                new_path = wav_file.parent / new_name
                
                # Rename file
                wav_file.rename(new_path)
                print(f"  Renamed to: {new_name}")
            else:
                print(f"  Key already in filename, skipping rename")
                
        except Exception as e:
            print(f"  Error processing {wav_file.name}: {e}")


if __name__ == '__main__':
    # Get directory from command line argument or use current directory
    target_dir = sys.argv[1] if len(sys.argv) > 1 else '.'
    
    print(f"Detecting keys in WAV files from: {target_dir}\n")
    rename_files_with_keys(target_dir)
    print("\nDone!")
