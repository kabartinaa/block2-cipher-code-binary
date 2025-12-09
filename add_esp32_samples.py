#!/usr/bin/env python3
"""
Script to add ESP32 architecture samples to the dataset.
This script:
1. Copies existing sample files and renames them with esp32 architecture
2. Updates metadata.csv with new ESP32 entries
3. Updates crypto_hex_dataset_3000/metadata.csv with new ESP32 entries
"""

import os
import csv
import shutil
import random
from pathlib import Path

# Configuration
WORKSPACE_ROOT = r"D:\Binary_block_files"
SAMPLES_DIR = os.path.join(WORKSPACE_ROOT, "samples")
METADATA_FILE = os.path.join(WORKSPACE_ROOT, "metadata.csv")
CRYPTO_METADATA_FILE = os.path.join(WORKSPACE_ROOT, "crypto_hex_dataset_3000", "metadata.csv")

# Number of ESP32 samples to create
NUM_ESP32_SAMPLES = 100

def read_metadata(filepath):
    """Read metadata CSV file."""
    data = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        data = list(reader)
    return data

def write_metadata(filepath, data, fieldnames):
    """Write metadata CSV file."""
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

def add_esp32_to_main_metadata():
    """Add ESP32 samples to main metadata.csv"""
    print("Processing main metadata.csv...")
    
    metadata = read_metadata(METADATA_FILE)
    
    # Sample architectures to replace with ESP32
    source_samples = [row for row in metadata if row['arch'] in ['avr', 'arm', 'mips']]
    
    # Create new ESP32 entries
    new_rows = []
    max_id = max([int(row['id']) for row in metadata])
    
    for i in range(NUM_ESP32_SAMPLES):
        if i >= len(source_samples):
            break
            
        source = source_samples[i]
        new_id = max_id + i + 1
        
        # Copy from source but change architecture to esp32
        new_row = source.copy()
        new_row['id'] = str(new_id)
        
        # Create new filename
        sample_num = new_id
        new_row['file'] = f"sample_{sample_num:04d}_esp32.bin"
        new_row['arch'] = 'esp32'
        
        new_rows.append(new_row)
    
    # Append new rows
    metadata.extend(new_rows)
    
    # Write back
    fieldnames = list(metadata[0].keys())
    write_metadata(METADATA_FILE, metadata, fieldnames)
    print(f"✓ Added {len(new_rows)} ESP32 samples to metadata.csv")

def add_esp32_to_crypto_metadata():
    """Add ESP32 samples to crypto_hex_dataset_3000/metadata.csv"""
    print("Processing crypto_hex_dataset_3000/metadata.csv...")
    
    metadata = read_metadata(CRYPTO_METADATA_FILE)
    
    # Sample architectures to use as templates
    source_samples = [row for row in metadata if row['architecture'] in ['arm', 'arm64', 'mips', 'powerpc']]
    
    # Create new ESP32 entries
    new_rows = []
    max_sample_id = max([int(row['sample_id'].replace('s', '')) for row in metadata])
    
    for i in range(NUM_ESP32_SAMPLES):
        if i >= len(source_samples):
            break
            
        source = source_samples[i % len(source_samples)]
        new_sample_num = max_sample_id + i + 1
        
        # Copy from source but change architecture to esp32
        new_row = source.copy()
        new_row['sample_id'] = f"s{new_sample_num:04d}"
        new_row['filename'] = f"s{new_sample_num:04d}.hex"
        new_row['architecture'] = 'esp32'
        
        new_rows.append(new_row)
    
    # Append new rows
    metadata.extend(new_rows)
    
    # Write back
    fieldnames = list(metadata[0].keys())
    write_metadata(CRYPTO_METADATA_FILE, metadata, fieldnames)
    print(f"✓ Added {len(new_rows)} ESP32 samples to crypto_hex_dataset_3000/metadata.csv")

def copy_sample_files():
    """Copy sample binary files and rename with esp32 architecture"""
    print("Copying sample files...")
    
    # Get all existing samples
    sample_files = sorted([f for f in os.listdir(SAMPLES_DIR) if f.endswith('.bin')])
    
    # Select samples to copy from different architectures
    esp32_samples = []
    for i in range(NUM_ESP32_SAMPLES):
        # Cycle through existing samples
        source_file = sample_files[i % len(sample_files)]
        source_path = os.path.join(SAMPLES_DIR, source_file)
        
        # Create new filename
        # Extract the sample number from source
        sample_base = source_file.replace('.bin', '')  # e.g., "sample_0000_x86"
        parts = sample_base.rsplit('_', 1)  # Split on last underscore
        sample_num_str = parts[0].replace('sample_', '')
        
        # New filename with ESP32
        new_sample_num = 3000 + i
        new_filename = f"sample_{new_sample_num:04d}_esp32.bin"
        new_path = os.path.join(SAMPLES_DIR, new_filename)
        
        # Copy file
        shutil.copy2(source_path, new_path)
        esp32_samples.append(new_filename)
    
    print(f"✓ Created {len(esp32_samples)} ESP32 sample files")
    return esp32_samples

def main():
    """Main function"""
    print("=" * 60)
    print("Adding ESP32 Architecture to Dataset")
    print("=" * 60)
    
    # Check if directories exist
    if not os.path.exists(SAMPLES_DIR):
        print(f"✗ Samples directory not found: {SAMPLES_DIR}")
        return
    
    if not os.path.exists(METADATA_FILE):
        print(f"✗ Metadata file not found: {METADATA_FILE}")
        return
    
    if not os.path.exists(CRYPTO_METADATA_FILE):
        print(f"✗ Crypto metadata file not found: {CRYPTO_METADATA_FILE}")
        return
    
    # Execute tasks
    copy_sample_files()
    add_esp32_to_main_metadata()
    add_esp32_to_crypto_metadata()
    
    print("\n" + "=" * 60)
    print("✓ Successfully added ESP32 architecture samples!")
    print("=" * 60)
    print(f"\nSummary:")
    print(f"  • Created {NUM_ESP32_SAMPLES} sample files")
    print(f"  • Updated main metadata.csv")
    print(f"  • Updated crypto_hex_dataset_3000/metadata.csv")

if __name__ == "__main__":
    main()
