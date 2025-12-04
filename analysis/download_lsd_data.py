#!/usr/bin/env python3
"""
Download the Carhart-Harris LSD dataset from OpenNeuro (ds003059).

This dataset contains resting-state fMRI from 15 subjects under LSD vs placebo.
Reference: Carhart-Harris et al. (2016) PNAS
"""

import os
import subprocess
import sys

# Dataset info
DATASET_ID = "ds003059"
DATASET_URL = f"https://openneuro.org/datasets/{DATASET_ID}"
OUTPUT_DIR = "/Users/iantodd/Desktop/highdimensional/25_lsd_dimensionality/data"

def check_aws_cli():
    """Check if AWS CLI is installed (needed for OpenNeuro download)."""
    try:
        subprocess.run(["aws", "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def check_datalad():
    """Check if datalad is installed (alternative download method)."""
    try:
        subprocess.run(["datalad", "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def download_with_aws():
    """Download dataset using AWS CLI (no sign-up required for OpenNeuro)."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cmd = [
        "aws", "s3", "sync",
        "--no-sign-request",
        f"s3://openneuro.org/{DATASET_ID}",
        os.path.join(OUTPUT_DIR, DATASET_ID)
    ]

    print(f"Downloading {DATASET_ID} from OpenNeuro...")
    print(f"Command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"Download complete: {os.path.join(OUTPUT_DIR, DATASET_ID)}")

def download_with_openneuro_cli():
    """Alternative: download using openneuro-cli if available."""
    try:
        subprocess.run(["openneuro-cli", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("openneuro-cli not found. Install with: npm install -g @openneuro/cli")
        return False

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cmd = ["openneuro-cli", "download", DATASET_ID, os.path.join(OUTPUT_DIR, DATASET_ID)]
    subprocess.run(cmd, check=True)
    return True

def main():
    print(f"="*60)
    print(f"OpenNeuro LSD Dataset Downloader")
    print(f"Dataset: {DATASET_ID}")
    print(f"URL: {DATASET_URL}")
    print(f"="*60)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if check_aws_cli():
        print("\nUsing AWS CLI for download (no sign-up required)...")
        download_with_aws()
    else:
        print("\nAWS CLI not found.")
        print("Install with: brew install awscli")
        print("\nAlternatively, download manually from:")
        print(f"  {DATASET_URL}")
        print(f"\nOr use: pip install openneuro-py")
        sys.exit(1)

if __name__ == "__main__":
    main()
