#!/usr/bin/env python3
"""
Entry point script for running heretic on Azure ML.
This script handles job arguments, runs heretic with auto-save mode,
and manages output directories for AML.
"""

import argparse
import os
import subprocess
import sys

# Check Python version
if sys.version_info < (3, 10):
    print(f"Error: Python 3.10 or higher is required (you have {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro})")
    print("Heretic uses Python 3.10+ features like match statements.")
    print("\nFor local testing, please use Python 3.10 or higher.")
    print("On Azure ML, the environment.yml specifies Python 3.11, so this will work correctly there.")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Run lucidity on Azure ML with automatic model saving")
    parser.add_argument("--model", type=str, required=True, help="Model name or path (e.g., 'meta-llama/Llama-2-7b-hf')")
    parser.add_argument('--hf-token', type=str, help='HuggingFace token')
    parser.add_argument("--n-trials", type=int, default=10, help="Number of optimization trials to run (default: 200)")
    parser.add_argument("--n-startup-trials", type=int, default=60, help="Number of random startup trials (default: 60)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for processing (0 = auto, default: 0)")
    parser.add_argument("--max-batch-size", type=int, default=128, help="Maximum batch size to try during auto-detection (default: 128)")
    parser.add_argument("--config", type=str, help="Path to custom config.toml file")
    parser.add_argument("--output", type=str, help="Output directory path (provided by Azure ML)")
    args = parser.parse_args()

    # Set up output directory - use the path provided by Azure ML component
    if args.output:
        model_output_dir = args.output
    else:
        output_dir = os.environ.get("AZUREML_OUTPUT_DIR", "./outputs")
        model_output_dir = os.path.join(output_dir, "model_outputs")
    os.makedirs(model_output_dir, exist_ok=True)

    print("=" * 80)
    print("Azure ML Lucidity Job")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"N Trials: {args.n_trials}")
    print(f"N Startup Trials: {args.n_startup_trials}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Output Directory: {model_output_dir}")
    print("=" * 80)
    print()

    # Get paths - we're already in the repo root
    repo_root = os.path.dirname(os.path.abspath(__file__))
    
    # Install heretic in editable mode (required for version metadata and proper module loading)
    print("Installing heretic package in editable mode...")
    install_result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-e", repo_root],
        capture_output=True,
        text=True
    )
    if install_result.returncode != 0:
        print(f"Error installing heretic: {install_result.stderr}")
        sys.exit(1)
    print("Heretic installed successfully.")
    print()

    # Build command to run heretic by calling the main function directly
    # We use python -c to call heretic.main:main() which is the entry point
    
    auto_upload_to_hf = bool(args.hf_token and args.hf_token.strip())
    
    cmd = [
        sys.executable,
        "-c",
        "from heretic.main import main; main()",
        "--model", args.model,
        "--n-trials", str(args.n_trials),
        "--n-startup-trials", str(args.n_startup_trials),
        "--batch-size", str(args.batch_size),
        "--max-batch-size", str(args.max_batch_size),
        "--auto-save", "true",
        "--hf-token", str(args.hf_token),
        "--auto-upload-to-hf", str(auto_upload_to_hf),
        "--output-dir", model_output_dir,
    ]

    if args.config:
        cmd.extend(["--config", args.config])

    print("Starting heretic optimization...")
    print(f"Running: {' '.join(cmd)}")
    print()
    print("="*80)

    try:
        # Run without capturing output so we can see what heretic is doing
        result = subprocess.run(cmd, cwd=repo_root, check=False)

        print("="*80)

        if result.returncode != 0:
            print(f"\nError: Heretic exited with code {result.returncode}", file=sys.stderr)
            sys.exit(result.returncode)

        print()
        print("=" * 80)
        print("Job completed successfully!")
        print(f"Model outputs saved to: {model_output_dir}")
        print("=" * 80)

        # List saved models
        if os.path.exists(model_output_dir):
            saved_dirs = sorted([d for d in os.listdir(model_output_dir) 
                               if os.path.isdir(os.path.join(model_output_dir, d))])
            if saved_dirs:
                print(f"\nSaved {len(saved_dirs)} model(s):")
                for d in saved_dirs:
                    print(f"  - {d}")
            else:
                print("\nWarning: No model directories found in output.")

    except Exception as e:
        print(f"\nError running heretic: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
