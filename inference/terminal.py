import os
import subprocess
import sys

def main():
    # Define paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_path = os.path.join(script_dir, "DeepSeek-V3-Demo")
    config_path = os.path.join(script_dir, "configs", "config_671B.json")

    # Check if the converted model weights exist
    if not os.path.exists(ckpt_path):
        print("Converted model weights not found.")
        print("Please run the following command to convert the weights first:")
        print(f"python {os.path.join(script_dir, 'convert.py')} --hf-ckpt-path /path/to/DeepSeek-V3 --save-path {ckpt_path} --n-experts 256 --model-parallel 1")
        sys.exit(1)

    # Command to run generate.py
    command = [
        "torchrun",
        "--nproc_per_node=1",
        os.path.join(script_dir, "generate.py"),
        "--ckpt-path",
        ckpt_path,
        "--config",
        config_path,
        "--interactive"
    ]

    print("Starting the interactive terminal...")
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running the terminal: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("torchrun not found. Please make sure you have torch installed and it is in your PATH.")
        sys.exit(1)

if __name__ == "__main__":
    main()