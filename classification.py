import argparse
import subprocess

def train():
    subprocess.run(["python", "./Train/training.py"])

def run():
    subprocess.run(["python", "./Run/run.py"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PDF Classification Script")
    parser.add_argument("mode", choices=["train", "process", "run"], help="Mode: train or run (default) the model")
    args = parser.parse_args()

    if args.mode == "train":
        train()
    else:
        run()