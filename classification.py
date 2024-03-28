import sys
import os
sys.path.append(os.getcwd())

import argparse
import subprocess

def train():
    ''' Trains the model and generates models '''
    subprocess.run(["python", "./runfiles/training.py"])

def run_existing(argument):
    ''' Runs the model from the generated data from run (Must be run for this to work properly) '''
    subprocess.run(["python", "./runfiles/run_existing.py", argument])

def run(argument):
    ''' Runs the desired model against the model of choice '''
    subprocess.run(["python", "./runfiles/run.py", argument])

if __name__ == "__main__":
    """ Main function takes in arguments and executes the correct program """
    parser = argparse.ArgumentParser(description="PDF Classification Script")
    parser.add_argument("mode", choices=["train", "run-existing", "run"], help="Mode: train, run-existing <model number> (default = 1), or run <model number> (default = 1) the model")
    parser.add_argument("argument", nargs="?", default=1, help="Model number to be run")
    args = parser.parse_args()

    if args.mode == "train":
        train()
    elif args.mode == "run-existing":
        run_existing(args.argument)
    elif args.mode == "run":
        run(args.argument)
    else:
        print("Mode: train or run (default) the model")
