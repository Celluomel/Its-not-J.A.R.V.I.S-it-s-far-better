import os

from brain_visualizer import run_brain_visualizer

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    run_brain_visualizer(base_dir)