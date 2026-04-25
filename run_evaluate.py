import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluation.evaluate import evaluate_dataset

if __name__ == "__main__":
    evaluate_dataset(
        val_dir="data/validation",
        ann_dir="annotation",
        save_dir="output_validation",
        iou_threshold=0.30,
    )
