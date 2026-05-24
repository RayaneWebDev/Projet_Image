"""
Évaluation du pipeline de classification classique (k-NN).

Lance evaluate_dataset() sur data/validation avec les annotations JSON
et affiche précision, rappel et taux d'identification globaux.

Utilisation :
    python run_evaluate.py

Auteurs : Équipe ImageGroupe
Date    : 2026
"""
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
