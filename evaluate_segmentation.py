import numpy as np
import pandas as pd
import tifffile as tif
import argparse
import os
from collections import OrderedDict
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score

# Assuming you have a function to evaluate the Dice coefficient
from train_tools.measures import evaluate_f1_score_cellseg, evaluate_dice_coefficient

def main():
    ### Directory path arguments ###
    parser = argparse.ArgumentParser("Compute metrics for cell segmentation results")
    parser.add_argument(
        "--gt_path",
        type=str,
        help="path to ground truth; file names end with _label.tiff",
        required=True,
    )
    parser.add_argument(
        "--pred_path", type=str, help="path to segmentation results", required=True
    )
    parser.add_argument("--save_path", default=None, help="path where to save metrics")
    parser.add_argument("--visualize", action='store_true', help="visualize some results")

    args = parser.parse_args()

    # Get files from the paths
    gt_path, pred_path = args.gt_path, args.pred_path
    names = sorted(os.listdir(pred_path))

    names_total = []
    precisions_total, recalls_total, f1_scores_total, dice_scores_total = [], [], [], []

    for name in tqdm(names):
        if not name.endswith("_label.tiff"):
            continue  # Skip files not ending with _label.tiff

        # Load the images
        gt = tif.imread(os.path.join(gt_path, name))
        pred = tif.imread(os.path.join(pred_path, name))

        # Evaluate metrics
        precision, recall, f1_score = evaluate_f1_score_cellseg(gt, pred, threshold=0.5)
        dice_score = evaluate_dice_coefficient(gt, pred)

        names_total.append(name)
        precisions_total.append(np.round(precision, 4))
        recalls_total.append(np.round(recall, 4))
        f1_scores_total.append(np.round(f1_score, 4))
        dice_scores_total.append(np.round(dice_score, 4))

        # Optional: Visualize some results
        if args.visualize and len(names_total) <= 5:  # Visualize first 5 images
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 3, 1)
            plt.imshow(gt, cmap='gray')
            plt.title('Ground Truth')
            plt.subplot(1, 3, 2)
            plt.imshow(pred, cmap='gray')
            plt.title('Prediction')
            plt.subplot(1, 3, 3)
            plt.imshow(gt, cmap='gray')
            plt.imshow(pred, cmap='jet', alpha=0.5)
            plt.title(f'Overlay (F1: {f1_score:.4f}, Dice: {dice_score:.4f})')
            plt.show()

    # Refine data as dataframe
    cellseg_metric = OrderedDict()
    cellseg_metric["Names"] = names_total
    cellseg_metric["Precision"] = precisions_total
    cellseg_metric["Recall"] = recalls_total
    cellseg_metric["F1_Score"] = f1_scores_total
    cellseg_metric["Dice_Score"] = dice_scores_total

    cellseg_metric = pd.DataFrame(cellseg_metric)
    print("Mean F1 Score:", np.mean(cellseg_metric["F1_Score"]))
    print("Mean Dice Score:", np.mean(cellseg_metric["Dice_Score"]))

    # Save results
    if args.save_path is not None:
        os.makedirs(args.save_path, exist_ok=True)
        cellseg_metric.to_csv(
            os.path.join(args.save_path, "seg_metric.csv"), index=False
        )

if __name__ == "__main__":
    main()
