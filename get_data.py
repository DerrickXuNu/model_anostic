import argparse
import os

import numpy as np
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon
from tqdm import tqdm


def boxes3d_to_polygons2d(boxes_array):
    polygons = np.array([
        Polygon([(box[i, 0], box[i, 1]) for i in range(4)])
        for box in boxes_array
    ])
    return polygons


def get_sorted_data_list(data_dir):
    data_list = sorted([
        os.path.join(data_dir, f) for f in os.listdir(data_dir)
        if os.path.isfile(os.path.join(data_dir, f))
    ])
    return data_list


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Compute Intersection over Union (IoU)")
    parser.add_argument(
        '--model-dir',
        type=str,
        required=True,
        help='Model prediction directory.',
    )
    parser.add_argument(
        '--iou-dist',
        type=float,
        default=10.0,
        help='Distance threshold: only compute IoUs for nearby polygons.',
    )
    args = parser.parse_args()
    return args


def get_polygon_centers(polygons):
    centers = np.array([[p.centroid.x, p.centroid.y]
                        for p in polygons]).reshape(-1, 2)
    return centers


def compute_ious(preds, trues, dist_threshold):
    pred_centers = get_polygon_centers(preds)
    true_centers = get_polygon_centers(trues)
    dist_mat = cdist(true_centers, pred_centers)
    ious = np.zeros(len(preds))
    for idx, gt in enumerate(trues):
        dists = dist_mat[idx]
        related = dists < dist_threshold
        related_preds = preds[related]
        current_ious = np.array([
            gt.intersection(pred).area / gt.union(pred).area
            for pred in related_preds
        ])
        ious[related] = np.maximum(ious[related], current_ious)
    return ious


def main():
    args = parse_arguments()
    data_dir = os.path.join(args.model_dir, "train")
    cali_path = os.path.join(args.model_dir, "calibration_data.npz")
    data_list = get_sorted_data_list(data_dir)
    if not data_list:
        raise ValueError("No file in this folder!")
    prob_list, iou_list = [], []
    for data_path in tqdm(data_list):
        data = np.load(data_path)
        # Predictive bounding boxes, probabilities, ground-truth boxes
        preds, probs, trues = data["preds"], data["probs"], data["trues"]
        pred_polygons = boxes3d_to_polygons2d(preds)
        true_polygons = boxes3d_to_polygons2d(trues)
        ious = compute_ious(pred_polygons, true_polygons, args.iou_dist)
        prob_list.append(probs)
        iou_list.append(ious)
    prob_array = np.hstack(prob_list)
    iou_array = np.hstack(iou_list)
    np.savez_compressed(
        cali_path,
        probs=prob_array,
        ious=iou_array,
    )
    print(f"Calibration data saved to {cali_path}.")


if __name__ == "__main__":
    main()
