import argparse
import os
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import softmax
from shapely.geometry import Polygon
from tqdm import trange

from calibrators import DoublyBoundedScaling
from calibrators import PlattScaling
from calibrators import TemperatureScaling


def convert_format(boxes_array):
    polygons = [
        Polygon([(box[i, 0], box[i, 1]) for i in range(4)])
        for box in boxes_array
    ]
    return np.array(polygons)


def compute_iou(box, boxes):
    iou = [box.intersection(b).area / box.union(b).area for b in boxes]
    return np.array(iou, dtype=np.float32)


def nms_rotated(boxes, scores, threshold):
    if boxes.shape[0] == 0:
        return np.array([], dtype=np.int32)
    polygons = convert_format(boxes)
    top = 1000
    ixs = scores.argsort()[::-1][:top]
    pick = []
    while len(ixs) > 0:
        i = ixs[0]
        pick.append(i)
        iou = compute_iou(polygons[i], polygons[ixs[1:]])
        remove_ixs = np.where(iou > threshold)[0] + 1
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)
    return np.array(pick, dtype=np.int32)


def caluclate_tp_fp(det_boxes, det_score, gt_boxes, result_stat, iou_thresh):
    fp = []
    tp = []
    gt = gt_boxes.shape[0]
    if det_boxes is not None:
        # sort the prediction bounding box by score
        score_order_descend = np.argsort(-det_score)
        det_polygon_list = list(convert_format(det_boxes))
        gt_polygon_list = list(convert_format(gt_boxes))
        # match prediction and gt bounding box
        for i in range(score_order_descend.shape[0]):
            det_polygon = det_polygon_list[score_order_descend[i]]
            ious = compute_iou(det_polygon, gt_polygon_list)
            if len(gt_polygon_list) == 0 or np.max(ious) < iou_thresh:
                fp.append(1)
                tp.append(0)
                continue
            fp.append(0)
            tp.append(1)
            gt_index = np.argmax(ious)
            gt_polygon_list.pop(gt_index)
    result_stat[iou_thresh]['fp'] += fp
    result_stat[iou_thresh]['tp'] += tp
    result_stat[iou_thresh]['gt'] += gt


def init_stats():
    stats = {}
    for iou in [0.3, 0.5, 0.7]:
        stats[iou] = {'tp': [], 'fp': [], 'gt': 0}
    return stats


def evaluation(preds, probs, trues, stats):
    for iou in [0.3, 0.5, 0.7]:
        caluclate_tp_fp(preds, probs, trues, stats, iou)


def voc_ap(rec, prec):
    """
    VOC 2010 Average Precision.
    """
    rec.insert(0, 0.0)
    rec.append(1.0)
    mrec = rec[:]

    prec.insert(0, 0.0)
    prec.append(0.0)
    mpre = prec[:]

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            i_list.append(i)

    ap = 0.0
    for i in i_list:
        ap += ((mrec[i] - mrec[i - 1]) * mpre[i])
    return ap, mrec, mpre


def calculate_ap(result_stat, iou):
    """
    Calculate the average precision and recall, and save them into a txt.

    Parameters
    ----------
    result_stat : dict
        A dictionary contains fp, tp and gt number.
    iou : float
    """
    iou_5 = result_stat[iou]

    fp = iou_5['fp']
    tp = iou_5['tp']
    assert len(fp) == len(tp)

    gt_total = iou_5['gt']

    cumsum = 0
    for idx, val in enumerate(fp):
        fp[idx] += cumsum
        cumsum += val

    cumsum = 0
    for idx, val in enumerate(tp):
        tp[idx] += cumsum
        cumsum += val

    rec = tp[:]
    for idx, val in enumerate(tp):
        rec[idx] = float(tp[idx]) / gt_total

    prec = tp[:]
    for idx, val in enumerate(tp):
        prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])

    ap, _, _ = voc_ap(rec[:], prec[:])

    return ap


def eval_final_results(result_stat):
    ap_30 = calculate_ap(result_stat, 0.30)
    ap_50 = calculate_ap(result_stat, 0.50)
    ap_70 = calculate_ap(result_stat, 0.70)
    print('The Average Precision at IOU 0.3 is %.3f, '
          'The Average Precision at IOU 0.5 is %.3f, '
          'The Average Precision at IOU 0.7 is %.3f' % (ap_30, ap_50, ap_70))


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model-dir',
        type=str,
        required=True,
        help='Model directory.',
    )
    parser.add_argument(
        '--cav-model-dir',
        type=str,
        default="none",
        help='CAV Model directory.',
    )
    parser.add_argument('--calibrator',
                        type=str,
                        default="none",
                        help='Choose calibrator from {none, ps, ts, dbs}.')
    parser.add_argument('--aggregator',
                        type=str,
                        default="nms",
                        help='Choose aggregator from {nms, psa}.')
    parser.add_argument('--threshold',
                        type=float,
                        default=0.0,
                        help='Confidence threshold for filtering.')
    args = parser.parse_args()
    print(args)
    return args


def non_maximum_suppression(preds, probs):
    keep_index = nms_rotated(preds, probs, threshold=0.15)
    preds = preds[keep_index]
    probs = probs[keep_index]
    return preds, probs


def load_data(frame_dir):
    if frame_dir is None:
        return None, None, None
    data = np.load(frame_dir)
    preds = data["preds"]
    probs = data["probs"]
    trues = data["trues"]
    return preds, probs, trues


def calibration(calibrator, probs):
    if calibrator is None or len(probs) == 0:
        return probs
    else:
        return calibrator.transform(probs)


def to_polygon(box):
    return Polygon([(box[i, 0], box[i, 1]) for i in range(4)])


def compute_self_iou_mat(boxes, dist_threshold=10.0):
    centers = boxes.mean(axis=1)
    dist_mat = cdist(centers, centers)
    iou_mat = np.zeros_like(dist_mat)
    np.fill_diagonal(iou_mat, 1.0)
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if dist_mat[i, j] < dist_threshold:
                polygon1 = to_polygon(boxes[i])
                polygon2 = to_polygon(boxes[j])
                iou = polygon1.intersection(polygon2).area / polygon1.union(
                    polygon2).area
                if iou > iou_mat[i, j]:
                    iou_mat[i, j] = iou
                    iou_mat[j, i] = iou
    return iou_mat


def aggregation(args, preds, probs):
    if args.aggregator == "nms":
        preds, probs = non_maximum_suppression(preds, probs)
    elif args.aggregator == "psa":
        iou_mat = compute_self_iou_mat(preds)
        clusters, visited, selected = [], [], []
        for idx, ious in enumerate(iou_mat):
            if idx in visited:
                continue
            neighbor_idxs = np.nonzero(ious)[0]
            clusters.append(neighbor_idxs)
            visited.extend(neighbor_idxs)
        for cluster in clusters:
            sub_iou_mat = iou_mat[np.ix_(cluster, cluster)]
            sub_probs = probs[cluster]
            values = sub_iou_mat.dot(sub_probs)
            soft_bools = softmax(values / 1e-6)
            bools = soft_bools > 0.5
            selected.extend(cluster[bools])
        preds = preds[selected]
        probs = probs[selected]
    else:
        raise ValueError("Choose aggregator from {nms, psa}.")
    return preds, probs


def load_calibrator(calibrator_name, calibrator_path):
    if calibrator_name == "none":
        return None
    elif calibrator_name == "ps":
        calibrator = PlattScaling()
    elif calibrator_name == "dbs":
        calibrator = DoublyBoundedScaling()
    elif calibrator_name == "ts":
        calibrator = TemperatureScaling()
    else:
        raise ValueError("Choose calibrator from {none, ps, ts, dbs}.")
    print(f"Loading calibrator from {calibrator_path}")
    calibrator.load_model(calibrator_path)
    return calibrator


def get_calibrators(args):
    calibrator_path = os.path.join(args.model_dir, f"{args.calibrator}.pt")
    calibrator = load_calibrator(args.calibrator, calibrator_path)
    if args.cav_model_dir == "none":
        cav_calibrator = None
    else:
        cav_calibrator_path = os.path.join(args.cav_model_dir,
                                           f"{args.calibrator}.pt")
        cav_calibrator = load_calibrator(args.calibrator, cav_calibrator_path)
    return calibrator, cav_calibrator


def filtration(args, preds, probs):
    if preds is None:
        return None, None
    selected = probs > args.threshold
    preds = preds[selected]
    probs = probs[selected]
    return preds, probs


def get_data_dirs(args):
    # Hetero setting
    if args.cav_model_dir != "none":
        data_dir = os.path.join(args.model_dir, f"hetero")
        cav_data_dir = os.path.join(args.cav_model_dir, f"hetero")
    else:  # Homo setting
        data_dir = os.path.join(args.model_dir, f"test")
        cav_data_dir = None
    # Make sure that the directories exist
    if not Path(data_dir).exists():
        raise ValueError(f"{data_dir} does not exist!")
    if cav_data_dir and not Path(cav_data_dir).exists():
        raise ValueError(f"{cav_data_dir} does not exist!")
    return data_dir, cav_data_dir


def get_frame_paths(data_dir, cav_data_dir, frame_id):
    frame_path = os.path.join(data_dir, f"{frame_id:04d}.npz")
    if not Path(frame_path).exists():
        #  print(f"{frame_path} is missing!")
        return None, None
    if cav_data_dir is None:
        cav_frame_path = None
    else:
        cav_frame_path = os.path.join(cav_data_dir, f"{frame_id:04d}.npz")
        if not Path(cav_frame_path).exists():
            #  print(f"{cav_frame_path} is missing!")
            return None, None
    return frame_path, cav_frame_path


def main():
    args = parse_arguments()
    data_dir, cav_data_dir = get_data_dirs(args)
    calibrator, cav_calibrator = get_calibrators(args)
    stats = init_stats()
    for frame_id in trange(2170):
        frame_path, cav_frame_path = get_frame_paths(data_dir, cav_data_dir,
                                                     frame_id)
        if frame_path is None and cav_frame_path is None:
            continue
        preds, probs, trues = load_data(frame_path)
        cav_preds, cav_probs, cav_trues = load_data(cav_frame_path)
        if cav_trues is not None:
            np.testing.assert_allclose(trues, cav_trues)
        probs = calibration(calibrator, probs)
        cav_probs = calibration(cav_calibrator, cav_probs)
        preds, probs = filtration(args, preds, probs)
        cav_preds, cav_probs = filtration(args, cav_preds, cav_probs)
        if cav_preds is not None:
            preds = np.vstack((preds, cav_preds))
        if cav_probs is not None:
            probs = np.hstack((probs, cav_probs))
        preds, probs = aggregation(args, preds, probs)
        evaluation(preds, probs, trues, stats)
    eval_final_results(stats)


if __name__ == "__main__":
    main()
