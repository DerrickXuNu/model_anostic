import argparse
import os

from netcal.metrics import ACE, ECE, MCE
import numpy as np

from calibrators import BCE
from calibrators import DoublyBoundedScaling
from calibrators import PlattScaling
from calibrators import TemperatureScaling


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train calibrator.")
    parser.add_argument(
        '--model-dir',
        type=str,
        required=True,
        help='Model directory.',
    )
    parser.add_argument(
        '--method',
        type=str,
        default="ps",
        help='Calibration method {ps, ts, dbs}',
    )
    parser.add_argument(
        '--bins',
        type=int,
        default=50,
        help='Number of bins for computing calibration error.',
    )
    parser.add_argument(
        '--iou-threshold',
        type=float,
        default=0.7,
        help='A threshold that transforms IoUs to labels.',
    )
    args = parser.parse_args()
    return args


def get_data(args):
    data_path = os.path.join(args.model_dir, "calibration_data.npz")
    data = np.load(data_path)
    probs, ious = data["probs"], data["ious"]
    labels = (ious > args.iou_threshold).astype(int)
    return probs, labels


def get_calibrator(args):
    if args.method == "ps":
        calibrator = PlattScaling()
    elif args.method == "dbs":
        calibrator = DoublyBoundedScaling()
    elif args.method == "ts":
        calibrator = TemperatureScaling()
    else:
        raise ValueError("Choose method from {ps, ts, dbs}.")
    return calibrator


class Metrics:

    def __init__(self, bins):
        self.ece = ECE(bins=bins)
        self.ace = ACE(bins=bins)
        self.mce = MCE(bins=bins)

    def print(self, probs, labels):
        ece_loss = self.ece.measure(probs, labels)
        ace_loss = self.ace.measure(probs, labels)
        mce_loss = self.mce.measure(probs, labels)
        log_loss = BCE(probs, labels)
        print(
            f"ECE: {ece_loss:.8f} ",
            f"ACE: {ace_loss:.8f} ",
            f"MCE: {mce_loss:.8f} ",
            f"LogLoss: {log_loss:.8f}",
        )


def main():
    args = parse_arguments()

    metrics = Metrics(args.bins)
    calibrator = get_calibrator(args)
    probs, labels = get_data(args)

    print("Confidence metrics before calibration (the smaller the better)")
    metrics.print(probs, labels)

    print("Training calibrator...")
    calibrator.fit(probs, labels)
    calibrator_path = os.path.join(args.model_dir, f"{args.method}.pt")
    calibrator.save_model(calibrator_path)
    calibrated_probs = calibrator.transform(probs)

    print("Confidence metrics after calibration")
    metrics.print(calibrated_probs, labels)

    print(f"Calibrator saved to {calibrator_path}")


if __name__ == "__main__":
    main()
