###############################################################################
# Homo: no calibration + NMS
###############################################################################
python main.py --model-dir ./models/pointpillar30 --calibrator none --aggregator nms
# Namespace(aggregator='nms', calibrator='none', model_dir='./models/pointpillar30', threshold=0.0)
# The Average Precision at IOU 0.3 is 0.866, The Average Precision at IOU 0.5 is 0.858, The Average Precision at IOU 0.7 is 0.781

###############################################################################
# Hetero1: no calibration + NMS
###############################################################################
python main.py --model-dir ./models/pointpillar30 --calibrator none --aggregator nms --cav-model-dir ./models/pointpillar5
# The Average Precision at IOU 0.3 is 0.844, The Average Precision at IOU 0.5 is 0.832, The Average Precision at IOU 0.7 is 0.691

###############################################################################
# Hetero2: no calibration + NMS
###############################################################################
python main.py --model-dir ./models/pointpillar30 --calibrator none --aggregator nms --cav-model-dir ./models/second5
# The Average Precision at IOU 0.3 is 0.857, The Average Precision at IOU 0.5 is 0.846, The Average Precision at IOU 0.7 is 0.723

###############################################################################
# Training calibrators
###############################################################################
python get_data.py --model-dir ./models/pointpillar30/
# Calibration data saved to ./models/pointpillar30/calibration_data.npz.
python train_calibrator.py --model-dir ./models/pointpillar30/ --method ps
# Confidence metrics before calibration (the smaller the better)
# ECE: 0.49429780  ACE: 0.41582962  MCE: 0.54603847  LogLoss: 0.93936052
# Confidence metrics after calibration
# ECE: 0.00253576  ACE: 0.00284886  MCE: 0.00912496  LogLoss: 0.36603310
# Calibrator saved to ./models/pointpillar30/ps.pt
python train_calibrator.py --model-dir ./models/pointpillar30/ --method ts
# ECE: 0.03765022  ACE: 0.03843984  MCE: 0.10528693  LogLoss: 0.37484726
# Calibrator saved to ./models/pointpillar30/ts.pt
python train_calibrator.py --model-dir ./models/pointpillar30/ --method dbs
# ECE: 0.00221240  ACE: 0.00248107  MCE: 0.01177005  LogLoss: 0.36602245
# Calibrator saved to ./models/pointpillar30/dbs.pt

python get_data.py --model-dir ./models/pointpillar5
# Calibration data saved to ./models/pointpillar5/calibration_data.npz.
python train_calibrator.py --model-dir ./models/pointpillar5/ --method ps
# Confidence metrics before calibration (the smaller the better)
# ECE: 0.34642801  ACE: 0.38754094  MCE: 0.51435972  LogLoss: 0.85705118
# Confidence metrics after calibration
# ECE: 0.01587025  ACE: 0.01807574  MCE: 0.05821192  LogLoss: 0.57571361
# Calibrator saved to ./models/pointpillar5/ps.pt
python train_calibrator.py --model-dir ./models/pointpillar5/ --method ts
# ECE: 0.11059543  ACE: 0.12670752  MCE: 0.21886260  LogLoss: 0.61584982
# Calibrator saved to ./models/pointpillar5/ts.pt
python train_calibrator.py --model-dir ./models/pointpillar5/ --method dbs
# ECE: 0.01558790  ACE: 0.01708735  MCE: 0.05004315  LogLoss: 0.57542663
# Calibrator saved to ./models/pointpillar5/dbs.pt

python get_data.py --model-dir ./models/second5
# Calibration data saved to ./models/second5/calibration_data.npz.
python train_calibrator.py --model-dir ./models/second5/ --method ps
# Confidence metrics before calibration (the smaller the better)
# ECE: 0.44418360  ACE: 0.44009368  MCE: 0.47569744  LogLoss: 0.92116294
# Confidence metrics after calibration
# ECE: 0.00512186  ACE: 0.00520094  MCE: 0.01680084  LogLoss: 0.47760822
# Calibrator saved to ./models/second5/ps.pt
python train_calibrator.py --model-dir ./models/second5/ --method ts
# ECE: 0.04679998  ACE: 0.04522248  MCE: 0.10824797  LogLoss: 0.48765903
# Calibrator saved to ./models/second5/ts.pt
python train_calibrator.py --model-dir ./models/second5/ --method dbs
# ECE: 0.00419671  ACE: 0.00472631  MCE: 0.01707935  LogLoss: 0.47747335
# Calibrator saved to ./models/second5/dbs.pt

###############################################################################
# Homo: our method (DBS + PSA)
###############################################################################
python main.py --model-dir ./models/pointpillar30 --calibrator dbs --aggregator psa  --threshold 0.8
# The Average Precision at IOU 0.3 is 0.875, The Average Precision at IOU 0.5 is 0.873, The Average Precision at IOU 0.7 is 0.813

###############################################################################
# Hetero1: our method (DBS + PSA)
###############################################################################
python main.py --model-dir ./models/pointpillar30 --calibrator dbs --aggregator psa --cav-model-dir ./models/pointpillar5 --threshold 0.8
# The Average Precision at IOU 0.3 is 0.846, The Average Precision at IOU 0.5 is 0.842, The Average Precision at IOU 0.7 is 0.750

###############################################################################
# Hetero2: our method (DBS + PSA)
###############################################################################
python main.py --model-dir ./models/pointpillar30 --calibrator dbs --aggregator psa --cav-model-dir ./models/second5 --threshold 0.8
# The Average Precision at IOU 0.3 is 0.878, The Average Precision at IOU 0.5 is 0.875, The Average Precision at IOU 0.7 is 0.784

###############################################################################
# Ablation Study: Hetero1 DBS
###############################################################################
python main.py --model-dir ./models/pointpillar30 --calibrator dbs --aggregator nms --cav-model-dir ./models/pointpillar5 --threshold 0.8
# The Average Precision at IOU 0.3 is 0.846, The Average Precision at IOU 0.5 is 0.842, The Average Precision at IOU 0.7 is 0.734

###############################################################################
# Ablation Study: Hetero2 DBS
###############################################################################
python main.py --model-dir ./models/pointpillar30 --calibrator dbs --aggregator nms --cav-model-dir ./models/second5 --threshold 0.8
# The Average Precision at IOU 0.3 is 0.878, The Average Precision at IOU 0.5 is 0.875, The Average Precision at IOU 0.7 is 0.776
