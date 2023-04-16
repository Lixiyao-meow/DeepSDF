'''
This file is for evaluating the prediction of the SDF decoder. 
We compare the TP, TN, FP, FN of the prediction with the ground truth.
'''

def eval_ROC(gt_sdf, pred_sdf):

    TP, TN, FP, FN = 0, 0, 0, 0
    n = gt_sdf.shape[0]

    for i in range(n):
        if gt_sdf[i] > 0 and pred_sdf[i] > 0:
            TP += 1
        elif gt_sdf[i] > 0 and pred_sdf[i] < 0:
            FN += 1
        elif gt_sdf[i] < 0 and pred_sdf[i] < 0:
            TN += 1
        else:
            FP += 1

    return [TP, TN, FP, FN]