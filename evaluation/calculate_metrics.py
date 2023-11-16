import json
import numpy as np
import scipy.optimize

from evaluation.map import *
from evaluation.mask_iou import *


def bbox_iou(boxA, boxB):

  # --------------------------------------------------------
  # References:
  # bbox_iou_evaluation: https://gist.github.com/AruniRC/c629c2df0e68e23aff7dcaeef87c72d4
  # --------------------------------------------------------

    # Determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = xB - xA + 1
    interH = yB - yA + 1

    if interW <=0 or interH <=0 :
        return -1.0

    interArea = interW * interH
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


''' Computer object IoU '''
def match_bboxes(bbox_gt, bbox_pred, IOU_THRESH=0.5):

    '''
    Given sets of true and predicted bounding-boxes,
    determine the best possible match.
    Parameters
    -----------
    bbox_gt, bbox_pred : N1x4 and N2x4 np array of bboxes [x1,y1,x2,y2]. 
      The number of bboxes, N1 and N2, need not be the same.
    
    Returns
    --------
    (idxs_true, idxs_pred, ious, labels)
        idxs_true, idxs_pred : indices into gt and pred for matches
        ious : corresponding IOU value of each match
        labels: vector of 0/1 values for the list of detections
    '''

    n_true = bbox_gt.shape[0]
    n_pred = bbox_pred.shape[0]
    MAX_DIST = 1.0
    MIN_IOU = 0.0

    # NUM_GT x NUM_PRED
    iou_matrix = np.zeros((n_true, n_pred))
    for i in range(n_true):
        for j in range(n_pred):
            iou_matrix[i, j] = bbox_iou(bbox_gt[i,:], bbox_pred[j,:])

    if n_pred > n_true:
      # there are more predictions than ground-truth - add dummy rows
      diff = n_pred - n_true
      iou_matrix = np.concatenate((iou_matrix, 
                                  np.full((diff, n_pred), MIN_IOU)), 
                                  axis=0)

    if n_true > n_pred:
      # more ground-truth than predictions - add dummy columns
      diff = n_true - n_pred
      iou_matrix = np.concatenate((iou_matrix, 
                                  np.full((n_true, diff), MIN_IOU)), 
                                  axis=1)

    # call the Hungarian matching
    idxs_true, idxs_pred = scipy.optimize.linear_sum_assignment(1 - iou_matrix)

    if (not idxs_true.size) or (not idxs_pred.size):
        ious = np.array([])
    else:
        ious = iou_matrix[idxs_true, idxs_pred]

    # remove dummy assignments
    sel_pred = idxs_pred<n_pred
    idx_pred_actual = idxs_pred[sel_pred] 
    idx_gt_actual = idxs_true[sel_pred]
    ious_actual = iou_matrix[idx_gt_actual, idx_pred_actual]
    sel_valid = (ious_actual > IOU_THRESH)
    label = sel_valid.astype(int)

    if len(ious_actual[sel_valid]) == 0:
        object_iou = 0
    else:
        object_iou = sum(ious_actual[sel_valid])/len(ious_actual[sel_valid])

    return object_iou


''' Compute Dice Coefficient '''
def compute_dice_coff(tp, fp, fn):
   
   dice_coff = (2 * tp) / ((2* tp) + fp + fn)
   return dice_coff


''' Compute negative pixel area '''
def compute_negative_area(prediction):

    '''
    Given the prediction from model,
    determine the area of predicted cones.
    Parameters
    -----------
    prediction : predicted image of size H x W x C

    Returns
    --------
    (area)
        area : total area (in terms of pixels) of all cones present in prediction
    '''

    pixel_count_1 = np.count_nonzero(prediction == 1)
    total_area = prediction.shape[0] * prediction.shape[1]

    return pixel_count_1/total_area

''' Compute object-wise metrics '''
def compute_object_metrics(gt_bboxes, pred_bboxes, iou_threshold):

    result_dict = get_single_image_results(gt_bboxes, pred_bboxes, iou_threshold)
    tps, fps, fns = result_dict["true_pos"], result_dict["false_pos"], result_dict["false_neg"]

    try:
      object_accuracy = tps/(tps+fps+fns)
      if object_accuracy == -1:
          object_accuracy = 0
    except ZeroDivisionError:
      object_accuracy = 0

    try:
      object_precision = tps/(tps+fps)
      if object_precision == -1:
          object_precision = 0
    except ZeroDivisionError:
      object_precision = 0

    try:
      object_recall = tps/(tps+fns)
      if object_recall == -1:
          object_recall = 0
    except ZeroDivisionError:
      object_recall = 0

    return object_accuracy, object_precision, object_recall


''' Compute mean Average Precision (mAP) '''
def compute_map(training_type):

    with open(f"results/ground_truth_boxes_{training_type}.json") as infile:
        gt_boxes = json.load(infile)

    with open(f"results/predicted_boxes_{training_type}.json") as infile:
        pred_boxes = json.load(infile)

    # Runs it for one IoU threshold
    iou_thr = 0.5
    data = get_avg_precision_at_iou(gt_boxes, pred_boxes, iou_thr=iou_thr)

    avg_precs = []
    iou_thrs = []
    for _, iou_thr in enumerate(np.linspace(0, 1, 11)):
        data = get_avg_precision_at_iou(gt_boxes, pred_boxes, iou_thr=iou_thr)
        avg_precs.append(data['avg_prec'])
        iou_thrs.append(iou_thr)

    avg_precs = [float('{:.4f}'.format(ap)) for ap in avg_precs]
    iou_thrs = [float('{:.4f}'.format(thr)) for thr in iou_thrs]
    map_1 = 100*np.mean(avg_precs)

    for _, iou_thr in enumerate(np.linspace(0.5, 0.95, 10)):
        data = get_avg_precision_at_iou(gt_boxes, pred_boxes, iou_thr=iou_thr)
        avg_precs.append(data['avg_prec'])
        iou_thrs.append(iou_thr)

    avg_precs = [float('{:.4f}'.format(ap)) for ap in avg_precs]
    iou_thrs = [float('{:.4f}'.format(thr)) for thr in iou_thrs]
    map_95 = 100*np.mean(avg_precs)

    return map_1, map_95


''' Compute Panoptic Quality '''
def compute_panoptic_quality(gt_bboxes, pred_bboxes, iou_threshold):

    result_dict, all_ious = get_single_image_results_for_pq(gt_bboxes, pred_bboxes, iou_threshold)
    tps, fps, fns = result_dict["true_pos"], result_dict["false_pos"], result_dict["false_neg"]

    if len(all_ious) == 0:
        return 0

    correct_prediction = [each_iou for each_iou in all_ious if each_iou >= 0.5]
    pq = sum(correct_prediction) / (tps + (0.5 * fps) + (0.5 * fns))

    return pq


''' Computer Mask IoU '''
def compute_mask_iou(ground_truth, prediction):

    gt_mask, pred_mask = separate_instances(ground_truth), separate_instances(prediction)
    mask_iou, _ = calculate_iou(gt_mask, pred_mask)

    return mask_iou