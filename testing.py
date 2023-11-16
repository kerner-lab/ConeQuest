import cv2
import numpy as np
import os
import pandas as pd
import segmentation_models_pytorch as smp
import sys
from tqdm import tqdm

from evaluation.bounding_box import mask_to_boxes
from evaluation.calculate_metrics import *
from evaluation.create_json_files import save_json_files

import torch
from torch.autograd import Variable


random_seed = 42

def testing(
    test_dl,
    device,
    train_model: str,
    training_type: str,
    eval_data: str,
    output_dir: str,
    name_of_run: str,
    testing_epoch: str,
    extra_info: str
):

    ### Create prediction and result directory
    prediction_dir = os.path.join(output_dir, "prediction", eval_data)
    if not os.path.exists(prediction_dir):
        os.makedirs(prediction_dir)

    result_dir = os.path.join("results")
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    ### Load model
    print("Evaluating on model -->", os.path.join(output_dir, f"{testing_epoch}.pth"))
    model_path = os.path.join(output_dir, f"{testing_epoch}.pth")
    if not os.path.exists(model_path):
        print(model_path, "does not exist...!!")
        print("-"*50)
        sys.exit()

    model = torch.load(model_path, map_location='cpu')
    model.to(device)
    model.eval()

    ### Initialize threshold and metrics list
    threhsold = 0.5

    pixel_iou, pixel_accuracy, pixel_dice, pixel_recall, pixel_precision, object_iou, object_accuracy, object_precision, object_recall, pq_list, mask_iou_list, pixel_accuracy_no_cone, area_no_cone = [], [], [], [], [], [], [], [], [], [], [], [], []

    for _, (input, label, filename) in enumerate(tqdm(test_dl)):

        input = Variable(input.type(torch.FloatTensor)).to(device)
        label = Variable(label.type(torch.FloatTensor)).to(device)

        output = model(input).sigmoid()

        ### Extract 4 coordinates of confusion matrix
        tp, fp, fn, tn = smp.metrics.get_stats(output, label.type(torch.int64), mode='binary', threshold=0.5)

        ### Extract bounding box
        label = label.squeeze(dim=1).cpu().numpy()[0, :, :].astype(np.uint8)
        output = output.squeeze(dim=1).cpu().detach().numpy()[0, :, :]
        prediction = (output > threhsold).astype(np.uint8)

        gt_bboxes = mask_to_boxes(label)
        pred_bboxes = mask_to_boxes(prediction)

        ### Save JSON file which sores deatils of prediction and groud truth bounding box (used to calculate mAP)
        save_json_files(gt_bboxes, pred_bboxes, output, prediction, filename[0], training_type)

        ### Save prediction of each test patch
        cv2.imwrite(os.path.join(prediction_dir, filename[0]), prediction)

        if len(gt_bboxes) != 0:

            ### Pixel-wise metrics
            pixel_accuracy.append(smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro").item())
            pixel_recall.append(smp.metrics.recall(tp, fp, fn, tn, reduction="macro").item())
            pixel_precision.append(smp.metrics.precision(tp, fp, fn, tn, reduction="macro").item())
            pixel_iou.append(smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro").item())
            pixel_dice.append(compute_dice_coff(tp, fp, fn).item())

            mask_iou_list.append(compute_mask_iou(label, prediction))

            ### Object-wise metrics
            object_iou.append(match_bboxes(gt_bboxes, pred_bboxes))

            current_accuracy, current_precision, current_recall = compute_object_metrics(gt_bboxes, pred_bboxes, 0.5)
            object_accuracy.append(current_accuracy)
            object_precision.append(current_precision)
            object_recall.append(current_recall)

            pq_list.append(compute_panoptic_quality(gt_bboxes, pred_bboxes, 0.5))

            # print(filename[0], "- Cone -> Object_IoU:", object_iou[-1], "Object_Accuracy:", object_accuracy[-1], "Object_Precision:", object_precision[-1],  "Object_Recall:", object_recall[-1],\
            #       "| Pixel_IoU:", pixel_iou[-1], "Pixel_Dice:", pixel_dice[-1], "Pixel_Accuracy:", pixel_accuracy[-1], "Pixel_Recall:", pixel_recall[-1], "Pixel_Precision:", pixel_precision[-1])

        else:

            ### Pixel-wise metrics
            pixel_accuracy_no_cone.append(smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro").item())

            ### Identify the area of wrong prediction
            total_cone_area = compute_negative_area(prediction)
            area_no_cone.append(total_cone_area)
            # print(filename[0], "- Non-cone -> Area:", area_no_cone[-1], "Pixel_Accuracy:", pixel_accuracy_no_cone[-1])


    map_1, map_95 = compute_map(training_type)
    mask_iou_mean, pixel_iou_mean, pixel_dice_mean, pixel_accuracy_mean, pixel_recall_mean, pixel_precision_mean, pq_mean, object_iou_mean, object_accuracy_mean, object_precision_mean, object_recall_mean, area_no_cone_mean = np.mean(mask_iou_list)*100, np.mean(pixel_iou)*100, np.mean(pixel_dice)*100, np.mean(pixel_accuracy)*100, np.mean(pixel_recall)*100, np.mean(pixel_precision)*100, np.mean(pq_list)*100, np.mean(object_iou)*100, np.mean(object_accuracy)*100, np.mean(object_precision)*100, np.mean(object_recall)*100, np.mean(area_no_cone)*100


    ### Save results in CSV file
    result_csv_path = os.path.join("results", f"result{extra_info}.csv")
    if os.path.exists(result_csv_path):
        result_df = pd.read_csv(result_csv_path)
    else:
        result_df = pd.DataFrame(columns=["Benchmark", "Training Region", "Testing Region", "Training Model", "Testing epoch", "Mask IoU", "Pixel IoU",  "Pixel Accuracy", "Pixel Precision", "Pixel Recall", "Panoptic Quality", "mAP", "Object IoU", "Object Accuracy", "Object Precision", "Object Recall", "Area"])

    current_result = [training_type, name_of_run, eval_data, train_model, testing_epoch, f"{format(mask_iou_mean, '.4f')}", f"{format(pixel_iou_mean, '.4f')}", f"{format(pixel_accuracy_mean, '.4f')}", f"{format(pixel_precision_mean, '.4f')}", f"{format(pixel_recall_mean, '.4f')}", f"{format(pq_mean, '.4f')}", f"{format(map_95, '.4f')}", f"{format(object_iou_mean, '.4f')}", f"{format(object_accuracy_mean, '.4f')}", f"{format(object_precision_mean, '.4f')}", f"{format(object_recall_mean, '.4f')}", f"{format(area_no_cone_mean, '.4f')}"]
    result_df.loc[len(result_df)] = current_result
    result_df.to_csv(result_csv_path, index=False)


    ### Print results
    print(f"Cone -->\nMask_IoU: {format(mask_iou_mean, '.4f')} \
            \nPixel_IoU: {format(pixel_iou_mean, '.4f')} \
            \nPixel_Dice: {format(pixel_dice_mean, '.4f')} \
            \nmAP (0-1): {format(map_1, '.4f')} \
            \nmAP (0.5-0.95): {format(map_95, '.4f')} \
            \nPixel_Accuracy: {format(pixel_accuracy_mean, '.4f')} \
            \nPixel_Recall: {format(pixel_recall_mean, '.4f')} \
            \nPixel_Precision: {format(pixel_precision_mean, '.4f')} \
            \nPanoptic_Quality: {format(pq_mean, '.4f')} \
            \nObject_IoU: {format(object_iou_mean, '.4f')} \
            \nObject_Accuracy: {format(object_accuracy_mean, '.4f')} \
            \nObject_Precision: {format(object_precision_mean, '.4f')} \
            \nObject_Recall: {format(object_recall_mean, '.4f')}")

    print(f"Non-cone -->\nArea: {format(area_no_cone_mean, '.4f')}")


    ### Remove created JSON files
    if os.path.exists(f"results/ground_truth_boxes_{training_type}.json"):
        os.remove(f"results/ground_truth_boxes_{training_type}.json")
    if os.path.exists(f"results/predicted_boxes_{training_type}.json"):
        os.remove(f"results/predicted_boxes_{training_type}.json")
