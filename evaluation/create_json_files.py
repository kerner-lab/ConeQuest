
import json
import numpy as np
import os

def save_json_files(gt_bboxes, pred_bboxes, output, prediction, filename, training_type):

    ### Save prediction files
    bbox_list, confidence_score_list = [], []
    for index in range(len(pred_bboxes)):
        each_bbox = pred_bboxes[index]
        bbox_list.append(each_bbox)
        prob_list = []
        for i in range(each_bbox[0], each_bbox[2]):
            for j in range(each_bbox[1], each_bbox[3]):
                if prediction[j, i] == 1:
                    prob_list.append(output[i, j])
        confidence_score = sum(prob_list)/len(prob_list)
        confidence_score_list.append(confidence_score)

    if os.path.exists(f"results/predicted_boxes_{training_type}.json"):
        f_pred = open(f"results/predicted_boxes_{training_type}.json")
        json_data = json.load(f_pred)
        json_data[filename] =  {
            "boxes": np.array(bbox_list).tolist(),
            "scores": confidence_score_list
        }
        f_pred.close()
        json_object = json.dumps(json_data, indent=4)
        with open(f"results/predicted_boxes_{training_type}.json", "w") as outfile:
            outfile.write(json_object)
    else:
        json_data =  {
            filename: {
                "boxes": np.array(bbox_list).tolist(),
                "scores": confidence_score_list
            }
        }
        json_object = json.dumps(json_data, indent=4)
        with open(f"results/predicted_boxes_{training_type}.json", "w") as outfile:
            outfile.write(json_object)


    ### Save ground truth files
    bbox_list = []
    for index in range(len(gt_bboxes)):
        each_bbox = gt_bboxes[index]
        bbox_list.append(each_bbox)

    if os.path.exists(f"results/ground_truth_boxes_{training_type}.json"):
        f_target = open(f"results/ground_truth_boxes_{training_type}.json")
        json_data = json.load(f_target)
        json_data[filename] = np.array(bbox_list).tolist()
        f_target.close()
        json_object = json.dumps(json_data, indent=4)
        with open(f"results/ground_truth_boxes_{training_type}.json", "w") as outfile:
            outfile.write(json_object)
    else:
        json_data =  {
            filename: np.array(bbox_list).tolist()
        }
        json_object = json.dumps(json_data, indent=4)
        with open(f"results/ground_truth_boxes_{training_type}.json", "w") as outfile:
            outfile.write(json_object)
