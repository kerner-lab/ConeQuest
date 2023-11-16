
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


def iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score


def calculate_iou_matrix(target_masks, predicted_masks):

    num_target_masks = len(target_masks)
    num_predicted_masks = len(predicted_masks)


    iou_matrix = np.zeros((num_target_masks, num_predicted_masks))
    for i in range(num_target_masks):
        for j in range(num_predicted_masks):
            iou_matrix[i, j] = iou(target_masks[i], predicted_masks[j])

    return iou_matrix


def calculate_iou(target_masks, predicted_masks):

    iou_matrix = calculate_iou_matrix(target_masks, predicted_masks)
    # Use the Hungarian algorithm to find the best assignment
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)

    if len(row_ind) == 0:
        return 0, 0

    # print("row_ind: ", row_ind)
    # print("col_ind: ", col_ind)

    total_iou = 0.0
    for i, j in zip(row_ind, col_ind):
        total_iou += iou_matrix[i, j]

    average_iou = total_iou / len(row_ind)
    return average_iou, iou_matrix[row_ind, col_ind]


def separate_instances(input_array):

  # Find contours of objects in the image
  thresh = cv2.threshold(input_array, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY)[1]
  cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  contours = cnts[0] if len(cnts) == 2 else cnts[1]

  # Initialize an empty list to store individual object images
  individual_images = []

  # Loop through the detected contours (objects)
  for contour in contours:

      # Create a mask for the current object
      mask = np.zeros_like(input_array)
      cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

      # Extract the object using the mask
      individual_object = cv2.bitwise_and(input_array, mask)

      # Append the individual object to the list
      individual_images.append(individual_object)

  instance_mask = np.array(individual_images)
  return instance_mask

