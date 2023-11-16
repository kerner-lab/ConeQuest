''' Training on 1/2 region and testing on that data itself or unseen full data '''

import cv2
import numpy as np
import os
import pandas as pd
from skimage import exposure
from sklearn.model_selection import train_test_split
from typing import List

from torch.utils.data import DataLoader, Dataset


random_seed = 42


class CustomDataset(Dataset):

    def __init__(self, data_list, data_dir):

        self.images = []
        self.labels = []

        for each_file in data_list:
          folder_name = each_file.split("_")[0] + "_" + each_file.split("_")[1]
          input_path = os.path.join(data_dir, folder_name, "input_dir", each_file)
          mask_path = os.path.join(data_dir, folder_name, "output_dir", each_file)
          self.images.append(input_path)
          self.labels.append(mask_path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        label_path = self.labels[idx]
        filename = image_path.split("/")[-1]

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = np.expand_dims(image, axis=0)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        label = np.expand_dims(label, axis=0)

        return image, label, filename


def data_preprocessing(
    data_dir: str,
    training_type: int,
    training_data_list: List[str],
    if_training: bool,
    if_testing: bool,
    if_training_positives: bool,
    eval_data: str,
    validation_split: float = 0.1,
    testing_split: float = 0.2,
    batch_size: int = 32
):

    ### Get data samples for the training regions
    whole_data_df = pd.read_csv(os.path.join(data_dir, "ConeQuest_data.csv"))

    training_list, validation_list = [], []


    if if_training:

        for each_data in training_data_list:

            if training_type == 1:

                train_df = whole_data_df.loc[(whole_data_df["Region"]==each_data) & (whole_data_df["BM-1 Set"]=="train")]
                val_df = whole_data_df.loc[(whole_data_df["Region"]==each_data) & (whole_data_df["BM-1 Set"]=="val")]
            
            if training_type == 2:
                train_df = whole_data_df.loc[(whole_data_df["Size"]==each_data) & (whole_data_df["BM-2 Set"]=="train")]
                val_df = whole_data_df.loc[(whole_data_df["Size"]==each_data) & (whole_data_df["BM-2 Set"]=="val")]

            if if_training_positives:
                train_df = train_df.loc[(train_df["Number of Cones"]!=0)]

            training_list.extend(train_df["Patch Id"].tolist())
            validation_list.extend(val_df["Patch Id"].tolist())

        print("Number of training samples -", len(training_list), "\nNumber of validation samples -", len(validation_list))

        ### Build torch dataset
        train_dataset = CustomDataset(training_list, data_dir)
        val_dataset = CustomDataset(validation_list, data_dir)

        ### Define the dataloader
        n_cpu = os.cpu_count()
        train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_cpu, drop_last=True)
        val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=n_cpu, drop_last=True)

        return train_dl, val_dl

    if if_testing:

        if training_type == 1:
            test_df = whole_data_df.loc[(whole_data_df["Region"]==eval_data) & (whole_data_df["BM-1 Set"]=="test")]

        if training_type == 2:
            test_df = whole_data_df.loc[(whole_data_df["Size"]==eval_data) & (whole_data_df["BM-2 Set"]=="test")]

        testing_list = test_df["Patch Id"].tolist()

        print("Number of testing samples -", len(testing_list))

        ### Build torch dataset
        test_dataset = CustomDataset(testing_list, data_dir)

        ### Define the dataloader
        test_dl = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)

        return test_dl
