import argparse
import os
import sys
import torch

from data_processing import data_preprocessing
from model_training import training
from testing import testing


### Ignore warnings
import warnings
warnings.filterwarnings("ignore")


### Parse command line arguments
argparser = argparse.ArgumentParser()
argparser.add_argument("--if_training", default=False, required=False, action="store_true")
argparser.add_argument("--if_testing", default=False, required=False, action="store_true")
argparser.add_argument("--if_training_positives", default=False, required=False, action="store_true",
                       help="True value of this parameter will do training only on positive samples from training data list")

argparser.add_argument("--data_dir", type=str, default="data", required=False)
argparser.add_argument("--training_type", type=int, default=1, required=True,
                       help="Provie 1 for region-wise training (BM-1) and 2 for size-wise training (BM-2)")
argparser.add_argument("--training_data_list", "--list", type=str, default=None, required=True,
                       help="Provide a list of training configurations; whether single-region, multi-region, single-size or multi-size")

argparser.add_argument("--validation_split", type=float, default=0.1, required=False)
argparser.add_argument("--testing_split", type=float, default=0.2, required=False)
argparser.add_argument("--train_model", type=str, default="U-Net", required=True)
argparser.add_argument("--backbone", type=str, default="resnet101", required=False)
argparser.add_argument("--backbone_weight", type=str, default="imagenet", required=False)
argparser.add_argument("--num_epochs", type=int, default=200, required=False)
argparser.add_argument("--batch_size", type=int, default=8, required=False)
argparser.add_argument("--extra_info", type=str, default="", required=False,
                       help="Provide any additional information which will be appended after the folder name where models are getting saved")

argparser.add_argument("--eval_data", type=str, default="", required=False,
                       help="Specify which region or size to evaluate")
argparser.add_argument("--testing_epoch", type=str, default="best_epoch", required=False,
                       help="Specify which trained checkpoint to use, whether last_epoch or best_epoch as training uses early stopping")

argparser.add_argument("--wandb_enabled", default=False, required=False, action="store_true",
                       help="True value of this parameter assumes that you have wandb account")
argparser.add_argument("--wandb_entity", type=str, default="", required=False,
                        help="Provide Wandb entity where plots will be available")
argparser.add_argument("--wandb_project", type=str, default="ConeQuest", required=False,
                        help="Provide Wandb project name for plots")


args = argparser.parse_args().__dict__

if_training = args["if_training"]
if_testing = args["if_testing"]
if_training_positives = args["if_training_positives"]
data_dir = args["data_dir"]
training_type = args["training_type"]
training_data_list = [item for item in args["training_data_list"].split(', ')]
validation_split = args["validation_split"]
testing_split = args["testing_split"]
train_model = args["train_model"]
backbone = args["backbone"]
backbone_weight = args["backbone_weight"]
num_epochs = args["num_epochs"]
batch_size = args["batch_size"]
extra_info = ("_" + args["extra_info"] if args["extra_info"] != "" else args["extra_info"]) + ("_positive_only" if if_training_positives else "")
eval_data = args["eval_data"]
testing_epoch = args["testing_epoch"]
wandb_enabled = args["wandb_enabled"]
wandb_entity = args["wandb_entity"]
wandb_project = args["wandb_project"]


### Check device type
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### Initializing data and model directory, unique name of current run
name_of_run = "_".join([(each_region.lower()).replace(" ", "_") for each_region in training_data_list]) + extra_info
output_dir = os.path.join("models", f"benchmark_{training_type}", train_model, name_of_run)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


### Training
if if_training:

    ### Data Preparation
    train_dl, val_dl = data_preprocessing(
        data_dir=data_dir,
        training_type=training_type,
        training_data_list=training_data_list,
        if_training=True,
        if_testing=False,
        if_training_positives=if_training_positives,
        eval_data=None,
        validation_split=validation_split,
        testing_split=testing_split,
        batch_size=batch_size
    )
    print("Data preparation done.")

    ### Model Training
    training(
        train_dl=train_dl,
        val_dl=val_dl,
        device=device,
        name_of_run=name_of_run,
        batch_size=batch_size,
        output_dir=output_dir,
        train_model=train_model,
        backbone=backbone,
        backbone_weight=backbone_weight,
        num_epochs=num_epochs,
        wandb_enabled=wandb_enabled,
        wandb_entity=wandb_entity,
        wandb_project=wandb_project
    )
    print("Training is done successfully and saved model at", output_dir)


### Model evaluation
if if_testing:

    print("\nTraining data list:", args["training_data_list"], "| Testing data:", eval_data, "| Train model:", train_model, "| Testing epoch:", testing_epoch)

    if eval_data == "":
        print("Provide either region (BM-1) or size (for BM-2) on which model will be evaluated for", name_of_run, "training.")
        sys.exit()

    ### Data Preparation
    test_dl = data_preprocessing(
        data_dir=data_dir,
        training_type=training_type,
        training_data_list=training_data_list,
        if_training=False,
        if_testing=True,
        if_training_positives=None,
        eval_data=eval_data,
        validation_split=validation_split,
        testing_split=testing_split,
        batch_size=batch_size
    )
    eval_data = (eval_data.lower()).replace(" ", "_")

    ### Testing
    testing(
        test_dl=test_dl,
        device=device,
        train_model=train_model,
        training_type=training_type,
        eval_data=eval_data,
        output_dir=output_dir,
        name_of_run=name_of_run,
        testing_epoch=testing_epoch,
        extra_info=extra_info
    )

    print("-"*50)
