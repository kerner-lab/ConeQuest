from matplotlib import pyplot as plt
import os
import segmentation_models_pytorch as smp
from tqdm import tqdm
import wandb

from model import load_model

import torch
from torch.autograd import Variable


def training(
    train_dl,
    val_dl,
    device,
    name_of_run: str,
    batch_size: int,
    output_dir: str,
    train_model: str,
    backbone: str,
    backbone_weight: str,
    num_epochs: int,
    wandb_enabled: bool,
    wandb_entity: str,
    wandb_project: str
):

    model = load_model(train_model, backbone, backbone_weight)
    model.to(device)

    loss = smp.losses.SoftBCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters())

    ### Check and set wandb
    if wandb_enabled:
        import wandb
        wandb.init(
            entity=wandb_entity,
            project=wandb_project,
            name=f"{train_model}_{name_of_run}",
            config={
                "Training data samples": len(train_dl),
                "Validation data samples": len(val_dl),
                "Model": train_model,
                "Which run": name_of_run,
                "Epochs": num_epochs,
                "Loss": loss,
                "Batch size": batch_size,
                "Optimizer": optimizer,
                "Backbone": backbone,
                "Model path": output_dir,
            }
        )

    ### Model training
    patience, epochs_since_improvement, best_val_loss, best_val_iou = 10, 0, float('inf'), -float('inf')

    with tqdm(range(num_epochs), desc="Epoch") as tqdm_epoch:

        for epoch in tqdm_epoch:

            train_loss, train_iou, train_accuracy, train_precision, train_recall, val_loss, val_iou, val_accuracy, val_precision, val_recall = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

            model.train()
            for _, (image, label, _) in enumerate(train_dl):
                image = Variable(image.type(torch.FloatTensor)).to(device)
                label = Variable(label.type(torch.FloatTensor)).to(device)
                optimizer.zero_grad()
                output = model(image)
                loss_value = loss(output, label)
                loss_value.backward()
                optimizer.step()
                current_loss = loss_value.item()

                tp, fp, fn, tn = smp.metrics.get_stats(output, label.type(torch.int64), mode='binary', threshold=0.5)
                current_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise").item()
                current_accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro").item()
                current_precision = smp.metrics.precision(tp, fp, fn, tn, reduction="macro").item()
                current_recall = smp.metrics.recall(tp, fp, fn, tn, reduction="macro").item()

                if wandb_enabled:
                    wandb.log(
                        {
                            "Training iteration loss": current_loss,
                            "Training iteration IoU": current_iou,
                            "Training iteration Accuracy": current_accuracy,
                            "Training iteration Precision": current_precision,
                            "Training iteration Recall": current_recall,
                        }
                    )

                train_loss += current_loss
                train_iou += current_iou
                train_accuracy += current_accuracy
                train_precision += current_precision
                train_recall += current_recall

            if wandb_enabled:
                wandb.log(
                    {
                        "Training epoch loss": train_loss / len(train_dl),
                        "Training epoch IoU": train_iou / len(train_dl),
                        "Training epoch Accuracy": train_accuracy / len(train_dl),
                        "Training epoch Precision": train_precision / len(train_dl),
                        "Training epoch Recall": train_recall / len(train_dl),
                    }
                )

            model.eval()
            with torch.no_grad():
                for _, (image, label, _) in enumerate(val_dl):
                    image = Variable(image.type(torch.FloatTensor)).to(device)
                    label = Variable(label.type(torch.FloatTensor)).to(device)
                    output = model(image)
                    loss_value = loss(output, label)
                    current_loss = loss_value.item()

                    tp, fp, fn, tn = smp.metrics.get_stats(output, label.type(torch.int64), mode='binary', threshold=0.5)
                    current_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise").item()
                    current_accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro").item()
                    current_precision = smp.metrics.precision(tp, fp, fn, tn, reduction="macro").item()
                    current_recall = smp.metrics.recall(tp, fp, fn, tn, reduction="macro").item()

                    if wandb_enabled:
                        wandb.log(
                            {
                                "Validation iteration loss": current_loss,
                                "Validation iteration IoU": current_iou,
                                "Validation iteration Accuracy": current_accuracy,
                                "Validation iteration Precision": current_precision,
                                "Validation iteration Recall": current_recall,
                            }
                        )

                    val_loss += current_loss
                    val_iou += current_iou
                    val_accuracy += current_accuracy
                    val_precision += current_precision
                    val_recall += current_recall

                if wandb_enabled:
                    wandb.log(
                        {
                            "Validation epoch loss": val_loss / len(val_dl),
                            "Validation epoch IoU": val_iou / len(val_dl),
                            "Validation epoch Accuracy": val_accuracy / len(val_dl),
                            "Validation epoch Precision": val_precision / len(val_dl),
                            "Validation epoch Recall": val_recall / len(val_dl),
                        }
                    )


            ### Check for the best epoch
            if (val_loss/len(val_dl)) < best_val_loss and (val_iou/len(val_dl)) > best_val_iou:
                best_epoch = epoch
                torch.save(model, os.path.join(output_dir, "best_epoch.pth"))
                best_val_loss = val_loss/len(val_dl)
                best_val_iou = val_iou/len(val_dl)
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1
                ''' Uncomment below 3 lines, if you want to do early stopping'''
                # if epochs_since_improvement >= patience:
                #     print("Early stopping!")
                #     break

            print(f"Epoch {epoch+1}/{num_epochs} --> train_loss: {train_loss/len(train_dl):.4f}, val_loss: {val_loss/len(val_dl):.4f} | train_iou: {train_iou/len(train_dl):.4f}, val_iou: {val_iou/len(val_dl):.4f} | train_accuracy: {train_accuracy/len(train_dl):.4f}, val_accuracy: {val_accuracy/len(val_dl):.4f} | train_precision: {train_precision/len(train_dl):.4f}, val_precision: {val_precision/len(val_dl):.4f} | train_recall: {train_recall/len(train_dl):.4f}, val_recall: {val_recall/len(val_dl):.4f}")

    torch.save(model, os.path.join(output_dir, "last_epoch.pth"))

    if wandb_enabled:
        wandb.config.update({"Best epoch": best_epoch})
        wandb.finish()
