### Initializing model
import segmentation_models_pytorch as smp


def load_model(
    train_model: str,
    backbone: str,
    backbone_weight: str
):

    model_choices = ["U-Net", "DeepLab", "FPN", "U-Net++", "MANet", "PSPNet", "DeepLab+"]

    if train_model == model_choices[0]:
        model = smp.Unet(
            encoder_name=backbone,
            encoder_weights=backbone_weight,
            in_channels=1,
            classes=1,
        )

    if train_model == model_choices[1]:
        model = smp.DeepLabV3(
            encoder_name=backbone,
            encoder_weights=backbone_weight,
            in_channels=1,
            classes=1,
        )

    if train_model == model_choices[2]:
        model = smp.FPN(
            encoder_name=backbone,
            encoder_weights=backbone_weight,
            in_channels=1,
            classes=1,
        )

    if train_model == model_choices[3]:
        model = smp.UnetPlusPlus(
            encoder_name=backbone,
            encoder_weights=backbone_weight,
            in_channels=1,
            classes=1,
        )

    if train_model == model_choices[4]:
        model = smp.MAnet(
            encoder_name=backbone,
            encoder_weights=backbone_weight,
            in_channels=1,
            classes=1,
        )

    if train_model == model_choices[5]:
        model = smp.PSPNet(
            encoder_name=backbone,
            encoder_weights=backbone_weight,
            in_channels=1,
            classes=1,
        )

    if train_model == model_choices[6]:
        model = smp.DeepLabV3Plus(
            encoder_name=backbone,
            encoder_weights=backbone_weight,
            in_channels=1,
            classes=1,
        )

    return model
