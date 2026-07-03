import io
import base64
import cv2
import numpy as np
import torch
from PIL import Image

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


def reshape_transform(tensor):
    """
    Converts ViT output tokens to feature maps.
    Input:
        [B, 197, C]
    Output:
        [B, C, 14, 14]
    """
    tensor = tensor[:, 1:, :]
    tensor = tensor.reshape(
        tensor.size(0),
        14,
        14,
        tensor.size(2)
    )
    tensor = tensor.permute(0, 3, 1, 2)
    return tensor


def get_attention_base64(model, input_tensor, original_image):
    """
    Generate Grad-CAM visualization for Vision Transformer.
    """

    model.eval()

    target_layers = [
        model.model.blocks[-1].norm1
    ]

    cam = GradCAM(
        model=model,
        target_layers=target_layers,
        reshape_transform=reshape_transform,
    )

    grayscale_cam = cam(
        input_tensor=input_tensor
    )[0]

    rgb_img = np.array(
        original_image.resize((224, 224))
    ).astype(np.float32) / 255.0

    visualization = show_cam_on_image(
        rgb_img,
        grayscale_cam,
        use_rgb=True
    )

    visualization = cv2.cvtColor(
        visualization,
        cv2.COLOR_RGB2BGR
    )

    visualization = Image.fromarray(visualization)

    buffer = io.BytesIO()
    visualization.save(buffer, format="PNG")

    return base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")