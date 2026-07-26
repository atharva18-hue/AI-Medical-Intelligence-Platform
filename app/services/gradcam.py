"""
Grad-CAM implementation for explainability.
Reference: Selvaraju et al. - Grad-CAM paper
Had to debug this a lot during project...
"""

import io
import base64
import numpy as np
import cv2
import torch
from PIL import Image


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # register hooks
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        self.model.zero_grad()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # backward pass for target class
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        # global average pooling on gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)

        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)  # normalize

        return cam

    def overlay_on_image(self, original_img: Image.Image, cam, alpha=0.4):
        # resize cam to match image
        img_array = np.array(original_img.resize((224, 224)))
        cam_resized = cv2.resize(cam, (224, 224))

        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        overlay = (alpha * heatmap + (1 - alpha) * img_array).astype(np.uint8)
        return Image.fromarray(overlay)


def generate_gradcam_image(model, input_tensor, original_img, target_layer):
    """Helper function used by API"""
    gradcam = GradCAM(model, target_layer)
    cam = gradcam.generate(input_tensor)

    overlay = gradcam.overlay_on_image(original_img, cam)

    # convert to base64 for sending to frontend
    buf = io.BytesIO()
    overlay.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return b64
