from bwtocolor.pytorchmodel import MainModel
from typing import Any, List
from PIL import Image
from torch.functional import Tensor
from torchvision import transforms
import numpy as np
from skimage.color import rgb2lab, lab2rgb
import torch

SIZE = 256


def lab_to_rgb(L: Tensor, ab: Tensor) -> List[Any]:
    """Takes a batch of images L and ab Layer and converts them to rbg"""
    L = (L + 1.) * 50.
    ab = ab * 110.
    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()
    rgb_imgs = []
    for img in Lab:
        img_rgb = lab2rgb(img)
        rgb_imgs.append(img_rgb)
    return np.stack(rgb_imgs, axis=0)


def prepare_image(image: Any) -> dict[str, Tensor]:
    """Prepare the image for the model
    Reizes to the model Size, returns L and ab as tensors (only L for pred needed)
    """
    tf = transforms.Resize((SIZE, SIZE), Image.BICUBIC)
    img = Image.fromarray(image).convert("RGB")
    img = tf(img)
    img = np.array(img)
    img_lab = rgb2lab(img).astype("float32")  # Converting RGB to L*a*b
    img_lab = transforms.ToTensor()(img_lab)
    L = img_lab[[0], ...] / 50. - 1.  # Between -1 and 1
    ab = img_lab[[1, 2], ...] / 110.  # Between -1 and 1
    return {'L': L.unsqueeze(0), 'ab': ab.unsqueeze(0)}


def predict_image(model: MainModel, data: dict[str, Tensor]) -> Any:
    """Uses the model to predicts the image
    needs the data format defined above ind prepare_image
    returns predicted rgb
    """
    model.net_G.eval()
    with torch.no_grad():
        model.setup_input(data)
        model.forward()
    model.net_G.train()
    fake_color = model.fake_color.detach()
    L = model.L
    fake_imgs = lab_to_rgb(L, fake_color)
    pred = fake_imgs[0]

    return pred
