from bwtocolor.model_utils import predict_image, prepare_image
from bwtocolor.pytorchmodel import MainModel
from typing import Any
import cv2 as cv
import numpy as np
from pathlib import Path
import torch
import streamlit as st

parent = Path(__file__).parents[1]
prototxt = str(parent / "model" / "colorization_deploy_v2.prototxt")
caffemodel = str(parent / "model" / "colorization_release_v2.caffemodel")
np_hull = str(parent / "model" / "pts_in_hull.npy")
own_model_path = str(parent / "model" / "v2_gan")


def load_caffe() -> Any:
    """Load the Caffe model and according data, return the finished model

    Returns:
        model: Initialized caffe model (I think its a pytorch one as well?)
    """
    model = cv.dnn.readNetFromCaffe(prototxt, caffemodel)
    pts = np.load(np_hull)
    # add the cluster centers as 1x1 convolutions to the model
    class8 = model.getLayerId("class8_ab")
    conv8 = model.getLayerId("conv8_313_rh")
    pts = pts.transpose().reshape(2, 313, 1, 1)
    model.getLayer(class8).blobs = [pts.astype("float32")]
    model.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]
    return model


def predict_caffe(image: Any) -> Any:
    """ Convert image to Lab, scale to model dims, pass through model.
    Combine predicted ab with original L, convert to RGB and scale back.

    Args:
        image (Any): Image to colorize, must be RGB format

    Returns:
        Image: Predicted / Colorized RGB image
    """
    model = load_caffe()
    scaled = image.astype("float32") / 255.0
    lab = cv.cvtColor(scaled, cv.COLOR_RGB2LAB)
    resized = cv.resize(lab, (224, 224))  # resize to model dimensions
    L = cv.split(resized)[0]  # extract L from LAB
    L -= 50  # mean centering
    model.setInput(cv.dnn.blobFromImage(L))
    ab = model.forward()[0, :, :, :].transpose((1, 2, 0))
    ab = cv.resize(ab, (image.shape[1], image.shape[0]))  # resize prediction back to img props
    # get original L layer, add a and b to it, convert to rgb
    L = cv.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    colorized = cv.cvtColor(colorized, cv.COLOR_LAB2RGB)
    colorized = np.clip(colorized, 0, 1)
    # convert from float range (0,1) to uint (0, 255)
    colorized = (255 * colorized).astype("uint8")
    return colorized


def load_own() -> MainModel:
    """Load the GAN + L1 Loss Model and initialize trained weights

    Returns:
        pytorch model: Initialized pytorch model
    """
    model = MainModel()
    model.load_state_dict(torch.load(own_model_path, map_location=torch.device('cpu')))
    return model


def predict_own(image: Any) -> Any:
    """predict from the given image (RGB format, only needs grey color)
    Takes original dimensions and L layer, resize predicted ab Layer to origin
    Concatenates L origin with ab prediction to get orginial dimensions

    Args:
        image (Any): Image to colorize, must be RGB format

    Returns:
        Image: Predicted / Colorized RGB image
    """
    model = load_own()
    original_dims = (image.shape[1], image.shape[0])
    L_orig = cv.split(cv.cvtColor(image, cv.COLOR_RGB2LAB))[0]
    input = prepare_image(image)
    prediction = predict_image(model, input)
    prediction = (255 * prediction).astype("uint8")
    pred_resized = cv.resize(prediction, original_dims)
    _, a_pred, b_pred = cv.split(cv.cvtColor(pred_resized, cv.COLOR_RGB2LAB))
    final_lab = cv.merge((L_orig, a_pred, b_pred))
    final_rgb = cv.cvtColor(final_lab, cv.COLOR_LAB2RGB)
    return final_rgb


@st.cache()
def colorize_image(image: Any, use_caffe: bool) -> Any:
    """Colorize the given image

    Args:
        image (Any): Image to colorize, must be RGB format
        use_caffe (bool): If to use the caffe model

    Returns:
        Image: Predicted / Colorized RGB image
    """
    if use_caffe:
        return predict_caffe(image)
    return predict_own(image)
