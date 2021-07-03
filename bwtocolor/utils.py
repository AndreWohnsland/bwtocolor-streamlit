from typing import Any, AnyStr, Tuple
import streamlit as st
import base64
import numpy as np
import cv2 as cv
from io import BytesIO
from PIL import Image


def picture_compare(p1: Any, p2: Any, h1: str, h2: str):
    """Displays both picutes side by side

    Args:
        p1 (Any): Picture Element
        p2 (Any): Second Pictue Element
        h1 (str): Header first picture
        h2 (str): Header second picture
    """
    col1, col2 = st.beta_columns(2)
    col1.image(p1, width=700, caption=h1)
    col2.image(p2, width=700, caption=h2)


def get_image_download_link(predicted_img: Any, f_name: str = "BWtoColor") -> str:
    """Generates a link allowing the PIL image to be downloaded
    Need to generate image with 
    from PIL import Image
    result = Image.fromarray(predicted_img)

    in:  PIL image
    out: href string
    """
    img = Image.fromarray(predicted_img)
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/jpg;base64,{img_str}" download="{f_name}_colorized.jpg"><input type="button" value="Download Result"></a>'
    return href


def prepare_input_image(uploaded_file: Any) -> Tuple[str, Any]:
    """Extract the filename and the BW layer from the uploaded image

    Args:
        uploaded_file (Any): Uploaded File

    Returns:
        Tuple[str, Any, Any]: Filename, BW layer, format for cv
    """
    file_name = uploaded_file.name.split(".")[0]
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv.imdecode(file_bytes, 1)
    image_bw = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    # cv_image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    return (file_name, image_bw, image)


def generate_information():
    """Just display some basic information for the user."""
    st.header("❓ How to Use?")
    st.write("Just use the sidebar to upload your image, then you can download the colorized version 😄.")
    st.write("""You can also use already colorful pictures and check the according box. 
    The model will convert it to BW and recolor based on the BW. 
    You can then compare the original with the model result""")


def generate_model_information():
    """Display some information about the modell"""
    st.header("⚙️ Model Information")
    st.write("Convolutional Neural Network, trained on RGB / Lab pictures to predict the colors.")
    st.markdown("""
    This model uses L1 Loss combined with adversarial (GAN) loss for the model to predict the colors of the blackwhite image,
    following [this concept](https://github.com/moein-shariatnia/Deep-Learning/tree/main/Image%20Colorization%20Tutorial) and [this paper](https://arxiv.org/pdf/1611.07004.pdf).
    The model was trained over 100 epochs, with about 8.000 image as input. Images were used from [emilwallner](https://www.floydhub.com/emilwallner/datasets/colornet/2/images).
    Using more images, or just doing more epochs (200-500) would propably improve the model by a good factor. 100 epochs took ~16 hours on a Nvidia GTX 1080.

    As a reference the [CV2 adaption](https://docs.opencv.org/3.4/d5/de7/tutorial_dnn_googlenet.html) of the [Caffe model](https://caffe.berkeleyvision.org/) was used.
    If you are interested, go check it out as well!
    """)


def generate_sidebar() -> Tuple[bool, bool, Any]:
    """Generates the sidebar with the option. Returns needed variables

    Returns:
        Tuple[bool, bool, Any]: if caffee is used, if the picture is color, the file to use
    """
    st.sidebar.header("Select Properties & Data")
    st.sidebar.write("We just need some information and your picture to start the action!")
    st.sidebar.subheader("Model")
    selecte_model = st.sidebar.radio("Select Model", ("GAN + U-Net", "Caffe (reference)"))
    use_caffe = selecte_model == "Caffe (reference)"
    st.sidebar.subheader("Properties")
    is_color = st.sidebar.checkbox("This is a color image for later comparision of model grade")
    st.sidebar.subheader("Picture")
    uploaded_file = st.sidebar.file_uploader("Upload your picture", type=["png", "jpg", "jpeg"])
    return (use_caffe, is_color, uploaded_file)


def generate_picture_comparison(image_bw: Any, uploaded_file: Any, predicted_color: Any, is_color: bool):
    """Generates the views of the input and output image.
    If the picture was RGB for reference show both BW and color vs prediction.

    Args:
        image_bw (Any): BW image
        uploaded_file (Any): Original file
        predicted_color (Any): RGB prediction from the model
        is_color (bool): If input was a color picture for reference
    """
    st.header("🖌️ BW and Colorized Picture")
    picture_compare(image_bw, predicted_color, "BW Picture", "Colorized by Model")
    if is_color:
        st.header("📷 Reference Object Picture")
        picture_compare(uploaded_file, predicted_color, "Original RGB", "Colorized by Model")


def generate_style_centering():
    """Style for Centered mode that app gets more width"""
    st.markdown(
        f"""
    <style>
        .reportview-container .main .block-container{{
            max-width: 1400px;
        }}
    </style>
    """,
        unsafe_allow_html=True,
    )
