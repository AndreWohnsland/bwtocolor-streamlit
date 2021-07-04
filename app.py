import streamlit as st

from bwtocolor.footer import footer
from bwtocolor.models import colorize_image, precheck_models
from bwtocolor.utils import (
    generate_picture_comparison, get_image_download_link, generate_information,
    generate_sidebar, prepare_input_image, generate_model_information,
)


st.set_page_config(
    page_title="BW 2 Color | Wohnsland & Bohn",
    page_icon="🖼️",
    layout="wide",
)

precheck_models()

(use_caffe, is_color, uploaded_file) = generate_sidebar()

st.title("🖼️ Blackwhite to Color Converter")
st.write("Some *magic* conditional GAN to colorize your favourite pictures from sad BW to sparkling colorized versions.")

if uploaded_file is not None:
    (file_name, image_bw, image) = prepare_input_image(uploaded_file)
    predicted_color = colorize_image(image, use_caffe)

    generate_picture_comparison(image_bw, uploaded_file, predicted_color, is_color)
    st.markdown(get_image_download_link(predicted_color, file_name), unsafe_allow_html=True)

else:
    generate_information()

generate_model_information()
st.markdown(footer, unsafe_allow_html=True)
