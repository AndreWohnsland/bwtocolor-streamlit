# BW 2 Color App

This was part of the practica for computer vision.
For package management, the new python [poetry](https://python-poetry.org/) package manager is used.

# Getting Started

First install the dependencies with:

```bash
poetry install
# or
pip install -r requirements.txt
```

The requirements file was generated with the pip-chill command since it doesn't list external dependencies:

```bash
pip-chill > requirements.txt
# or use poetry with
poetry export -f requirements.txt --without-hashes --output requirements.txt
```

Then you can run the web app with:

```bash
streamlit run app.py
# or
poetry run streamlit run app.py
```

# The App

You just upload your picture in the sidebar. Then you can select if the reference (Caffe) model or the self trained GAN + U-Net model shall be used. You can also use colored pictures and tick the checkbox to show the original and prediction side by side. A download option for the result ist also available over the download button.
