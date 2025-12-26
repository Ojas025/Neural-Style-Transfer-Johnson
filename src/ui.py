import streamlit as st
import os

from style import *

models = os.listdir("./src/models/pretrained")

models = [model.split(".")[0] for model in models]

st.title("Neural Style Transfer")

choice = st.selectbox("Select a model", models)

input_image = st.file_uploader("Upload image to stylize")

if input_image:
    st.image(input_image, width=400, caption="Input image")

button = st.button("Stylize", use_container_width=True)

stylized_image = None

if button:
    with st.spinner("Styling your image..."):
        stylized_image = stylize({
            'model_name': f"{choice}.pth",
            'content_image': 'input.jpg',
            'output_path': './src/data/output',
            'output_image': f'{choice}_output.jpg',
            'image_size': 256,
            'pretrained_models_path': './src/models/pretrained'
        }, input_image)        

if stylized_image is not None:
    stylized_image = detransform(stylized_image)
    st.image(stylized_image, caption="Stylized Image", width=400)
    
    st.download_button(
        label="Download stylized image",
        data=PIL_to_bytes(stylized_image),
        file_name="stylized.png",
        mime="image/png"
    )