import streamlit as st
import os

from style import *

model_dir = os.path.join(os.path.dirname(__file__), "models/pretrained")

models = os.listdir(model_dir)

models = [model.split(".")[0] for model in models]

st.title("Neural Style Transfer")

choice = st.selectbox("Select a model", models)

input_image = st.file_uploader("Upload image to stylize", type=['jpg', 'png'])

button = st.button("Stylize", use_container_width=True)

stylized_image = None

if button:
    with st.spinner("Styling your image..."):
        stylized_image = stylize({
            'model_name': f"{choice}.pth",
            'content_image': 'input.jpg',
            'output_path': 'src/data/output',
            'output_image': f'{choice}_output.jpg',
            'image_size': 512,
            'pretrained_models_path': 'src/models/pretrained'
        }, input_image)        

if stylized_image is not None:
    stylized_image = detransform(stylized_image)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(input_image, caption="Original")
        
    with col2:
        st.image(PIL_to_bytes(stylized_image), caption="Stylized Image", width=400)   
        
    # image_comparison(
    #     img1=input_image,
    #     img2=stylized_image,
    #     label1="Original",
    #     label2="Stylized"
    # )           
    
    st.download_button(
        label="Download stylized image",
        data=PIL_to_bytes(stylized_image),
        file_name="stylized.png",
        mime="image/png"
    )