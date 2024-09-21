import streamlit as st

from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
from src.facial_key_point.utils.facial_key_points_detection import FacialKeyPointDetection

st.markdown('## Facial Key Point Detection')

image = st.file_uploader('Facial Image', ['jpg', 'png', 'jpeg'], accept_multiple_files = False)
if image is not None:
    image = Image.open(image).convert('RGB')
    st.image(image)
    image, kp = FacialKeyPointDetection.predict(image)

    fig = plt.figure()
    plt.imshow(image)
    plt.scatter(kp[0], kp[1], s=4, c='r')
    st.pyplot(fig)