import cv2
import numpy as np
import streamlit as st
from PIL import Image


@st.cache(allow_output_mutation=True)
def get_predictor_model():
    from model import Model
    model = Model()
    return model


header = st.container()
model = get_predictor_model()

with header:
    st.title('Hello!')
    st.text(
        'Using this app you can classify whether there is fight on a street? or fire? or car crash? or everything is okay?')

uploaded_file = st.file_uploader("Or choose an image...")
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    image = np.array(image)
    label_text = model.predict(image=image)['label'].title()
    st.write(f'Predicted label is: **{label_text}**')
    st.write('Original Image')
    if len(image.shape) == 3:
        cv_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.image(image)
