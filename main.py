import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.python.keras.models import Sequential, load_model
from PIL import Image

from util import classify, set_background



# title
st.title('Pneumonia classification')

# header
st.header('Upload an image of a chest X-ray')

# upload file
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

# classifier


model = load_model('./converted_keras/keras_model.h5')


# class names
with open('./converted_keras/labels.txt', 'r') as f:
    class_names = [a[:-1].split(' ')[1] for a in f.readlines()]
    f.close()



# display image
if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)

# classify image
class_name, conf_score = classify(image, model, class_names)
# write classification
st.write("## {}".format(class_name))
st.write("### score: {}%".format(int(conf_score * 1000) / 10))
