
import streamlit as st
# import tensorflow as tf
# import numpy as np
# import plotly.graph_objects as go
# import cv2
# import json
# import google.generativeai as genai
# import PIL.Image
# import os
# import matplotlib.pyplot as plt


# from pprint import pprint
# from scipy.ndimage import zoom
# from google.colab import userdata
# from dotenv import load_dotenv

# from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing import image
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, Input
from utils import (
    load_models, 
    display_explanation_tabs,
    display_prediction_and_confidence,
    display_results,
    load_and_predict
)


##########################################################################################
#
#     START OF FRONTEND UI
#
##########################################################################################

models = load_models()
modelNames = list(models.keys())

st.title('Brain Tumor Classification')
st.write('Upload an image of a brain MRI scan to classify.')
uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:

  # Tabs for each model
  model_tab1, model_tab2 = st.tabs([name.title() for name in modelNames])

  # Model 1 Tab
  with model_tab1:
      modelName = modelNames[0]
      model1 = models[modelName].model[0]
      img_size1 = models[modelName].img_size

      st.write(f'## {modelName.title()} Model Results')

      prediction, result, predicted_class_idx = load_and_predict(model1, uploaded_file, img_size1)
      display_results(model1, uploaded_file, prediction, result, predicted_class_idx, img_size=img_size1)
      display_prediction_and_confidence(result, prediction[0][predicted_class_idx])
      display_explanation_tabs(uploaded_file, f'saliency_maps/{uploaded_file.name}', result, prediction[0][predicted_class_idx])

  # Model 2 Tab
  with model_tab2:
      modelName = modelNames[1]
      model2 = models[modelName].model[0]
      img_size2 = models[modelName].img_size

      st.write(f'## {modelName.title()} Model Results')

      prediction, result, predicted_class_idx = load_and_predict(model2, uploaded_file, img_size2)
      display_results(model2, uploaded_file, prediction, result, predicted_class_idx, img_size=img_size2)
      display_prediction_and_confidence(result, prediction[0][predicted_class_idx])
      display_explanation_tabs(uploaded_file, f'saliency_maps/{uploaded_file.name}', result, prediction[0][predicted_class_idx])
