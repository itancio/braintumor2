
import streamlit as st
import tensorflow as tf
import numpy as np
import plotly.graph_objects as go
import cv2
import json
import google.generativeai as genai
import PIL.Image
import os
import matplotlib.pyplot as plt

from pprint import pprint
from scipy.ndimage import zoom
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.utils.scores import CategoricalScore
from tf_keras_vis.utils import normalize
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from google.colab import userdata
from dotenv import load_dotenv

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Input
# from tensorflow.keras.applications.vgg16 import preprocess_input
# from tensorflow.keras.layers import Dense, Dropout, Flatten
# from tensorflow.keras.optimizers import Adamax
# from tensorflow.keras.metrics import Precision, Recall




load = load_dotenv()
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))


# def load_xception_model(model_path):
#   img_shape=(299, 299, 3)
#   base_model = tf.keras.application.Xception(include_top=False, weights='imagenet', input_shape=img_shape, pooling='max')

#   model = Sequential([
#       base_model,
#       Flatten(),
#       Dropout(rate=0.3),
#       Dense(128, activation='relu'),
#       Dropout(rate=0.25),
#       Dense(4, activation='softmax')
#   ])

#   model.build((None, ) + img_shape)

#   # Compile the model
#   model.compile(Adamax(learning_rate=0.001),
#     loss='categorical_crossentropy',
#     metrics=['accuracy', Precision(), Recall()
#   ])

#   model.load_weights(model_path)

#   return model


def create_model_probability_chart(probabilities, model):
    models = list(probabilities.keys())
    probs = list(probabilities.values())

    fig = go.Figure(data=[
        go.Bar(
            y=models,
            x=probs,
            orientation="h",
            text=[f"{p: .4%}" for p in probs],
            textposition="auto"
        )
    ])
    fig.update_layout(
        title= f"Brain Tumor Classification Probability by the {model}",
        yaxis_title="Tumor types",
        xaxis_title="Probability",
        xaxis=dict(tickformat=".0%", range=[0, 1]),
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    return fig


def generate_saliency_map(model, img_array, class_index, img_size):
  # Create the saliency_maps directory if it doesn't exist
  output_dir = 'saliency_maps'
  os.makedirs(output_dir, exist_ok=True)

  # Convert image to tensor
  with tf.GradientTape() as tape:
    img_tensor = tf.convert_to_tensor(img_array)
    tape.watch(img_tensor)
    predictions = model(img_tensor)
    target_class = predictions[:, class_index]

  gradients = tape.gradient(target_class, img_tensor)
  gradients = tf.math.abs(gradients)
  gradients = tf.reduce_max(gradients, axis=-1)
  gradients = gradients.numpy().squeeze()
  # st.write(f'saliency graidents {gradients.shape}')

  # Resize gradients to match original image size
  gradients = cv2.resize(gradients, img_size)
  # st.write(f'saliency graidents resize {gradients.shape}')

  # Create a circular mask for the brain area
  center = (gradients.shape[0] // 2, gradients.shape[1] // 2)
  radius = min(center[0], center[1]) - 10
  y, x = np.ogrid[:gradients.shape[0], :gradients.shape[1]]
  mask = ((x - center[0])**2 + (y - center[1])**2) <= radius**2

  # Apply mask to gradients
  gradients = gradients * mask

  # Normalize only the brain area
  brain_gradients = gradients[mask]
  if brain_gradients.max() > brain_gradients.min():
    brain_gradients = (brain_gradients - brain_gradients.min()) / (brain_gradients.max() - brain_gradients.min())
  gradients[mask] = brain_gradients

  # Apply a higher threshold
  threshold = np.percentile(gradients[mask], 80)
  gradients[gradients < threshold] = 0

  # Apply more aggressive smoothing
  gradients = cv2.GaussianBlur(gradients, (11, 11), 0)
  # st.write(f'saliency graidents {gradients.shape}')

  # Create a heatmap overlay with enhanced contrast
  heatmap = cv2.applyColorMap(np.uint8(255 * gradients), cv2.COLORMAP_JET)
  heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
  # st.write(f'saliency headmap: {heatmap.shape}')

  # Resize heatmap to match original image size
  heatmap = cv2.resize(heatmap, img_size)
  st.write(f'saliency heatmap resize:  {gradients.shape}')

  # Superimpose the heatmap on original image with increased opacity
  original_img = image.img_to_array(img)
  superimposed_img = heatmap * 0.90 + original_img * 0.35
  superimposed_img = superimposed_img.astype(np.uint8)
  # st.write(f'saliency superimposed: {superimposed_img.shape}')

  img_path = os.path.join(output_dir, uploaded_file.name)
  with open(img_path, 'wb') as f:
    f.write(uploaded_file.getbuffer())

  saliency_map_path = f'{output_dir}/{uploaded_file.name}'

  # Save the saliency map
  cv2.imwrite(saliency_map_path, cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR))

  return superimposed_img


def generate_explanation(img_path, model_prediction, confidence):

  prompt1 = f"""
    You are a leading neurologist and specialist in brain tumor diagnosis.

    Your task is to interpret a saliency map from an MRI scan. This map, created by a deep learning model, has highlighted specific brain regions it focused on to classify the tumor. The model was trained to identify four categories: glioma, meningioma, pituitary tumor, or no tumor.

    The model has classified the image as '{model_prediction}' with a confidence level of {confidence * 100:.2f}%.

    In your explanation:
    - Describe the regions of the brain that the model focused on, particularly those highlighted in light cyan on the saliency map.
    - Offer insights on why these specific areas contributed to the model's classification.
    - Avoid directly stating 'the model is focusing on areas in light cyan'—instead, convey this more naturally in your explanation.
    - Limit your explanation to 4 sentences for concise clarity.
    - Think through your explanation step-by-step, ensuring accuracy in each detail.
  """

  prompt2 = f"""
    As a skilled communication expert, your task is to explain this diagnosis to various audiences.

    Please generate four distinct explanations for the following groups:
    - 'genz': Use language that resonates with Gen Z, keeping it informal and minimizing medical jargon.
    - '5th-grader': Use simple, accessible language suited for a 5th-grader, avoiding any technical terms.
    - 'normal': Cater to an adult audience with clear, everyday language that balances simplicity with thoroughness.
    - 'expert': Tailor your explanation for medical professionals, integrating technical details relevant to tumor diagnosis.

    In your response for each group:
    - Describe the brain regions the model focused on (highlighted in light cyan).
    - Suggest why these regions might have led the model to its classification.
    - Do not explicitly mention phrases like 'the model is focusing on areas in light cyan'—instead, phrase this naturally within your explanation.
    - Keep each explanation brief (no more than 4 sentences).

    Avoid using the words 'computer' or 'AI'.

    Please format your response as JSON, using this schema:
    {{
        "genz": str,
        "5th-grader": str,
        "normal": str,
        "expert": str
    }}
    Return: list[Mode]
  """

  img = PIL.Image.open(img_path)

  model = genai.GenerativeModel(model_name='gemini-1.5-flash')
  response = model.generate_content([prompt1, prompt2, img])
  return response.text

##########################################################################################
#
#     START OF FRONTEND UI
#
##########################################################################################


st.title('Brain Tumor Classification')
st.write('Upload an image of a brain MRI scan to classify.')
uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:

  selected_model = st.radio(
      'Select Model',
      ('Transfer Learning - Xception', 'Custom CNN')
  )

  if selected_model == 'Transfer Learning - Xception':
    model = load_model('/content/xception_1_.keras', compile=True)
    img_size = (299, 299)
  else:
    model = load_model('/content/cnn1_.keras', compile=True)
    img_size = (224, 224)

  labels = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
  img = image.load_img(uploaded_file, target_size=img_size)
  img_array = image.img_to_array(img)
  img_array = np.expand_dims(img_array, axis=0)
  img_array /= 255.0

  prediction = model.predict(img_array)

  # Get the class with the highest probability
  predicted_class_idx = np.argmax(prediction[0])
  result = labels[predicted_class_idx]

  prediction_dict = {label: prob for label, prob in zip(labels, prediction[0])}

  fig = create_model_probability_chart(prediction_dict, selected_model)
  st.plotly_chart(fig)

  saliency_map = generate_saliency_map(model, img_array, predicted_class_idx, img_size)
  saliency_map_path = f'saliency_maps/{uploaded_file.name}'

  col1, col2 = st.columns(2)

  with col1:
    st.image(uploaded_file, caption='Uploaded Image', use_container_width=True)
  with col2:
    st.image(saliency_map, caption='Saliency Map', use_container_width=True)

  st.write('## Classification Results')

  col1, col2 = st.columns([1, 1])

  with col1:
      st.markdown(
          f"""
          <div style='display: flex; align-items: center; justify-content: center; height: 100%; width: 100%;'>
              <div style='background-color: #000000; color: #ffffff; padding: 20px; border-radius: 15px; text-align: center; width: 100%;'>
                  <h3 style='color: #ffffff; font-size: 20px;'>Prediction</h3>
                  <p style='font-size: 36px; font-weight: 800; color: #ff0000; margin: 0;'>
                      {result}
                  </p>
              </div>
          </div>
          """,
          unsafe_allow_html=True
      )

  with col2:
      st.markdown(
          f"""
          <div style='display: flex; align-items: center; justify-content: center; height: 100%; width: 100%;'>
              <div style='background-color: #000000; color: #ffffff; padding: 20px; border-radius: 15px; text-align: center; width: 100%;'>
                  <h3 style='color: #ffffff; font-size: 20px;'>Confidence</h3>
                  <p style='font-size: 36px; font-weight: 800; color: #2196F3; margin: 0;'>
                      {prediction[0][predicted_class_idx]:.4%}
                  </p>
              </div>
          </div>
          """,
          unsafe_allow_html=True
      )

  # Parse the response to a JSON data
  data = generate_explanation(saliency_map_path, result, prediction[0][predicted_class_idx])
  data_string_cleaned = data.replace('```json', '').replace('```', '').strip()
  explanations = json.loads(data_string_cleaned)

  st.write('#### Explanation')

  with st.container(border=True):

    tab1, tab2, tab3, tab4 = st.tabs([
        'Normal',
        'Expert',
        '5th-grader',
        'Gen-Z',
    ])

    with tab1:
      explanations['normal']
    with tab2:
      explanations['expert']
    with tab3:
      explanations['5th-grader']
    with tab4:
      explanations['genz']

