
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
import gdown

from pprint import pprint
from scipy.ndimage import zoom
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

class Model:
  def __init__(self, model, img_size):
    self.model = model,
    self.img_size = img_size


labels = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

@st.cache_resource
def load_models():
    models = {}

    # Load xception model
    file = 'https://drive.google.com/file/d/1kOsrVm-nfTksSwK6AoCfgb4NNIlPQ4-O/view?usp=sharing'
    url = f'https://drive.google.com/uc?id={file.split("/")[-2]}'
    output = 'xception_1_.keras'
    gdown.download(url, output, quiet=False)
    model = load_model(output)
    img_size = (299, 299)
    models['xception'] = Model(model, img_size)

    # Load cnn model
    file = 'https://drive.google.com/file/d/1b0KD81MByk2Q4gQ9ktjx-BORKVt94G1M/view?usp=sharing'
    url = f'https://drive.google.com/uc?id={file.split("/")[-2]}'
    output = 'cnn1_.keras'
    gdown.download(url, output, quiet=False)
    model = load_model(output)
    img_size = (224, 224)
    models['sequential'] = Model(model, img_size)

    return models

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

def convert_img_to_array(img):
  img_array = image.img_to_array(img)
  img_array = np.expand_dims(img_array, axis=0)
  img_array /= 255.0
  return img_array

def generate_saliency_map(model, img, uploaded_file, class_index, img_size):
  # Convert image to numpy array and batch
  img_array = convert_img_to_array(img)

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


def generate_explanation(uploaded_file, img_path, model_prediction, confidence):
  orig = PIL.Image.open(uploaded_file)
  img = PIL.Image.open(img_path)
  model = genai.GenerativeModel(model_name='gemini-1.5-flash')

  prompt1 = f"""
    You are a leading neurologist and specialist in brain tumor diagnosis.
    You are provided with two images:
    * {orig} : This image is an MRI scan of the brain.
    * {img}: This image highlights the area of where the model is looking at.

    Your task is to interpret a saliency map from an MRI scan. This map, created by a deep learning model, 
    has highlighted specific brain regions it focused on to classify the tumor. 
    The model was trained to identify four categories: glioma, meningioma, pituitary tumor, or no tumor.

    The model has classified the image as '{model_prediction}' with a confidence level of {confidence * 100:.2f}%.

    In your explanation:
    - Describe the regions of the brain that the model focused on, particularly those highlighted in light cyan on the saliency map.
    - Offer insights on why these specific areas contributed to the model's classification.
    - Avoid directly stating 'the model is focusing on areas in light cyan'—instead, convey this more naturally in your explanation.
    - Limit your explanation to 4 sentences for concise clarity.
    - Think through your explanation step-by-step, ensuring accuracy in each detail.
  """

  response = model.generate_content([
    prompt1, 
    orig,
    img,
    ])
  return response.text


def generate_modes(explanation):
  model = genai.GenerativeModel(model_name='gemini-1.5-flash')
  prompt2 = f"""
    As a skilled communication expert, your task is to explain this {explanation} to various audiences.

    Please generate four distinct explanations for the following groups:
    - 'genz': Use language that resonates with Gen Z, keeping it informal and minimizing medical jargon.
    - '5th-grader': Use simple, accessible language suited for a 5th-grader, avoiding any technical terms.
    - 'normal': Cater to an adult audience with clear, everyday language that balances simplicity with thoroughness.
    - 'expert': Tailor your explanation for medical professionals, integrating technical details relevant to tumor diagnosis.

    In your response for each group:
    - Do not explicitly mention phrases like 'the model is focusing on areas in light cyan'—instead, phrase this naturally within your explanation.
    - Keep each explanation brief (no more than 10 sentences).
    - Avoid using the words 'computer' or 'AI'.

    Please format your response as JSON, using this schema:
    {{
        "genz": str,
        "5th-grader": str,
        "normal": str,
        "expert": str
    }}
    Return: list[Mode]
  """
  
  response = model.generate_content([
      prompt2,
      explanation, 
      ])
  return response.text

def generate_cross_diagnosis(uploaded_file, img_path, model_prediction, confidence, explanation):
    orig = PIL.Image.open(uploaded_file)
    img = PIL.Image.open(img_path)
    model = genai.GenerativeModel(model_name='gemini-1.5-flash')
    prompt = f"""
    The model predicted the image {model_prediction} with {confidence *100}% confidence. Here's the model's explanation: {explanation}
    As an LLM analyzing MRI scans, your task is to evaluate two images of brain MRI scans.
    On a magnetic resonance imaging (MRI) scan, brain tumors typically appear as abnormal masses or growths. 
    Healthcare providers look for the following characteristics when evaluating MRI scans for brain tumors:
    * Size, shape, and location: The tumor's size, shape, and location are noted. 
    * Contrast enhancement: Contrast-enhanced MRI scans are especially useful 
    because they can help distinguish tumor tissue from surrounding healthy tissue. 
    * T1-weighted images: Most brain tumors are hypointense on T1-weighted images. 
    * T2-weighted images: Most brain tumors are hyperintense on T2-weighted images. 
    * FLAIR sequences: Most brain tumors are hyperintense on FLAIR sequences. 
    * Diffusion-weighted imaging (DWI): Absence of restricted water movement is a typical finding. 
    * Metabolites: Certain metabolites, such as lipids, myo-inositol, and 2-hydroxyglutarate, can indicate the presence of a brain tumor. 
    
    Other characteristics of brain tumors on MRI scans include: 
    * A large cystic lesion with a mural nodule 
    * A mass that has spread in the middle area of the brain 
    * Degrees of brightness with contrast

    Based on these characteristics you will follow a detailed step-by-step process to identify areas that may correspond to tumor formations. 
    For each image, perform the following steps to reach a comprehensive assessment of the model’s predictions:

    1. Identify the Predicted Area:
      a. Begin by locating the specific region in the MRI scan where the model has identified 
      a potential abnormality or tumor. This may be in areas associated with common brain tumors, such as:
          - Pituitary (often a small, pea-sized structure at the base of the brain)
          - Glioma Region (usually in the cerebral hemispheres, involving glial cells in the brain)
          - Meningioma Region (typically located in the meninges, the protective layers surrounding the brain and spinal cord)
      b. Mark the exact coordinates or spatial area as identified by the model, if possible, using any available labeling or overlay for the model’s predicted location.
  
    2. Analyze the Area’s Characteristics:
      a. Describe the specific features of the identified area in the scan, including:
          - Size, shape, and contrast (lighter or darker areas that may indicate density differences).
          - Presence of any irregularities in the tissue, such as:
          - Tumor-like formations (e.g., unusual mass, irregular borders).
          - Enlargement or swelling (increased volume around the area).
          - Changes in structure, such as displacement of surrounding tissues.
          - other characteristics mentioned above.
      b. Note any observable signs of common brain tumors. For instance:
          - Pituitary tumors often cause enlargement of the pituitary gland.
          - Gliomas may show irregular, diffuse areas, especially in the frontal or temporal lobes.
          - Meningiomas typically appear as well-circumscribed masses near the skull base.

    3. Identify Abnormalities:  
      a. Based on the observations, assess whether there are abnormalities in the area, such as:
          - Presence of masses, unusual densities, or structural distortions.
          - Symmetry and consistency with known anatomical structures.
      b. Indicate if the characteristics align with typical tumor presentations or other possible conditions (e.g., cysts, benign growths, or swelling).
    
    4. Cross-Image Assessment:
      a. Next, analyze the second MRI image where the model has made another prediction. Apply the same identification and analysis steps:
        - Observe and describe any structures and characteristics in the predicted area.
        - Check for consistency with the first image’s findings, especially if it is a follow-up or alternate view of the same region.
    
    5. Evaluate the Model’s Prediction:
      a. Based on your analysis of both images, assess if the model’s prediction is likely correct. Consider:
        - Alignment of the model’s prediction with observed physical characteristics (e.g., size, shape, location).
        - Any anatomical or clinical features that may either confirm or contradict the presence of a tumor.
        - Whether similar features appear across both images, supporting the presence of a tumor or other abnormality.
      b. Provide a detailed explanation of why you think the model’s prediction is accurate or inaccurate. If inaccurate, suggest possible reasons (e.g., overlapping structures, artifacts, or limitations in MRI clarity).
    6. Conclude Your Evaluation:
      a.Summarize your final assessment, stating whether you agree or disagree with the model’s initial prediction. 
      Specify if the detected area in the MRI scan corresponds to a tumor based on observed abnormalities and anatomical context.
      b. If possible, suggest further imaging or analysis steps that might enhance accuracy, 
      such as using higher-resolution images, additional MRI slices, or applying contrast agents to differentiate between tumor and healthy tissue.
    """
    model = genai.GenerativeModel(model_name='gemini-1.5-flash')
    response = model.generate_content([
      prompt,
      orig, 
      img
      ])


    return response.text


# Function to load image and predict
def load_and_predict(model, uploaded_file, img_size=(299, 299)):
    img = image.load_img(uploaded_file, target_size=img_size)
    img_array = convert_img_to_array(img)
    prediction = model.predict(img_array)

    # Get the class with the highest probability
    predicted_class_idx = np.argmax(prediction[0])
    result = labels[predicted_class_idx]
    return prediction, result, predicted_class_idx

# Function to generate and display saliency map and results
def display_results(model, uploaded_file, prediction, result, predicted_class_idx, img_size):
    prediction_dict = {label: prob for label, prob in zip(labels, prediction[0])}
    fig = create_model_probability_chart(prediction_dict, model)
    st.plotly_chart(fig)

    saliency_map = generate_saliency_map(model, image.load_img(uploaded_file, target_size=img_size), uploaded_file, predicted_class_idx, img_size)
    saliency_map_path = f'saliency_maps/{uploaded_file.name}'

    col1, col2 = st.columns(2)
    with col1:
        st.image(uploaded_file, caption='Uploaded Image', use_container_width=True)
    with col2:
        st.image(saliency_map, caption='Saliency Map', use_container_width=True)

# Function to display prediction and confidence
def display_prediction_and_confidence(result, confidence):
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
                        {confidence:.4%}
                    </p>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

# Function to create explanation tabs
def display_explanation_tabs(uploaded_file, saliency_map_path, result, confidence):
    explanation = generate_explanation(uploaded_file, saliency_map_path, result, confidence)
    data = generate_modes(explanation)
    data_string_cleaned = data.replace('```json', '').replace('```', '').strip()
    explanations = json.loads(data_string_cleaned)

    st.write('#### Explanation')
    with st.container(border=True):
        tab1, tab2, tab3, tab4 = st.tabs(['Normal', 'Expert', '5th-grader', 'Gen-Z'])

        with tab1:
            st.write(explanations['normal'])
        with tab2:
            st.write(explanations['expert'])
        with tab3:
            st.write(explanations['5th-grader'])
        with tab4:
            st.write(explanations['genz'])
    
    st.write('#### Cross-Diagnosis')
    diagnosis = generate_cross_diagnosis(uploaded_file, saliency_map_path, result, confidence, explanation)
    st.write(diagnosis)
