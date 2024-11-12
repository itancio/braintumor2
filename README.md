# Brain Tumor Classification with Neural Network

[Demo]()

## Part 1

- Downloading the dataset
- Understanding the dataset
- Pre-processing the dataset
- Training the Transfer Learning model
- Training the CNN model
- Evaluating the modesl

## Part 2

- Building the web app
- Creating the UI and visualizations
- Generating explanations using Gemini 1.5 Flash

# Developer's Notes

## Add the git repo:

````bash
git init
git add *
git commit -m 'Initial commit'
git branch -M main
git remote add orgin https://github.com/itancio/braintumor.git
git push -u origin main

## Install required packages
```bash
pip install -r requirements.txt
````

## Create Python virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

## Running streamlit

```bash
cd streamlit
streamlit run main.py
```

* Error: clearing large files in the commit history
```bash
git filter-branch --force --index-filter \
'git rm --cached --ignore-unmatch notebook/xception_model.weights.h5' \
--prune-empty --tag-name-filter cat -- --all

git push origin --force --all
```
