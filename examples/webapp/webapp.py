"""Automated Tool for Optimized Modeling (ATOM).

Author: Mavs
Description: Webb application tutorial.
See https://medium.com/towards-data-science/from-raw-data-to-web-app-deployment-with-atom-and-streamlit-d8df381aa19f

"""

import pandas as pd
import streamlit as st

from atom import ATOMClassifier


# Expand the web app across the whole screen
st.set_page_config(layout="wide")

st.sidebar.title("Pipeline")

# Data cleaning options
st.sidebar.subheader("Data cleaning")
scale = st.sidebar.checkbox("Scale", value=False, key="scale")
encode = st.sidebar.checkbox("Encode", value=False, key="encode")
impute = st.sidebar.checkbox("Impute", value=False, key="impute")

# Model options
st.sidebar.subheader("Models")
models = {
    "gnb": st.sidebar.checkbox("Gaussian Naive Bayes", value=True, key="gnb"),
    "rf": st.sidebar.checkbox("Random Forest", value=True, key="rf"),
    "et": st.sidebar.checkbox("Extra-Trees", value=False, key="et"),
    "xgb": st.sidebar.checkbox("XGBoost", value=False, key="xgb"),
    "lgb": st.sidebar.checkbox("LightGBM", value=False, key="lgb"),
}


st.header("Data")
data = st.file_uploader("Upload data:", type="csv")

# If a dataset is uploaded, show a preview
if data is not None:
    data = pd.read_csv(data)
    st.text("Data preview:")
    st.dataframe(data.head())


st.header("Results")

if st.sidebar.button("Run"):
    placeholder = st.empty()  # Empty to overwrite write statements
    placeholder.write("Initializing atom...")

    # Initialize atom
    atom = ATOMClassifier(data, verbose=2, random_state=1)

    if scale:
        placeholder.write("Scaling the data...")
        atom.scale()
    if encode:
        placeholder.write("Encoding the categorical features...")
        atom.encode(strategy="Target", max_onehot=10)
    if impute:
        placeholder.write("Imputing the missing values...")
        atom.impute(strat_num="drop", strat_cat="most_frequent")

    placeholder.write("Fitting the models...")
    to_run = [key for key, value in models.items() if value]
    atom.run(models=to_run, metric="f1")

    # Display metric results
    placeholder.write(atom.evaluate())

    # Draw plots
    col1, col2 = st.beta_columns(2)
    col1.write(atom.plot_roc(title="ROC curve", display=None))
    col2.write(atom.plot_prc(title="PR curve", display=None))

else:
    st.write("No results yet. Click the run button!")
