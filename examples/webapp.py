import pandas as pd
import streamlit as st
from atom import ATOMClassifier


st.set_page_config(layout="wide")

# Sidebar ============================= >>
st.sidebar.title("Pipeline")

# Data cleaning options
st.sidebar.subheader("Data cleaning")
scale = st.sidebar.checkbox("Scale", True, "scale")
impute = st.sidebar.checkbox("Impute", False, "impute")
encode = st.sidebar.checkbox("Encode", False, "encode")

# Model options
st.sidebar.subheader("Models")
models = {
    "gnb": st.sidebar.checkbox("Gaussian Naive Bayes", True, "gnb"),
    "rf": st.sidebar.checkbox("Random Forest", True, "rf"),
    "et": st.sidebar.checkbox("Extra-Trees", False, "et"),
    "xgb": st.sidebar.checkbox("XGBoost", False, "xgb"),
    "lgb": st.sidebar.checkbox("LightGBM", False, "lgb"),
    "mlp": st.sidebar.checkbox("Multi-layer Perceptron", False, "mlp"),
}


# Data ingestion ====================== >>
st.header("Data")
data = st.file_uploader("Upload data:", type="csv")

if data is not None:
    data = pd.read_csv(data)
    st.text("Data preview:")
    st.dataframe(data.head())


st.header("Results")


# Model training ====================== >>
if st.sidebar.button("Run"):
    placeholder = st.empty()  # Empty to overwrite prints
    placeholder.write("Initializing atom...")

    atom = ATOMClassifier(data,
                          shuffle=False,
                          n_rows=1,
                          test_size=0.3,
                          n_jobs=1,
                          warnings=False,
                          verbose=2,
                          # logger="auto",
                          # experiment="test",
                          random_state=1)

    if scale:
        placeholder.write("Scaling the data...")
        atom.clean()
    if impute:
        placeholder.write("Imputing the missing values...")
        atom.impute(strat_num="drop", strat_cat="most_frequent", max_nan_rows=None)
    if encode:
        placeholder.write("Encoding the categorical features...")
        atom.encode(strategy="LeaveOneOut", max_onehot=10, frac_to_other=0.04)

    placeholder.write("Fitting the models...")
    atom.run(models=[key for key, value in models.items() if value], metric='f1')

    placeholder.write(atom.scoring())

    # Draw plots
    col1, col2 = st.beta_columns(2)
    col1.write(atom.plot_roc(title="ROC curve", display=None))
    col2.write(atom.plot_prc(title="Precision-Recall curve", display=None))

else:
    st.write("No results yet. Click the run button on the sidebar to fit the pipeline.")
