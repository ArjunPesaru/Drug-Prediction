
# Import necessary packages
import streamlit as st
import joblib
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from sklearn.preprocessing import StandardScaler

# Load models and scaler
rf_model = joblib.load("random_forest_model.pkl")
xgb_model = joblib.load("xgboost_model.pkl")
scaler = joblib.load("scaler.pkl")

# Function to compute molecular descriptors
def calculate_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return [
            Descriptors.MolWt(mol), Descriptors.MolLogP(mol), 
            Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol),
            rdMolDescriptors.CalcTPSA(mol), Descriptors.NumRotatableBonds(mol)
        ]
    else:
        return [None] * 6  # Return None for invalid SMILES

# Function to predict pIC50 using the ensemble model
def predict_pIC50(descriptors):
    descriptors = np.array(descriptors).reshape(1, -1)
    descriptors_scaled = scaler.transform(descriptors)
    rf_pred = rf_model.predict(descriptors_scaled)
    xgb_pred = xgb_model.predict(descriptors_scaled)
    return (rf_pred + xgb_pred) / 2  # Ensemble prediction

# Streamlit UI
st.title("💊 Computational Biology Drug Prediction App")
st.write("Predict pIC50 values for molecules based on their molecular properties.")

# Option to enter a SMILES string
st.header("🔬 Predict for a Single Molecule")
smiles_input = st.text_input("Enter a SMILES string:", "")

if st.button("Predict"):
    if smiles_input:
        descriptors = calculate_descriptors(smiles_input)
        if None in descriptors:
            st.error("⚠️ Invalid SMILES string. Please enter a valid molecule structure.")
        else:
            prediction = predict_pIC50(descriptors)
            st.success(f"📊 Predicted pIC50 Value: {prediction[0]:.3f}")
    else:
        st.warning("⚠️ Please enter a SMILES string.")

# Option to upload a CSV file
st.header("📂 Batch Prediction from CSV")
st.write("Upload a CSV file containing SMILES strings to get pIC50 predictions.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if "SMILES" not in df.columns:
        st.error("⚠️ CSV must contain a 'SMILES' column.")
    else:
        df["Descriptors"] = df["SMILES"].apply(calculate_descriptors)
        df = df.dropna()  # Remove invalid SMILES
        df_features = pd.DataFrame(df["Descriptors"].tolist(), columns=["MolWt", "LogP", "HDonors", "HAcceptors", "TPSA", "RotatableBonds"])
        
        # Normalize and predict
        df_features_scaled = scaler.transform(df_features)
        df["Predicted pIC50"] = (rf_model.predict(df_features_scaled) + xgb_model.predict(df_features_scaled)) / 2
        
        st.write("✅ Predictions Completed!")
        st.dataframe(df[["SMILES", "Predicted pIC50"]])
        
        # Provide download option
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Predictions", csv, "predictions.csv", "text/csv")

st.write("🔬 **Developed using Machine Learning & Bayesian Optimization for Drug Discovery**")
