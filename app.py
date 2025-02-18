import os
import sys
import streamlit as st

# Initialize session state for error tracking
if 'initialization_error' not in st.session_state:
    st.session_state.initialization_error = None

try:
    import pandas as pd
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    import joblib
    from sklearn.preprocessing import StandardScaler
except Exception as e:
    st.error(f"Failed to import required packages: {str(e)}")
    st.stop()

# Function to safely load models
def load_models():
    try:
        rf_model = joblib.load("random_forest_model.pkl")
        xgb_model = joblib.load("xgboost_model.pkl")
        scaler = joblib.load("scaler.pkl")
        return rf_model, xgb_model, scaler
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.write("Current directory contents:", os.listdir())
        return None, None, None

# Function to compute molecular descriptors
def calculate_descriptors(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return [
                Descriptors.MolWt(mol),
                Descriptors.MolLogP(mol),
                Descriptors.NumHDonors(mol),
                Descriptors.NumHAcceptors(mol),
                rdMolDescriptors.CalcTPSA(mol),
                Descriptors.NumRotatableBonds(mol)
            ]
    except Exception as e:
        st.error(f"Error calculating descriptors: {str(e)}")
    return [None] * 6

# Function to predict pIC50
def predict_pIC50(descriptors, rf_model, xgb_model, scaler):
    try:
        descriptors = np.array(descriptors).reshape(1, -1)
        descriptors_scaled = scaler.transform(descriptors)
        rf_pred = rf_model.predict(descriptors_scaled)
        xgb_pred = xgb_model.predict(descriptors_scaled)
        return (rf_pred + xgb_pred) / 2
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None

# Main UI
def main():
    st.title("💊 Computational Biology Drug Prediction App")
    st.write("Predict pIC50 values for molecules based on their molecular properties.")

    # Load models
    rf_model, xgb_model, scaler = load_models()
    if None in (rf_model, xgb_model, scaler):
        st.error("Failed to load required models. Please check if model files exist.")
        return

    # Single molecule prediction
    st.header("🔬 Predict for a Single Molecule")
    smiles_input = st.text_input("Enter a SMILES string:", "")

    if st.button("Predict"):
        if smiles_input:
            descriptors = calculate_descriptors(smiles_input)
            if None in descriptors:
                st.error("⚠️ Invalid SMILES string or calculation error.")
            else:
                prediction = predict_pIC50(descriptors, rf_model, xgb_model, scaler)
                if prediction is not None:
                    st.success(f"📊 Predicted pIC50 Value: {prediction[0]:.3f}")
        else:
            st.warning("⚠️ Please enter a SMILES string.")

    # Batch prediction
    st.header("📂 Batch Prediction from CSV")
    st.write("Upload a CSV file containing SMILES strings.")
    
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            if "SMILES" not in df.columns:
                st.error("⚠️ CSV must contain a 'SMILES' column.")
                return
                
            with st.spinner("Processing..."):
                df["Descriptors"] = df["SMILES"].apply(calculate_descriptors)
                df = df.dropna()
                
                if len(df) == 0:
                    st.error("No valid SMILES strings found in the file.")
                    return
                    
                df_features = pd.DataFrame(df["Descriptors"].tolist(), 
                                         columns=["MolWt", "LogP", "HDonors", 
                                                "HAcceptors", "TPSA", "RotatableBonds"])
                
                df_features_scaled = scaler.transform(df_features)
                df["Predicted pIC50"] = (rf_model.predict(df_features_scaled) + 
                                       xgb_model.predict(df_features_scaled)) / 2
                
                st.write("✅ Predictions Completed!")
                st.dataframe(df[["SMILES", "Predicted pIC50"]])
                
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Download Predictions", 
                                 csv, 
                                 "predictions.csv", 
                                 "text/csv")
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.write("Error details:", sys.exc_info())