import os
import tempfile
import zipfile
import json

# External libraries
import chardet
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from scipy.stats import skew, kurtosis, ks_2samp, chisquare
import joblib  # For loading the saved model

app = Flask(__name__)
CORS(app)

# Load the saved model (from the file my_model.pkl)
loaded_model = joblib.load("my_model.pkl")
print("Model loaded successfully from my_model.pkl")


def detect_encoding(file_path: str, sample_size: int = 65536) -> str:
    """
    Attempts to automatically detect the encoding of a text file using the 'chardet' library.
    Returns the detected encoding string. If detection fails, defaults to 'utf-8'.
    """
    try:
        with open(file_path, 'rb') as f:
            raw_data = f.read(sample_size)
        result = chardet.detect(raw_data)
        encoding = result['encoding']
        if not encoding:
            encoding = 'utf-8'
        return encoding
    except Exception:
        return 'utf-8'  # fallback if anything goes wrong


def read_csv_file(file_path: str) -> pd.DataFrame:
    """
    Reads a CSV file using chardet for auto-detected encoding,
    then tries fallback encodings if the first attempt fails.
    Returns a DataFrame if successful. Otherwise raises an Exception.
    """
    primary_enc = detect_encoding(file_path)

    # We'll try the detected encoding first, then some fallback encodings.
    encodings_to_try = [primary_enc]
    fallbacks = ['utf-8-sig', 'utf-8', 'latin-1', 'cp1255']
    for fb in fallbacks:
        if fb.lower() != primary_enc.lower():
            encodings_to_try.append(fb)

    last_error = None
    for enc in encodings_to_try:
        try:
            df = pd.read_csv(file_path, encoding=enc)
            return df  # If successful, return immediately
        except Exception as e:
            last_error = e
            # We'll move on to the next encoding in the list

    # If we failed all attempts, raise an error
    raise ValueError(f"Could not read CSV with any encoding from {encodings_to_try}. Last error: {last_error}")


def read_json_file_generic(file_path: str) -> pd.DataFrame:
    """
    Reads a JSON file using auto-detected encoding, then tries fallback encodings.
    After loading the JSON into a Python dict/list, tries:
      - pd.json_normalize(...) to flatten nested structures
      - fallback to pd.DataFrame(...) if data is a list of dicts

    Raises an Exception if it cannot produce a valid DataFrame.
    """
    primary_enc = detect_encoding(file_path)

    encodings_to_try = [primary_enc]
    fallbacks = ['utf-8-sig', 'utf-8', 'latin-1', 'cp1255']
    for fb in fallbacks:
        if fb.lower() != primary_enc.lower():
            encodings_to_try.append(fb)

    last_error = None
    for enc in encodings_to_try:
        try:
            with open(file_path, 'r', encoding=enc) as f:
                data = json.load(f)

            # Try flattening with pd.json_normalize
            try:
                df = pd.json_normalize(data)
                if df.empty:
                    # maybe it's a list of dicts?
                    if isinstance(data, list) and all(isinstance(x, dict) for x in data):
                        df2 = pd.DataFrame(data)
                        if not df2.empty:
                            return df2
                    raise ValueError("The JSON data could not be flattened into a non-empty DataFrame.")
                return df
            except Exception as e2:
                # Fallback: direct DataFrame if data is a list
                if isinstance(data, list):
                    try:
                        df2 = pd.DataFrame(data)
                        if not df2.empty:
                            return df2
                    except Exception:
                        pass
                raise ValueError(f"Cannot convert JSON to a DataFrame: {e2}")

        except Exception as e:
            last_error = e
            continue  # try the next encoding

    raise ValueError(f"Could not read JSON with any encoding from {encodings_to_try}. Last error: {last_error}")


def prepare_dataset_for_prediction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes all features (skew, kurtosis, chi, ks, etc.) for each column in df.
    We do not have 'is_categorical' here, because that's what the model will predict.
    Returns a DataFrame where each row corresponds to one column of 'df'.
    """
    feature_data = []

    for col in df.columns:
        col_data = df[col]

        # Check if the column is numeric or not
        if pd.api.types.is_numeric_dtype(col_data):
            # Numeric column
            col_data_dropna = col_data.dropna()
            n_total = len(col_data_dropna)
            value_counts = col_data_dropna.value_counts()
            n_unique = len(value_counts)
            unique_ratio = (n_unique / n_total * 100) if n_total > 0 else 0
            max_count = value_counts.max() if len(value_counts) > 0 else 0

            if n_total > 1:
                column_skewness = skew(col_data_dropna)
                column_kurtosis = kurtosis(col_data_dropna)
            else:
                column_skewness = 0
                column_kurtosis = 0

            ks_p_value = 0
            chi_p_value = 0
            if n_total > 1:
                # Kolmogorov-Smirnov test against uniform distribution
                _, ks_p_value = ks_2samp(col_data_dropna, np.random.uniform(size=n_total))

                # Chi-square test (binning numeric data into 10 bins)
                binned_series = pd.cut(col_data_dropna, bins=10, labels=False)
                observed = binned_series.value_counts()
                expected = np.ones_like(observed) * observed.sum() / len(observed)
                _, chi_p_value = chisquare(f_obs=observed, f_exp=expected)

        else:
            # Non-numeric column - factorize it
            col_factor, _ = pd.factorize(col_data)
            col_factor = pd.Series(col_factor).replace(-1, np.nan).dropna()
            n_total = len(col_factor)
            value_counts = col_factor.value_counts()
            n_unique = len(value_counts)
            unique_ratio = (n_unique / n_total * 100) if n_total > 0 else 0
            max_count = value_counts.max() if len(value_counts) > 0 else 0

            if n_total > 1:
                column_skewness = skew(col_factor)
                column_kurtosis = kurtosis(col_factor)
            else:
                column_skewness = 0
                column_kurtosis = 0

            ks_p_value = 0
            chi_p_value = 0
            if n_total > 1:
                # Kolmogorov-Smirnov test against uniform distribution
                _, ks_p_value = ks_2samp(col_factor, np.random.uniform(size=n_total))

                # Chi-square test on factorized values
                observed = value_counts
                expected = np.ones_like(observed) * observed.sum() / len(observed)
                _, chi_p_value = chisquare(f_obs=observed, f_exp=expected)

        feature_data.append({
            "skewness": column_skewness,
            "kurtosis": column_kurtosis,
            "relative_unique_ratio": unique_ratio,
            "max_repeated_value_count": max_count,
            "ks_p_value": ks_p_value,
            "chi_p_value": chi_p_value
        })

    return pd.DataFrame(feature_data)


@app.route("/upload", methods=["POST"])
def upload_and_predict():
    """
    Receives one or more files (CSV, JSON, or ZIP).
    Merges them (if multiple) into a single DataFrame,
    computes features for each column, and uses the loaded model
    to predict which columns are categorical (1) or numerical (0).

    Returns JSON with two lists:
      - "categorical_recommendation"
      - "numerical_recommendation"
    """
    uploaded_files = request.files.getlist("files")
    if not uploaded_files:
        return jsonify({"error": "No files were uploaded"}), 400

    dataframes = []

    for file in uploaded_files:
        tmp_file_path = None
        try:
            # 1) Save the uploaded file into a NamedTemporaryFile
            with tempfile.NamedTemporaryFile(delete=False, suffix="") as tmp:
                file.save(tmp.name)
                tmp_file_path = tmp.name

            # 2) Detect the extension by filename
            filename_lower = file.filename.lower()

            if filename_lower.endswith(".zip"):
                # If it's a ZIP file, extract each CSV/JSON inside
                with zipfile.ZipFile(tmp_file_path, 'r') as zip_ref:
                    for member in zip_ref.namelist():
                        if member.lower().endswith(".csv"):
                            # Extract to a temp file
                            with zip_ref.open(member) as zf, tempfile.NamedTemporaryFile(delete=False) as extracted_tmp:
                                extracted_bytes = zf.read()
                                extracted_tmp.write(extracted_bytes)
                                extracted_tmp_path = extracted_tmp.name

                            try:
                                df_sub = read_csv_file(extracted_tmp_path)
                                dataframes.append(df_sub)
                            except Exception as e:
                                os.remove(extracted_tmp_path)
                                return jsonify({"error": f"Failed to read CSV inside ZIP: {member}, error: {str(e)}"}), 400
                            finally:
                                if os.path.exists(extracted_tmp_path):
                                    os.remove(extracted_tmp_path)

                        elif member.lower().endswith(".json"):
                            # Extract to a temp file
                            with zip_ref.open(member) as zf, tempfile.NamedTemporaryFile(delete=False) as extracted_tmp:
                                extracted_bytes = zf.read()
                                extracted_tmp.write(extracted_bytes)
                                extracted_tmp_path = extracted_tmp.name
                            try:
                                df_sub = read_json_file_generic(extracted_tmp_path)
                                dataframes.append(df_sub)
                            except Exception as e:
                                os.remove(extracted_tmp_path)
                                return jsonify({"error": f"Failed to read JSON inside ZIP: {member}, error: {str(e)}"}), 400
                            finally:
                                if os.path.exists(extracted_tmp_path):
                                    os.remove(extracted_tmp_path)
                        else:
                            # If there's some other file type inside the ZIP, skip or handle accordingly
                            pass

            elif filename_lower.endswith(".csv"):
                # It's a CSV file
                df_csv = read_csv_file(tmp_file_path)
                dataframes.append(df_csv)

            elif filename_lower.endswith(".json"):
                # It's a JSON file
                df_json = read_json_file_generic(tmp_file_path)
                dataframes.append(df_json)

            else:
                # Unsupported file extension
                return jsonify({"error": f"Unsupported file type: {file.filename}"}), 400

        except Exception as e:
            # If there's any error while handling this file
            if tmp_file_path and os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)
            return jsonify({"error": f"Failed to process file {file.filename}: {str(e)}"}), 400

        finally:
            # Remove the temp file if still exists
            if tmp_file_path and os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)

    # If we got no valid DataFrames at all, return error
    if not dataframes:
        return jsonify({"error": "No valid data was found in the uploaded files"}), 400

    # Merge all dataframes side-by-side if multiple
    if len(dataframes) == 1:
        combined_data = dataframes[0]
    else:
        # We do an 'inner' join on the row indices
        combined_data = pd.concat(dataframes, axis=1, join='inner')

    # 1) Compute feature engineering
    feature_df = prepare_dataset_for_prediction(combined_data)

    # 2) Predict with the loaded model: 1 = categorical, 0 = numeric
    preds = loaded_model.predict(feature_df)

    # 3) Build lists of columns based on the model's prediction
    predicted_categorical_columns = []
    predicted_numerical_columns = []
    all_cols = list(combined_data.columns)

    for i, col_name in enumerate(all_cols):
        if preds[i] == 1:
            predicted_categorical_columns.append(col_name)
        else:
            predicted_numerical_columns.append(col_name)

    return jsonify({
        "message": "Success",
        "categorical_recommendation": predicted_categorical_columns,
        "numerical_recommendation": predicted_numerical_columns
    })


if __name__ == "__main__":
    # Run the Flask server on port 5000
    app.run(host="0.0.0.0", port=5000, debug=True)
