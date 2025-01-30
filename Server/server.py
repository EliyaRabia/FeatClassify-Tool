import os
import tempfile
import zipfile
import json

import chardet
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from scipy.stats import skew, kurtosis, ks_2samp, chisquare
import joblib

app = Flask(__name__)
CORS(app)

# Load the saved model
loaded_model = joblib.load("my_model.pkl")
print("Model loaded successfully from my_model.pkl")


def detect_encoding(file_path: str, sample_size: int = 65536) -> str:
    """
    Detect file encoding using chardet. Fallback to utf-8 if detection fails.
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
        return 'utf-8'


def read_csv_file(file_path: str) -> pd.DataFrame:
    """
    Reads a CSV file with auto-detected encoding, tries fallbacks if needed.
    Returns a DataFrame if successful, otherwise raises an Exception.
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
            df = pd.read_csv(file_path, encoding=enc)
            return df
        except Exception as e:
            last_error = e
    raise ValueError(f"Could not read CSV. Tried encodings {encodings_to_try}. Last error: {last_error}")


def read_json_file_generic(file_path: str) -> pd.DataFrame:
    """
    Reads a JSON file (detects encoding), tries to flatten it using pd.json_normalize.
    If that fails, tries to create a DataFrame from a list of dicts. 
    Raises an Exception if no valid DataFrame can be built.
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

            try:
                df = pd.json_normalize(data)
                if df.empty:
                    # Maybe it's a list of dicts
                    if isinstance(data, list) and all(isinstance(x, dict) for x in data):
                        df2 = pd.DataFrame(data)
                        if not df2.empty:
                            return df2
                    raise ValueError("JSON data is empty or could not be flattened.")
                return df
            except Exception as e2:
                # Fallback: direct DataFrame if it's a list
                if isinstance(data, list):
                    try:
                        df2 = pd.DataFrame(data)
                        if not df2.empty:
                            return df2
                    except Exception:
                        pass
                raise ValueError(f"Cannot convert JSON to DataFrame: {e2}")

        except Exception as e:
            last_error = e
            continue

    raise ValueError(f"Could not read JSON. Tried encodings {encodings_to_try}. Last error: {last_error}")


def prepare_dataset_for_prediction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes features for each column in df (skewness, kurtosis, unique ratio, etc.)
    for use by the model that classifies each column as categorical or numerical.
    Returns a DataFrame where each row corresponds to one column of df.
    """
    feature_data = []

    for col in df.columns:
        col_data = df[col]

        if pd.api.types.is_numeric_dtype(col_data):
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
                # KS test comparing to uniform distribution
                _, ks_p_value = ks_2samp(col_data_dropna, np.random.uniform(size=n_total))

                # Chi-square test (numeric bins)
                binned_series = pd.cut(col_data_dropna, bins=10, labels=False)
                observed = binned_series.value_counts()
                expected = np.ones_like(observed) * observed.sum() / len(observed)
                _, chi_p_value = chisquare(f_obs=observed, f_exp=expected)

        else:
            # Non-numeric
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
                _, ks_p_value = ks_2samp(col_factor, np.random.uniform(size=n_total))
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
    Accepts one or more files (CSV, JSON, or ZIP).
    For each file:
      - Reads it (and unzips if ZIP).
      - Merges all found data frames if multiple are inside ZIP.
      - Prepares and predicts with a pre-trained model.
      - Returns a list of columns predicted as categorical or numerical.
    Responds with JSON containing an array `files`, 
    where each element has {filename, categorical_recommendation, numerical_recommendation}.
    """
    uploaded_files = request.files.getlist("files")
    if not uploaded_files:
        return jsonify({"error": "No files were uploaded"}), 400

    results = []

    for file in uploaded_files:
        tmp_file_path = None
        filename_str = file.filename
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix="") as tmp:
                file.save(tmp.name)
                tmp_file_path = tmp.name

            filename_lower = filename_str.lower()

            file_dataframes = []

            if filename_lower.endswith(".zip"):
                with zipfile.ZipFile(tmp_file_path, 'r') as zip_ref:
                    for member in zip_ref.namelist():
                        if member.lower().endswith(".csv"):
                            with zip_ref.open(member) as zf, tempfile.NamedTemporaryFile(delete=False) as extracted_tmp:
                                extracted_bytes = zf.read()
                                extracted_tmp.write(extracted_bytes)
                                extracted_tmp_path = extracted_tmp.name

                            try:
                                df_sub = read_csv_file(extracted_tmp_path)
                                file_dataframes.append(df_sub)
                            except Exception as e:
                                os.remove(extracted_tmp_path)
                                return jsonify({"error": f"Failed to read CSV inside ZIP: {member}, error: {str(e)}"}), 400
                            finally:
                                if os.path.exists(extracted_tmp_path):
                                    os.remove(extracted_tmp_path)

                        elif member.lower().endswith(".json"):
                            with zip_ref.open(member) as zf, tempfile.NamedTemporaryFile(delete=False) as extracted_tmp:
                                extracted_bytes = zf.read()
                                extracted_tmp.write(extracted_bytes)
                                extracted_tmp_path = extracted_tmp.name

                            try:
                                df_sub = read_json_file_generic(extracted_tmp_path)
                                file_dataframes.append(df_sub)
                            except Exception as e:
                                os.remove(extracted_tmp_path)
                                return jsonify({"error": f"Failed to read JSON inside ZIP: {member}, error: {str(e)}"}), 400
                            finally:
                                if os.path.exists(extracted_tmp_path):
                                    os.remove(extracted_tmp_path)
                        else:
                            # Ignore other file types in ZIP
                            pass

            elif filename_lower.endswith(".csv"):
                df_csv = read_csv_file(tmp_file_path)
                file_dataframes.append(df_csv)

            elif filename_lower.endswith(".json"):
                df_json = read_json_file_generic(tmp_file_path)
                file_dataframes.append(df_json)

            else:
                return jsonify({"error": f"Unsupported file type: {filename_str}"}), 400

            if not file_dataframes:
                return jsonify({"error": f"No valid data found in file {filename_str}"}), 400

            if len(file_dataframes) == 1:
                combined_data = file_dataframes[0]
            else:
                # Merge side-by-side by index, using inner join
                combined_data = pd.concat(file_dataframes, axis=1, join='inner')

            # Prepare features and predict
            feature_df = prepare_dataset_for_prediction(combined_data)
            preds = loaded_model.predict(feature_df)

            predicted_categorical_columns = []
            predicted_numerical_columns = []
            all_cols = list(combined_data.columns)

            for i, col_name in enumerate(all_cols):
                if preds[i] == 1:
                    predicted_categorical_columns.append(col_name)
                else:
                    predicted_numerical_columns.append(col_name)

            results.append({
                "filename": filename_str,
                "categorical_recommendation": predicted_categorical_columns,
                "numerical_recommendation": predicted_numerical_columns
            })

        except Exception as e:
            if tmp_file_path and os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)
            return jsonify({"error": f"Failed to process file {filename_str}: {str(e)}"}), 400
        finally:
            if tmp_file_path and os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)

    return jsonify({
        "message": "Success",
        "files": results
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
