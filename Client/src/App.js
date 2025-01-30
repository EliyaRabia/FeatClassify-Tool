import React, { useState } from "react";
import { Button, Box, Typography, Paper, Alert } from "@mui/material";
import CloudUploadIcon from "@mui/icons-material/CloudUpload";
import axios from "axios";

const App = () => {
  const [files, setFiles] = useState([]);
  const [fileResults, setFileResults] = useState([]);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  // Short paragraph text
  const paragraphText = `
    In this site you can upload one or more files 
    (CSV, JSON, or ZIP), the system will analyze each file's columns 
    and classify them as either categorical or numerical for better analyze 
    data or prepare Modeling.
  `;

  const handleFileChange = (event) => {
    if (event.target.files) {
      setFiles(Array.from(event.target.files)); // convert FileList to Array
    }
  };

  const handleUpload = async () => {
    if (!files || files.length === 0) {
      setError("Please select at least one file.");
      return;
    }

    setError(null);
    setLoading(true);
    setFileResults([]); // clear previous results

    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
      formData.append("files", files[i]);
    }

    try {
      const response = await axios.post(
        "http://127.0.0.1:5000/upload",
        formData,
        {
          headers: { "Content-Type": "multipart/form-data" },
        }
      );

      if (response.data?.files) {
        // server returns an array of {filename, categorical_recommendation, numerical_recommendation}
        setFileResults(response.data.files);
      } else {
        setError("Unexpected response from server.");
      }
    } catch (err) {
      setError("Error uploading files. Please try again.");
    }

    setLoading(false);
  };

  // Reset all states: clear files and results, and also clear the file input’s value
  const handleReset = () => {
    setFiles([]);
    setFileResults([]);
    setError(null);
    setLoading(false);

    // Clear the actual file input's value
    const fileInput = document.getElementById("file-upload");
    if (fileInput) {
      fileInput.value = null;
    }
  };

  return (
    <Box
      sx={{
        width: "100%",
        minHeight: "100vh",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        backgroundColor: "#f5f5f5",
        py: 4,
      }}
    >
      {/* Title in the center - now with the same primary color */}
      <Typography
        variant="h3"
        align="center"
        sx={{ fontWeight: "bold", mb: 2, color: "primary.main" }}
      >
        Feature Classifier Tool
      </Typography>

      {/* Short paragraph about the site */}
      <Typography
        variant="body1"
        align="center"
        sx={{ maxWidth: 600, mb: 4, lineHeight: 1.6 }}
      >
        {paragraphText}
      </Typography>

      {/* Wrapper for file upload */}
      <Paper
        elevation={3}
        sx={{
          p: 4,
          textAlign: "center",
          maxWidth: 600,
          width: "100%",
          borderRadius: 2,
        }}
      >
        <Typography
          variant="h5"
          gutterBottom
          sx={{ fontWeight: "bold", color: "#1976d2" }}
        >
          Upload Files
        </Typography>
        <Typography variant="subtitle1" gutterBottom sx={{ color: "#555" }}>
          Choose one or more files (CSV, JSON, or ZIP).
        </Typography>

        {error && (
          <Alert severity="error" sx={{ my: 2 }}>
            {error}
          </Alert>
        )}

        {/* Hidden file input */}
        <input
          type="file"
          multiple
          onChange={handleFileChange}
          style={{ display: "none" }}
          id="file-upload"
        />
        <label htmlFor="file-upload">
          <Button
            variant="contained"
            component="span"
            startIcon={<CloudUploadIcon />}
            sx={{ mt: 2 }}
            color="primary" // same color as Reset now
          >
            Select Files
          </Button>
        </label>

        {/* Display selected file names */}
        {files.length > 0 && (
          <Box sx={{ mt: 2, textAlign: "left" }}>
            <Typography variant="body2" sx={{ fontWeight: "bold" }}>
              Selected Files:
            </Typography>
            {files.map((f, idx) => (
              <Typography key={idx} variant="body2">
                {f.name}
              </Typography>
            ))}
          </Box>
        )}

        {/* Upload and Reset buttons */}
        <Box sx={{ mt: 2 }}>
          <Button
            variant="contained"
            color="primary"
            onClick={handleUpload}
            disabled={loading}
            sx={{ mr: 2 }}
          >
            {loading ? "Uploading..." : "Upload"}
          </Button>

          <Button variant="contained" color="primary" onClick={handleReset}>
            Reset
          </Button>
        </Box>
      </Paper>

      {/* Classification results per file */}
      <Box sx={{ mt: 4, maxWidth: 600, width: "100%" }}>
        {fileResults.map((result, index) => (
          <Paper
            key={index}
            elevation={2}
            sx={{ p: 2, mb: 3, backgroundColor: "#fafafa" }}
          >
            <Typography variant="h6" sx={{ fontWeight: "bold", mb: 1 }}>
              Results for: {result.filename}
            </Typography>

            {/* Categorical columns */}
            {result.categorical_recommendation &&
              result.categorical_recommendation.length > 0 && (
                <Box
                  sx={{
                    mb: 2,
                    backgroundColor: "#e3f2fd",
                    p: 1,
                    borderRadius: 1,
                  }}
                >
                  <Typography variant="subtitle1" sx={{ fontWeight: "bold" }}>
                    Categorical Features
                  </Typography>
                  {result.categorical_recommendation.map((col, i) => (
                    <Typography key={i} variant="body2">
                      {col}
                    </Typography>
                  ))}
                </Box>
              )}

            {/* Numerical columns */}
            {result.numerical_recommendation &&
              result.numerical_recommendation.length > 0 && (
                <Box
                  sx={{
                    mb: 1,
                    backgroundColor: "#fffde7",
                    p: 1,
                    borderRadius: 1,
                  }}
                >
                  <Typography variant="subtitle1" sx={{ fontWeight: "bold" }}>
                    Numerical Features
                  </Typography>
                  {result.numerical_recommendation.map((col, i) => (
                    <Typography key={i} variant="body2">
                      {col}
                    </Typography>
                  ))}
                </Box>
              )}
          </Paper>
        ))}
      </Box>
    </Box>
  );
};

export default App;
