import React, { useState } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [uploadResult, setUploadResult] = useState(null);
  const [error, setError] = useState(null);

  // פונקציה לטיפול בבחירת קבצים
  const handleFileChange = (e) => {
    setSelectedFiles(e.target.files);
  };

  // פונקציה לשילוח הקבצים לשרת
  const handleUpload = async () => {
    if (!selectedFiles || selectedFiles.length === 0) {
      setError("אנא בחר לפחות קובץ אחד");
      return;
    }
    setError(null);
    setUploadResult(null);

    const formData = new FormData();
    for (let i = 0; i < selectedFiles.length; i++) {
      formData.append("files", selectedFiles[i]);
    }

    try {
      // מניחים שהשרת רץ על פורט 5000
      const response = await axios.post(
        "http://localhost:5000/upload",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );
      setUploadResult(response.data);
    } catch (err) {
      setError(err?.response?.data?.error || "קרתה שגיאה בהעלאה");
    }
  };

  return (
    <div className="App">
      <header className="header">
        <h1>ברוכים הבאים למערכת סיווג תכונות</h1>
        <p>
          האתר מקבל קבצים (אחד או יותר) בפורמט CSV, מחבר אותם, מאמן מודל ומציע
          אילו תכונות מתאימות להיות קטגוריאליות או מספריות.
        </p>
      </header>

      <div className="upload-section">
        <input type="file" multiple onChange={handleFileChange} accept=".csv" />
        <button onClick={handleUpload}>שלח לשרת</button>
      </div>

      {error && <div className="error">{error}</div>}

      {uploadResult && (
        <div className="results">
          <h2>תוצאות הסיווג</h2>
          <div className="list-container">
            <div className="list-block">
              <h3>תכונות מומלצות כקטגוריאליות:</h3>
              <ul>
                {uploadResult.categorical_recommendation.map((col, index) => (
                  <li key={index}>{col}</li>
                ))}
              </ul>
            </div>
            <div className="list-block">
              <h3>תכונות מומלצות כמספריות:</h3>
              <ul>
                {uploadResult.numerical_recommendation.map((col, index) => (
                  <li key={index}>{col}</li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
