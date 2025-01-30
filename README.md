# Feature Classifier Tool
![Screenshot 2025-01-30 134553](https://github.com/user-attachments/assets/d8ca2cfb-8f21-4804-95c3-d67ebe35542a)
![Screenshot 2025-01-30 134533](https://github.com/user-attachments/assets/56329c26-617c-46e7-aa6d-6fefdf9b8cd1)


## Abstract
In today's data-driven world, distinguishing between categorical and numerical features is essential for effective data preprocessing and analysis. The Feature Classifier Tool is designed to automate this process, helping data scientists, analysts, and engineers easily classify dataset features. This tool provides:<br>
- Automated Feature Classification: Uses machine learning to classify columns as categorical or numerical.
- Multi-File Support: Accepts multiple file formats, including CSV, JSON, and ZIP.
- User-Friendly Interface: A simple drag-and-drop UI for seamless file uploads.
- Efficiency & Accuracy: Minimizes manual preprocessing, saving time and reducing errors.

This tool is ideal for machine learning practitioners, data preprocessing tasks, and general dataset exploration.

## Installation
1. Prerequisites
Before getting started, ensure you have the following installed:
- [VS Code](https://code.visualstudio.com/download) (or any preferred IDE)
- [Node.js](https://nodejs.org/en/download) (for the frontend React application)
- [Python 3.8+](https://www.python.org/downloads/) (for the backend Flask server)

2. Clone the Repository
Open your terminal and navigate to the directory where you want to clone the project:<br>
git clone https://github.com/EliyaRabia/feature-classifier-tool.git

Then, navigate to the project folder:
cd feature-classifier-tool

3. Backend Setup (Flask)
Create a virtual environment (recommended for dependency management):
- python -m venv venv
- source venv/bin/activate  # On Windows: venv\Scripts\activate
- 
Install the required Python dependencies:
pip install -r requirements.txt

4. Frontend Setup (React)
Move into the Client directory:

cd Client
Install the required Node.js dependencies:
npm install

Running the Application

Navigate to the Client folder if you are not already there:

cd Client

npm start

Run the Flask server:

cd ../Server

python server.py

By default, the Web will run on http://127.0.0.1:3000.


## How to Use the Website

### Upload Files:

Click the "Select Files" button and choose one or more CSV/JSON/ZIP files.

The selected files will be displayed.

### Analyze Features:

Click the "Upload" button to process the files.

The server will classify each column as categorical or numerical.

### View Results:

The results will be displayed per file, showing categorical and numerical columns separately.

### Reset & Try Again:

Click the "Reset" button to clear all selections and start over.

## Related Work
During my deep dive into data analysis, I developed the machine learning model that powers this tool. This model was trained using advanced techniques to accurately classify dataset features. For those interested in a more detailed explanation of the model and the development process, please refer to the following link:<br>
[Data Project](https://github.com/EliyaRabia/Tabular-Data-Project)
