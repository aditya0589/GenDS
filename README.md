# GenDS
Generate End to End Data Science Workflows directly using natural language prompts. 

GenDS is a Streamlit-based web application that allows users to upload data files (PDF, CSV, JSON, Excel) and generate Python code for various data science tasks. By leveraging Google’s Generative AI and FAISS vector search, the app processes uploaded files and generates code tailored to user-specified prompts across the data science workflow, including data loading, preprocessing, visualization, modeling, and more. Users can download the generated code as .py or .ipynb files and view visualizations directly in the app.

get the deployed app here : https://gends-ai-app.streamlit.app/

# Features

File Upload Support: Upload and process PDF, CSV, JSON, and Excel files to extract context for code generation.

Data Science Workflow: Generate code for key steps:
Data Loading and Inspection
Data Preprocessing and Cleaning
Exploratory Data Analysis
Data Visualization
Feature Engineering
Model Training
Model Evaluation
Model Deployment

Code Generation: Create well-commented, executable Python code using popular libraries (pandas, numpy, scikit-learn, matplotlib, seaborn).

Download Options: Download generated code as a .py file or Jupyter Notebook (.ipynb).

Visualization Display: View matplotlib/seaborn plots directly in the app for visualization and evaluation tasks.

User-Friendly Interface: Streamlit-based UI with clear prompts, workflow selection, and error handling.

Secure Execution: Restricted environment for visualization code execution to ensure safety.

# Prerequisites

Python: Version 3.8 or higher.

Google API Key: Required for Google Generative AI services. Obtain from Google Cloud Console.

System Dependencies: For FAISS, ensure C++ compiler and BLAS/LAPACK are installed (e.g., libopenblas-dev on Ubuntu, Visual Studio Build Tools on Windows).

Internet Connection: For API calls to Google Generative AI.

Git: For cloning the repository.

# Installation

Clone the Repository:

git clone https://github.com/aditya0589/GenDS.git
cd data-science-code-generator

Set Up a Virtual Environment:

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install Dependencies: Create a requirements.txt file with the following content:

streamlit==1.38.0
PyPDF2==3.0.1
pandas==2.2.2
numpy==1.26.4
langchain==0.3.0
langchain-google-genai==1.0.8
langchain-community==0.3.0
google-generativeai==0.7.2
faiss-cpu==1.8.0
python-dotenv==1.0.1
openpyxl==3.1.5
nbformat==5.10.4
seaborn==0.13.2
matplotlib==3.9.2

Then install:

## pip install -r requirements.txt

Configure Environment Variables: Create a .env file in the project root:

GOOGLE_API_KEY=your_api_key_here

Replace your_api_key_here with your Google API key.

Install System Dependencies for FAISS (if needed):

Ubuntu/Debian:

sudo apt-get install libopenblas-dev

macOS:

brew install libomp

Windows: Install Visual Studio Build Tools with C++ support.

Usage


Run the Application:

streamlit run data_science_code_generator.py

This opens the app in your default browser (typically http://localhost:8501).



Upload Files:
In the sidebar, upload one or more files (PDF, CSV, JSON, Excel).
Click "Process Files" to extract context and create a FAISS index.

Generate Code:

Select a data science workflow step from the dropdown (e.g., "Data Visualization").
Enter a detailed prompt (e.g., "Generate code to plot a histogram of the 'age' column from my CSV file")
Click Enter to generate code.

View/Download Code:
The generated code appears in a code block.
Use the "Download as .py" or "Download as .ipynb" buttons to save the code.

View Visualizations:
For "Data Visualization" or "Model Evaluation" steps, plots (using matplotlib/seaborn) are displayed below the code.
Console output (e.g., print statements) is shown if present.

Example Workflow

Upload a CSV file (data.csv):

age,salary
25,50000
30,60000
35,75000
40,80000

Process the file.
Select "Data Visualization" and enter: "Generate code to plot a histogram of the 'age' column."
View the generated code, download it, and see the histogram in the app.

Project Structure

data-science-code-generator/
├── app.py                          # Main Streamlit app
├── requirements.txt                # Python dependencies
├── .env                            # Environment variables (not tracked)
├── faiss_index/                    # FAISS index directory (created after processing)
└── README.md                       # Project documentation

# Dependencies
Python Libraries:
streamlit: Web app framework
PyPDF2: PDF text extraction
pandas, numpy: Data manipulation
langchain, langchain-google-genai, langchain-community: AI and vector search
google-generativeai: Google AI API
faiss-cpu: Vector similarity search
python-dotenv: Environment variable management
openpyxl: Excel file support
nbformat: Jupyter Notebook creation
matplotlib, seaborn: Visualization

System Dependencies:
C++ compiler and BLAS/LAPACK for FAISS

Internet access for API calls

Troubleshooting


API Errors:

Confirm your Google API key is valid and has sufficient quota.

get your own GOOGLE_API_KEY.
For additional help, open an issue on GitHub or check the Streamlit documentation.

# Contributions are welcome! To contribute:

Fork the repository.
Create a feature branch (git checkout -b feature/your-feature).
Commit changes (git commit -m "Add your feature").

Push to the branch (git push origin feature/your-feature).
Open a pull request.
Please include tests and update documentation as needed.

# License
This project is licensed under the MIT License. See the LICENSE file for details.

# Acknowledgments

Built with Streamlit and LangChain.
Powered by Google Generative AI.
Inspired by the need for automated data science workflows.
