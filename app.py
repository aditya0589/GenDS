import streamlit as st
import pandas as pd
import json
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from nbformat import v4 as nbf
from contextlib import redirect_stdout

# Load environment variables
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# --- Utility Functions ---

def get_file_text(uploaded_files):
    text = ""
    for file in uploaded_files:
        file_extension = os.path.splitext(file.name)[1].lower()
        
        if file_extension == '.pdf':
            pdf_reader = PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
                
        elif file_extension == '.csv':
            df = pd.read_csv(file)
            text += f"CSV file content (first few rows):\n{df.head().to_string()}\n"
            text += f"CSV columns: {list(df.columns)}\n"
            
        elif file_extension == '.json':
            data = json.load(file)
            text += f"JSON file content (truncated):\n{json.dumps(data, indent=2)[:1000]}\n"
            
        elif file_extension in ['.xlsx', '.xls']:
            df = pd.read_excel(file)
            text += f"Excel file content (first few rows):\n{df.head().to_string()}\n"
            text += f"Excel columns: {list(df.columns)}\n"
            
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_code_generation_chain():
    prompt_template = """
    You are an expert data scientist. Generate Python code for the requested data science task based on the provided context and user prompt. 
    The code should be:
    - Complete and executable
    - Well-commented
    - Following best practices
    - Using popular libraries (pandas, numpy, scikit-learn, matplotlib, seaborn, etc.)
    - Appropriate for the specified data science workflow step
    - For visualization/evaluation, use plt.figure() to create figures, avoid plt.show(), and ensure plots are fully specified (e.g., include titles, labels)
    - Assume data is available as 'df' for CSV/Excel or 'data' for JSON

    Available workflow steps:
    1. Data Loading and Inspection
    2. Data Preprocessing and Cleaning
    3. Exploratory Data Analysis
    4. Data Visualization
    5. Feature Engineering
    6. Model Training
    7. Model Evaluation
    8. Model Deployment

    Context (file information):
    {context}

    User Prompt: 
    {question}

    Generated Code:
    ```python
    # Your generated code here
    ```
    """

    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type='stuff', prompt=prompt)
    return chain

def generate_code(user_prompt):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    try:
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_prompt)
        
        chain = get_code_generation_chain()
        response = chain({"input_documents": docs, "question": user_prompt}, return_only_outputs=True)
        return response["output_text"]
    except Exception as e:
        return f"Error: Please process files first. Details: {str(e)}"

def create_notebook(code):
    nb = nbf.new_notebook()
    cells = [nbf.new_code_cell(code)]
    nb['cells'] = cells
    return nb

def download_file(content, filename, mime_type, label):
    buffer = io.BytesIO()
    if mime_type == 'application/json':
        import json
        buffer.write(json.dumps(content).encode('utf-8'))
    else:
        if isinstance(content, str):
            buffer.write(content.encode('utf-8'))
        else:
            buffer.write(content)
    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode()
    href = f'<a href="data:{mime_type};base64,{b64}" download="{filename}">{label}</a>'
    st.markdown(href, unsafe_allow_html=True)

def execute_visualization_code(code, uploaded_files):
    safe_globals = {
        'pd': pd, 'pandas': pd,
        'np': __import__('numpy'),
        'plt': plt, 'matplotlib': __import__('matplotlib'),
        'sns': sns, 'seaborn': sns
    }
    
    for file in uploaded_files:
        file_extension = os.path.splitext(file.name)[1].lower()
        file.seek(0)
        try:
            if file_extension == '.csv':
                safe_globals['df'] = pd.read_csv(file)
            elif file_extension in ['.xlsx', '.xls']:
                safe_globals['df'] = pd.read_excel(file)
            elif file_extension == '.json':
                safe_globals['data'] = json.load(file)
        except Exception as e:
            return None, f"Error loading file {file.name}: {str(e)}"
    
    output = io.StringIO()
    with redirect_stdout(output):
        try:
            exec(code, safe_globals, {})
        except Exception as e:
            return None, f"Error executing visualization code: {str(e)}"
    
    figures = [plt.figure(i) for i in plt.get_fignums()]
    return figures, output.getvalue()

# --- Main Streamlit App ---

def show_data_science_code_generator():
    st.header("GenDS")
    st.subheader("Generate end-to-end Data Science workflows in just one line")
    st.write(
        """
        Upload your data files (PDF, CSV, JSON, Excel) in the sidebar, then specify a data science task to generate code for.
        Select a workflow step and provide a detailed prompt for the desired code.
        Download the code as a .py or .ipynb file, or view visualizations directly.
        """
    )

    workflow_steps = [
        "Data Loading and Inspection",
        "Data Preprocessing and Cleaning",
        "Exploratory Data Analysis",
        "Data Visualization",
        "Feature Engineering",
        "Model Training",
        "Model Evaluation",
        "Model Deployment"
    ]
    selected_step = st.selectbox("Select Data Science Workflow Step:", workflow_steps)

    user_prompt = st.text_area(
        f"Enter your code generation prompt for {selected_step}:",
        placeholder="Example: Generate code to plot a histogram of the 'age' column from my CSV file."
    )

    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []

    if user_prompt:
        try:
            with st.spinner('Generating code...'):
                code_response = generate_code(f"{selected_step}: {user_prompt}")
                
                if "```python" in code_response:
                    code_block = code_response.split("```python")[1].split("```")[0].strip()
                else:
                    code_block = code_response
                
                st.subheader("Generated Code:")
                st.code(code_block, language="python")

                st.subheader("Download Code:")
                col1, col2 = st.columns(2)
                with col1:
                    download_file(code_block, "generated_code.py", "text/x-python", "Download as .py")
                with col2:
                    notebook = create_notebook(code_block)
                    download_file(notebook, "generated_code.ipynb", "application/json", "Download as .ipynb")

                if selected_step in ["Data Visualization", "Model Evaluation"] and st.session_state.uploaded_files:
                    st.subheader("Visualization Output:")
                    with st.spinner("Generating visualizations..."):
                        figures, output = execute_visualization_code(code_block, st.session_state.uploaded_files)
                        if figures:
                            for fig in figures:
                                st.pyplot(fig)
                            if output.strip():
                                st.write("Console Output:")
                                st.text(output)
                            plt.close('all')
                        elif output:
                            st.error(output)
                        else:
                            st.warning("No visualizations generated. Check if the code produces plots using matplotlib/seaborn.")
                
        except Exception as e:
            st.error(f"Error generating code: {str(e)}")

    with st.sidebar:
        st.subheader("Upload and Process Files")
        uploaded_files = st.file_uploader(
            "Upload files (PDF, CSV, JSON, Excel)",
            accept_multiple_files=True,
            type=['pdf', 'csv', 'json', 'xlsx', 'xls']
        )

        if st.button("⚡ Process Files"):
            if uploaded_files:
                with st.spinner("⏳ Processing your files..."):
                    raw_text = get_file_text(uploaded_files)
                    if raw_text:
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        st.session_state.uploaded_files = uploaded_files
                        st.success("✅ Files processed successfully! You can now generate code.")
                    else:
                        st.warning("No text could be extracted from the uploaded files.")
            else:
                st.warning("Please upload at least one file first.")

if __name__ == "__main__":
    show_data_science_code_generator()