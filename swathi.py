import os
import streamlit as st
import re
import csv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import google.generativeai as genai
import time

# Directly insert your Google API key here
GOOGLE_API_KEY = "AIzaSyDOrv3RayLX8j0B9C_cWwncoDjVfVHwZds"  # Replace with your actual API key

# Set up Google Generative AI
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Custom CSS for enhanced UI
def load_css():
    st.markdown("""
    <style>
    /* Main theme colors */
    :root {
        --primary-color: #4F46E5;
        --secondary-color: #7c3aed;
        --accent-color: #10b981;
        --background-color: #f8fafc;
        --text-color: #1e293b;
        --light-text: #64748b;
    }
    
    /* Page background */
    .stApp {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
    }
    
    /* Header styling */
    h1, h2, h3 {
        color: var(--primary-color);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-weight: 700;
        margin-bottom: 1.5rem;
    }
    
    h1 {
        background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem !important;
        text-align: center;
        padding: 1rem;
        border-bottom: 2px solid #e2e8f0;
    }
    
    /* Chat container */
    .stChatMessage {
        border-radius: 15px;
        padding: 10px 15px;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transition: transform 0.2s ease-in-out;
    }
    
    .stChatMessage:hover {
        transform: translateY(-2px);
    }
    
    /* Buttons styling */
    .stButton>button {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
        background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
        color: white;
        border: none;
    }
    
    .stButton>button:hover {
        background: linear-gradient(90deg, var(--secondary-color), var(--primary-color));
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(79, 70, 229, 0.2);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #4338ca 0%, #3730a3 100%);
    }
    
    .css-1d391kg .css-1v3fvcr {
        color: white;
    }
    
    /* Input fields */
    .stTextInput>div>div>input {
        border-radius: 10px;
        border: 2px solid #cbd5e1;
        padding: 1rem;
        transition: all 0.3s ease;
    }
    
    .stTextInput>div>div>input:focus {
        border-color: var(--primary-color);
        box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.2);
    }
    
    /* Select box styling */
    .stSelectbox>div>div {
        border-radius: 10px;
        border: 2px solid #cbd5e1;
    }
    
    /* Card styling for info boxes */
    .info-card {
        background-color: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.05);
        margin-bottom: 1.5rem;
        border-left: 5px solid var(--primary-color);
    }
    
    /* Badge styling */
    .badge {
        display: inline-block;
        padding: 0.35em 0.65em;
        font-size: 0.75em;
        font-weight: 700;
        line-height: 1;
        color: #fff;
        text-align: center;
        white-space: nowrap;
        vertical-align: baseline;
        border-radius: 0.25rem;
        background-color: var(--accent-color);
        margin-right: 0.5rem;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .animated {
        animation: fadeIn 0.5s ease-out forwards;
    }
    
    /* Custom chat bubbles */
    .user-bubble {
        background-color: #818cf8;
        color: white;
        border-radius: 18px 18px 0 18px;
        padding: 12px 16px;
        margin: 10px 0;
        max-width: 80%;
        align-self: flex-end;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        animation: fadeIn 0.3s ease-out forwards;
    }
    
    .assistant-bubble {
        background-color: white;
        color: #1e293b;
        border-radius: 18px 18px 18px 0;
        padding: 12px 16px;
        margin: 10px 0;
        max-width: 80%;
        align-self: flex-start;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        animation: fadeIn 0.3s ease-out forwards;
        border-left: 4px solid var(--primary-color);
    }
    
    /* Suggestion buttons */
    .suggestion-button {
        background-color: white;
        border: 1px solid #e2e8f0;
        border-radius: 20px;
        padding: 0.5rem 1rem;
        margin: 0.25rem;
        font-size: 0.85rem;
        cursor: pointer;
        transition: all 0.2s ease;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    
    .suggestion-button:hover {
        background-color: #f8fafc;
        border-color: var(--primary-color);
        transform: translateY(-2px);
    }
    
    /* Loading animation */
    .loading-spinner {
        display: inline-block;
        width: 50px;
        height: 50px;
        border: 5px solid #f3f3f3;
        border-radius: 50%;
        border-top: 5px solid var(--primary-color);
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Tooltip styling */
    .tooltip {
        position: relative;
        display: inline-block;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: #1e293b;
        color: white;
        text-align: center;
        border-radius: 6px;
        padding: 10px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 0.8rem;
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    
    /* Page transitions */
    .fade-in {
        animation: fadeIn 0.5s ease-out forwards;
    }
    </style>
    """, unsafe_allow_html=True)

# Function to extract text from PDF files for college chatbot
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into chunks for college chatbot
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create a vector store for college chatbot
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to create a conversational chain for college chatbot
def get_conversational_chain():
    prompt_template = """ Answer the question as detailed as possible from the provided context. If the answer is not in the provided context, say, "Answer is not available in the context." Context:\n{context}\n Question:\n{question}\n Answer: """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-002", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to handle user input and generate responses for college chatbot
def user_input(user_question):
    normalized_question = user_question.lower()
    keyword_mapping = {
        "hod": "head",
        "cse": "computer science and engineering",
        "ece": "electrical and communication engineering",
        "eee": "electrical and electronics engineering",
        "ai": "artificial intelligence",
        "ds": "data science",
    }
    
    for key, value in keyword_mapping.items():
        normalized_question = normalized_question.replace(key, value)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(normalized_question)
    
    chain = get_conversational_chain()
    
    # Simulate loading with a progress bar
    with st.spinner("🤔 Thinking..."):
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.01)  # Small delay for visual effect
            progress_bar.progress(i + 1)
        
        response = chain({"input_documents": docs, "question": normalized_question}, return_only_outputs=True)
    
    return response["output_text"]

# Function to extract data from CSV files for student marks analysis
def extract_csv(pathname: str) -> list[str]:
    parts = [f"---- START OF CSV {pathname} ---"]
    
    with open(pathname, "r", newline="") as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            parts.append(" ".join(row))  # Join the row into a single string
            
    return parts

# Function to validate roll number format and existence in data
def is_valid_roll_number(roll_number: str, combined_data: list) -> bool:
    pattern = r'^(20|21)[A-Z]{2}[0-9]{1}[A-Z]{1}[0-9]{4}$'
    
    if re.match(pattern, roll_number):
        return any(roll_number in entry for entry in combined_data)
    
    return False

# Function to display animated loading 
def display_loading_animation():
    cols = st.columns(3)
    with cols[1]:
        st.markdown('<div class="loading-spinner"></div>', unsafe_allow_html=True)
    time.sleep(1.5)  # Simulate processing time

# Function to show custom welcome animation
def show_welcome_animation():
    st.markdown("""
    <div class="animated" style="text-align: center; margin-bottom: 2rem;">
        <div style="font-size: 3rem; margin-bottom: 1rem;">👋</div>
        <h2>Welcome to RCEE Interactive Assistant</h2>
        <p>Your intelligent guide to college information and student marks analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    time.sleep(0.5)  # Brief pause for animation effect

# Display pretty formatted message bubbles
def display_chat_message(message, is_user=True):
    if is_user:
        st.markdown(f'<div class="user-bubble">{message}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="assistant-bubble">{message}</div>', unsafe_allow_html=True)

# Create a card with shadow and border
def create_info_card(title, content, icon="ℹ️"):
    st.markdown(f"""
    <div class="info-card">
        <h3>{icon} {title}</h3>
        <p>{content}</p>
    </div>
    """, unsafe_allow_html=True)

# Streamlit app UI setup
def main():
    st.set_page_config(
        page_title="🎓 RCEE Interactive Assistant",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Load custom CSS
    load_css()
    
    # Sidebar with gradient and improved navigation
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <h2 style="color: white;">🎓 RCEE Interactive Assistant</h2>
            <p style="color: rgba(255,255,255,0.8);">Your intelligent campus companion</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Profile or avatar in sidebar
        st.markdown("""
        <div style="display: flex; justify-content: center; margin: 1rem 0;">
            <div style="width: 80px; height: 80px; border-radius: 50%; background: white; display: flex; align-items: center; justify-content: center; font-size: 2rem;">
                🤖
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Navigation menu with icons
        st.markdown("""
        <div style="margin: 2rem 0; color: white;">
            <h3 style="color: white; border-bottom: 1px solid rgba(255,255,255,0.2); padding-bottom: 0.5rem;">Navigation</h3>
        </div>
        """, unsafe_allow_html=True)
        
        app_mode = st.radio(
            "",
            ["🏛️ College Info", "📊 Student Marks"],
            key="navigation",
            label_visibility="collapsed"
        )
        
        # Add sidebar footer with version info
        st.markdown("""
        <div style="position: absolute; bottom: 20px; left: 20px; right: 20px; text-align: center; color: rgba(255,255,255,0.7);">
            <p style="font-size: 0.8rem;"></p>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content area
    if "🏛️" in app_mode:  # College Info mode
        # Show welcome animation only on first load
        if "welcome_shown" not in st.session_state:
            show_welcome_animation()
            st.session_state["welcome_shown"] = True
        
        st.markdown('<h1 class="fade-in">🏛️ RCEE College Interactive Assistant</h1>', unsafe_allow_html=True)
        #st.markdown("<div style='display: flex; align-items: center; gap: 15px;'><img src='C:/Users/rishi/Downloads/download.jpg' width='100'><h1>Welcome to RCEE Interactive Assistant</h1></div>", unsafe_allow_html=True)

        # Dashboard stats in cards
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class="info-card">
                <h3>🏆 Established</h3>
                <p style="font-size: 1.5rem; font-weight: bold;">2001</p>
                <p>Years of Academic Excellence</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="info-card">
                <h3>👨‍🎓 Students</h3>
                <p style="font-size: 1.5rem; font-weight: bold;">5000+</p>
                <p>Enrolled in Various Programs</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="info-card">
                <h3>👨‍🏫 Faculty</h3>
                <p style="font-size: 1.5rem; font-weight: bold;">300+</p>
                <p>Experienced Teaching Staff</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Initialize session state for chat messages and input for college chatbot
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "input" not in st.session_state:
            st.session_state.input = ""
        
        # Specify the PDF file paths for college information
        pdf_file_paths = [r"C:\Users\rishi\Desktop\GCP\RCEE.pdf"]
        
        # Process the specified PDF files once when the app starts
        raw_text = get_pdf_text(pdf_file_paths)
        text_chunks = get_text_chunks(raw_text)
        get_vector_store(text_chunks)
        
        # Chat container with custom styling
        st.markdown('<div class="fade-in">', unsafe_allow_html=True)
        
        # Chat header
        st.markdown("""
        <div style="background: white; padding: 1rem; border-radius: 10px 10px 0 0; border-bottom: 1px solid #e2e8f0; margin-bottom: 1rem;">
            <div style="display: flex; align-items: center;">
                <div style="width: 40px; height: 40px; border-radius: 50%; background: #4F46E5; display: flex; align-items: center; justify-content: center; margin-right: 1rem;">
                    <span style="color: white; font-weight: bold;">AI</span>
                </div>
                <div>
                    <h3 style="margin: 0;">RCEE Assistant</h3>
                    <p style="margin: 0; color: #64748b;">Ask me anything about the college</p>
                </div>
                <div style="margin-left: auto; background: #10b981; color: white; padding: 0.25rem 0.5rem; border-radius: 1rem; font-size: 0.8rem;">
                    Online
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Chat message container
        chat_container = st.container()
        
        with chat_container:
            # Display intro message if no messages
            if not st.session_state.messages:
                st.markdown("""
                <div class="assistant-bubble" style="background: #f8fafc; border-left: 4px solid #4F46E5;">
                    <h3 style="margin-top: 0;">Hello there! 👋</h3>
                    <p>I'm your RCEE College Assistant. I can answer questions about programs, faculty, facilities, and more. What would you like to know about RCEE?</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Display chat history
            for message in st.session_state.messages:
                if message["role"] == "user":
                    st.markdown(f'<div class="user-bubble">{message["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="assistant-bubble">{message["content"]}</div>', unsafe_allow_html=True)
        
        # Define suggestions for college chatbot with improved styling
        st.markdown('<h3 style="margin-top: 1.5rem;">Quick Questions:</h3>', unsafe_allow_html=True)
        
        suggestions = [
            "🎓 What programs are offered?",
            "👨‍🏫 Who is the principal?",
            "🌟 What is the vision of RCEE?",
            "🏛 Tell me about the college?",
            "📚 What academic accreditations does RCEE hold?"
        ]
        
        # Display suggestion buttons with custom styling
        suggestion_html = '<div style="display: flex; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 1.5rem;">'
        for suggestion in suggestions:
            suggestion_html += f'<button class="suggestion-button" onclick="this.style.backgroundColor=\'#eef2ff\'; document.querySelector(\'[data-testid=stChatInput]\')">{suggestion}</button>'
        suggestion_html += '</div>'
        
        st.markdown(suggestion_html, unsafe_allow_html=True)
        
        # Input box for user query with improved styling
        user_query = st.chat_input("Ask me about RCEE College...") or st.session_state.input
        
        if user_query:
            # Clear the input session state after question is asked
            st.session_state.input = ""
            
            # Add user message to session state
            st.session_state.messages.append({"role": "user", "content": user_query})
            
            # Display user message in the chat (done automatically via session state refresh)
            
            # Generate content based on user input using college chatbot logic with visual loading
            with st.status("Processing your question..."):
                st.write("Analyzing college data...")
                time.sleep(0.5)
                st.write("Generating response...")
                time.sleep(0.5)
                response = user_input(user_query)
            
            # Add assistant response to session state
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Force refresh to show new messages
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        
    # Student Marks mode    
    elif "📊" in app_mode:  # Student Marks mode
        st.markdown('<h1 class="fade-in">📊 Student Result Analysis System</h1>', unsafe_allow_html=True)
        
        # Create tabs for better organization
        tab1, tab2, tab3 = st.tabs(["📋 Results Analysis", "📈 Performance Trends", "❓ Help"])
        
        with tab1:
            # Improved batch selection UI
            st.markdown("""
            <div class="info-card">
                <h3>Select Your Batch</h3>
                <p>Choose your batch to access your academic records and analytics</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Dropdown menu with improved styling
            batch_selection = st.selectbox(
                'Batch:',
                ['Select Your Batch', 'AI & DS 2021-2025', 'AI & DS 2020-2024'],
                index=0,
                key="batch_select"
            )
            
            if batch_selection == 'Select Your Batch':
                # Show attractive info cards when no batch is selected
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("""
                    <div class="info-card" style="height: 100%;">
                        <h3>📊 Academic Performance Analysis</h3>
                        <p>Access your semester-wise performance data, subject scores, and overall grades.</p>
                        <ul>
                            <li>Comprehensive semester reports</li>
                            <li>Subject-wise breakdown</li>
                            <li>Performance comparisons</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("""
                    <div class="info-card" style="height: 100%;">
                        <h3>🔍 Smart Insights</h3>
                        <p>Ask questions about your academic performance and get AI-powered insights.</p>
                        <ul>
                            <li>Natural language queries</li>
                            <li>Personalized analytics</li>
                            <li>Performance improvement suggestions</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
            
            elif batch_selection == 'AI & DS 2021-2025':
                st.subheader(f"{batch_selection} Student Marks Portal")
                
                # Set fixed paths for CSV files specific to AI & DS 2021-2025 batch
                csv_paths_ai_ds_2021_2025 = {
                    "1-1": r"C:\Users\rishi\Desktop\GCP\2021\1-1sem.csv",
                    "1-2": r"C:\Users\rishi\Desktop\GCP\2021\1-2sem.csv",
                    "2-1": r"C:\Users\rishi\Desktop\GCP\2021\2-1sem.csv",
                    "2-2": r"C:\Users\rishi\Desktop\GCP\2021\2-2sem.csv",
                    "3-1": r"C:\Users\rishi\Desktop\GCP\2021\3-1sem.csv",
                    "3-2": r"C:\Users\rishi\Desktop\GCP\2021\3-2sem.csv",
                }
                
                # Improved semester selection with icons
                semester_options = ["1-1", "1-2", "2-1", "2-2", "3-1", "3-2"]
                semester_col1, semester_col2 = st.columns([1, 3])
                
                with semester_col1:
                    st.markdown("""
                    <div style="background: white; padding: 1rem; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);">
                        <h4 style="margin-top: 0;">Select Semester</h4>
                    </div>
                    """, unsafe_allow_html=True)
                
                with semester_col2:
                    semester = st.select_slider(
                        '',
                        options=semester_options,
                        label_visibility="collapsed"
                    )
                
                combined_data_2021_2025 = extract_csv(csv_paths_ai_ds_2021_2025[semester])
                
                # Create an improved query input box with placeholder and example
                st.markdown("""
                <div class="info-card">
                    <h3>🔍 Ask about your results</h3>
                    <p>Enter your roll number and ask about your performance. Try questions like:</p>
                    <ul>
                        <li>"Show my results for 21A91A6630"</li>
                        <li>"How did I perform in Mathematics? My roll number is 21A91A6630"</li>
                        <li>"What's my CGPA? Roll number: 21A91A6630"</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                
                user_question_with_roll_number = st.text_input(
                    "",
                    placeholder="Example: What are my marks in all subjects? My roll number is 21A91A6630",
                    label_visibility="collapsed"
                )
                
                # Add a fancy submit button
                submit_col1, submit_col2, submit_col3 = st.columns([1, 1, 1])
                
                with submit_col2:
                    submit_button = st.button("🚀 Analyze My Results", use_container_width=True)
                
                if submit_button:
                    if user_question_with_roll_number:
                        # Visual loading effect
                        display_loading_animation()
                        
                        roll_number_match = re.search(r'\b(20|21)[A-Z]{2}[0-9]{1}[A-Z]{1}[0-9]{4}\b', user_question_with_roll_number)
                        
                        if roll_number_match:
                            roll_number_input = roll_number_match.group(0)  # Extracted roll number
                            
                            if is_valid_roll_number(roll_number_input, combined_data_2021_2025):
                                # Show verification badge
                                st.markdown(f"""
                                <div style="display: flex; align-items: center; background: #f0fdf4; padding: 0.75rem; border-radius: 10px; margin-bottom: 1rem; border-left: 4px solid #10b981;">
                                    <div style="background: #10b981; color: white; border-radius: 50%; width: 24px; height: 24px; display: flex; align-items: center; justify-content: center; margin-right: 0.75rem;">✓</div>
                                    <div>
                                        <strong>Verified Student:</strong> {roll_number_input}
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                with st.spinner("Processing your request..."):
                                    model_2021_25 = genai.GenerativeModel(
                                        model_name="gemini-1.5-pro-latest",
                                        generation_config={
                                            "temperature": 1,
                                            "top_p": 0.95,
                                            "top_k": 0,
                                            "max_output_tokens": 8192,
                                        }
                                    )
                                    
                                    # Create prompt for student marks analysis
                                    prompt = f"""You are a helpful Student Marks Analyzer chatbot. 
                                    Based on the following CSV data for semester {semester}, answer the question about roll number {roll_number_input}.
                                    
                                    CSV DATA:
                                    {"".join(combined_data_2021_2025)}
                                    
                                    USER QUESTION: {user_question_with_roll_number}
                                    
                                    Provide a detailed and informative answer. Include specific marks, ranks, percentages, and comparisons where relevant.
                                    Format the response with clear sections and highlights for important information.
                                    If you cannot find information about this roll number or if the question cannot be answered from the data, politely say so.
                                    """
                                    
                                    response_2021_25 = model_2021_25.generate_content(prompt)
                                    
                                    # Display response in a formatted card
                                    st.markdown("""
                                    <div style="background: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 10px 25px rgba(0, 0, 0, 0.05); margin-top: 1rem; border-left: 5px solid #4F46E5;">
                                        <h3 style="margin-top: 0; color: #4F46E5;">📊 Analysis Results</h3>
                                    """, unsafe_allow_html=True)
                                    
                                    # Process and display formatted response
                                    formatted_response = response_2021_25.text
                                    formatted_response = formatted_response.replace("\n", "<br>")
                                    formatted_response = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', formatted_response)
                                    formatted_response = re.sub(r'\*(.*?)\*', r'<em>\1</em>', formatted_response)
                                    
                                    st.markdown(f"<div>{formatted_response}</div>", unsafe_allow_html=True)
                                    st.markdown("</div>", unsafe_allow_html=True)
                            else:
                                st.error(f"Roll number {roll_number_input} not found in the database for semester {semester}. Please check and try again.")
                        else:
                            st.warning("Please include a valid roll number in the format 21XXXAXXXX or 20XXXAXXXX in your question.")
                    else:
                        st.info("Please enter a question with your roll number to analyze your results.")
            
            elif batch_selection == 'AI & DS 2020-2024':
                st.subheader(f"{batch_selection} Student Marks Portal")
                
                # Set fixed paths for CSV files specific to AI & DS 2020-2024 batch
                csv_paths_ai_ds_2020_2024 = {
                    "1-1": r"C:\Users\rishi\Desktop\GCP\2020\1-1sem.csv",
                    "1-2": r"C:\Users\rishi\Desktop\GCP\2020\1-2sem.csv",
                    "2-1": r"C:\Users\rishi\Desktop\GCP\2020\2-1sem.csv",
                    "2-2": r"C:\Users\rishi\Desktop\GCP\2020\2-2sem.csv",
                    "3-1": r"C:\Users\rishi\Desktop\GCP\2020\3-1sem.csv",
                    "3-2": r"C:\Users\rishi\Desktop\GCP\2020\3-2sem.csv",
                    "4-1": r"C:\Users\rishi\Desktop\GCP\2020\4-1sem.csv",
                    "4-2": r"C:\Users\rishi\Desktop\GCP\2020\4-2sem.csv",
                }
                
                # Improved semester selection with icons
                semester_options_2020 = ["1-1", "1-2", "2-1", "2-2", "3-1", "3-2", "4-1", "4-2"]
                semester_col1, semester_col2 = st.columns([1, 3])
                
                with semester_col1:
                    st.markdown("""
                    <div style="background: white; padding: 1rem; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);">
                        <h4 style="margin-top: 0;">Select Semester</h4>
                    </div>
                    """, unsafe_allow_html=True)
                
                with semester_col2:
                    semester_2020 = st.select_slider(
                        '',
                        options=semester_options_2020,
                        label_visibility="collapsed"
                    )
                
                combined_data_2020_2024 = extract_csv(csv_paths_ai_ds_2020_2024[semester_2020])
                
                # Create an improved query input box with placeholder and example
                st.markdown("""
                <div class="info-card">
                    <h3>🔍 Ask about your results</h3>
                    <p>Enter your roll number and ask about your performance. Try questions like:</p>
                    <ul>
                        <li>"Show my results for 20A91A6630"</li>
                        <li>"How did I perform in Mathematics? My roll number is 20A91A6630"</li>
                        <li>"What's my CGPA? Roll number: 20A91A6630"</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                
                user_question_with_roll_number_2020 = st.text_input(
                    "",
                    placeholder="Example: What are my marks in all subjects? My roll number is 20A91A6630",
                    label_visibility="collapsed",
                    key="input_2020"
                )
                
                # Add a fancy submit button
                submit_col1_2020, submit_col2_2020, submit_col3_2020 = st.columns([1, 1, 1])
                
                with submit_col2_2020:
                    submit_button_2020 = st.button("🚀 Analyze My Results", use_container_width=True, key="submit_2020")
                
                if submit_button_2020:
                    if user_question_with_roll_number_2020:
                        # Visual loading effect
                        display_loading_animation()
                        
                        roll_number_match_2020 = re.search(r'\b(20|21)[A-Z]{2}[0-9]{1}[A-Z]{1}[0-9]{4}\b', user_question_with_roll_number_2020)
                        
                        if roll_number_match_2020:
                            roll_number_input_2020 = roll_number_match_2020.group(0)  # Extracted roll number
                            
                            if is_valid_roll_number(roll_number_input_2020, combined_data_2020_2024):
                                # Show verification badge
                                st.markdown(f"""
                                <div style="display: flex; align-items: center; background: #f0fdf4; padding: 0.75rem; border-radius: 10px; margin-bottom: 1rem; border-left: 4px solid #10b981;">
                                    <div style="background: #10b981; color: white; border-radius: 50%; width: 24px; height: 24px; display: flex; align-items: center; justify-content: center; margin-right: 0.75rem;">✓</div>
                                    <div>
                                        <strong>Verified Student:</strong> {roll_number_input_2020}
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                with st.spinner("Processing your request..."):
                                    model_2020_24 = genai.GenerativeModel(
                                        model_name="gemini-1.5-pro-latest",
                                        generation_config={
                                            "temperature": 1,
                                            "top_p": 0.95,
                                            "top_k": 0,
                                            "max_output_tokens": 8192,
                                        }
                                    )
                                    
                                    # Create prompt for student marks analysis
                                    prompt_2020 = f"""You are a helpful Student Marks Analyzer chatbot. 
                                    Based on the following CSV data for semester {semester_2020}, answer the question about roll number {roll_number_input_2020}.
                                    
                                    CSV DATA:
                                    {"".join(combined_data_2020_2024)}
                                    
                                    USER QUESTION: {user_question_with_roll_number_2020}
                                    
                                    Provide a detailed and informative answer. Include specific marks, ranks, percentages, and comparisons where relevant.
                                    Format the response with clear sections and highlights for important information.
                                    If you cannot find information about this roll number or if the question cannot be answered from the data, politely say so.
                                    """
                                    
                                    response_2020_24 = model_2020_24.generate_content(prompt_2020)
                                    
                                    # Display response in a formatted card
                                    st.markdown("""
                                    <div style="background: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 10px 25px rgba(0, 0, 0, 0.05); margin-top: 1rem; border-left: 5px solid #4F46E5;">
                                        <h3 style="margin-top: 0; color: #4F46E5;">📊 Analysis Results</h3>
                                    """, unsafe_allow_html=True)
                                    
                                    # Process and display formatted response
                                    formatted_response_2020 = response_2020_24.text
                                    formatted_response_2020 = formatted_response_2020.replace("\n", "<br>")
                                    formatted_response_2020 = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', formatted_response_2020)
                                    formatted_response_2020 = re.sub(r'\*(.*?)\*', r'<em>\1</em>', formatted_response_2020)
                                    
                                    st.markdown(f"<div>{formatted_response_2020}</div>", unsafe_allow_html=True)
                                    st.markdown("</div>", unsafe_allow_html=True)
                            else:
                                st.error(f"Roll number {roll_number_input_2020} not found in the database for semester {semester_2020}. Please check and try again.")
                        else:
                            st.warning("Please include a valid roll number in the format 20XXXAXXXX in your question.")
                    else:
                        st.info("Please enter a question with your roll number to analyze your results.")
                        
        with tab2:
            st.markdown('<h2 class="fade-in">📈 Performance Trends</h2>', unsafe_allow_html=True)
            
            # Placeholder for performance trends visualization
            st.markdown("""
            <div class="info-card">
                <h3>📊 Batch Performance Insights</h3>
                <p>Select your batch and semester above to view aggregate performance trends.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Create columns for layout
            chart_col1, chart_col2 = st.columns([1, 1])
            
            with chart_col1:
                st.markdown("""
                <div style="background: white; padding: 1rem; border-radius: 10px; height: 300px; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);">
                    <div style="text-align: center;">
                        <h3>Subject-wise Performance</h3>
                        <p>Select a batch and semester to view subject performance analysis</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with chart_col2:
                st.markdown("""
                <div style="background: white; padding: 1rem; border-radius: 10px; height: 300px; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);">
                    <div style="text-align: center;">
                        <h3>Class Distribution</h3>
                        <p>Select a batch and semester to view grade distribution</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            # Comparison feature
            st.markdown("""
            <div class="info-card" style="margin-top: 2rem;">
                <h3>💪 Compare Your Performance</h3>
                <p>Enter your roll number to see how you compare with your batch average</p>
                <p><em>Select a batch and semester in the Results Analysis tab first</em></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Placeholder for future comparison feature
            compare_col1, compare_col2 = st.columns([3, 1])
            
            with compare_col1:
                compare_roll = st.text_input("Your Roll Number", placeholder="e.g., 21A91A6630")
            
            with compare_col2:
                st.button("Compare", use_container_width=True)
                
        with tab3:
            st.markdown('<h2 class="fade-in">❓ Help & Guidance</h2>', unsafe_allow_html=True)
            
            # Help accordion
            with st.expander("How to use the Student Result Analysis System"):
                st.markdown("""
                1. **Select your batch** from the dropdown menu
                2. **Choose the semester** you want to analyze
                3. **Enter your question** including your roll number
                4. **Click "Analyze My Results"** to get your personalized analysis
                """)
            
            with st.expander("Example Questions You Can Ask"):
                st.markdown("""
                - "Show my results for 21A91A6630"
                - "What subjects did I score highest in? Roll number: 21A91A6630"
                - "How does my performance compare to the class average? My roll number is 21A91A6630"
                - "What was my rank in Mathematics? Roll: 21A91A6630"
                - "Show me my progress across all semesters. Roll: 21A91A6630"
                """)
            
            with st.expander("Troubleshooting"):
                st.markdown("""
                **Roll Number Not Found**
                - Ensure you're using the correct format (e.g., 21A91A6630)
                - Verify you've selected the correct batch and semester
                - Check for typos in your roll number
                
                **No Response**
                - Ensure your question includes your roll number
                - Try rephrasing your question more specifically
                - Check your internet connection
                """)
                
            # Contact information
            st.markdown("""
            <div class="info-card" style="margin-top: 2rem;">
                <h3>📞 Need Additional Help?</h3>
                <p>Contact the Student Support Cell:</p>
                <ul>
                    <li>Email: support@rcee.edu.in</li>
                    <li>Phone: +91 98765 43210</li>
                    <li>Location: Academic Block, Room 103</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

# Run the Streamlit app
if __name__ == "__main__":
    main()