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

# Directly insert your Google API key here
GOOGLE_API_KEY = "AIzaSyDOrv3RayLX8j0B9C_cWwncoDjVfVHwZds"  # Replace with your actual API key

# Set up Google Generative AI
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

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
    response = chain({"input_documents": docs, "question": normalized_question}, return_only_outputs=True)
    
    return response["output_text"]

# Function to extract data from CSV files for student marks analysis
def extract_csv(pathname: str) -> list[str]:
    parts = []
    try:
        with open(pathname, "r", newline="") as csvfile:
            csv_reader = csv.reader(csvfile)
            header = next(csv_reader)
            parts.append(",".join(header))  # Add header to data
            for row in csv_reader:
                parts.append(",".join(row))
        return parts
    except FileNotFoundError:
        st.error(f"File not found: {pathname}")
        return []
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return []

# Function to validate roll number format and existence in data
def is_valid_roll_number(roll_number: str, combined_data: list) -> bool:
    pattern = r'^(20|21)[A-Z]{2}[0-9]{1}[A-Z]{1}[0-9]{4}$'
    
    if re.match(pattern, roll_number):
        return any(roll_number in entry for entry in combined_data)
    
    return False

# Function to load backlog data from CSV
def load_backlog_data(path: str) -> list[dict]:
    backlog_data = []
    with open(path, "r", newline="") as csvfile:
        csv_reader = csv.DictReader(csvfile)
        for row in csv_reader:
            backlog_data.append(row)
    return backlog_data

# Streamlit app UI setup
def main():
    st.set_page_config(page_title="🎓 College & Student Marks Chatbot", layout="wide")
    
    # Sidebar navigation options
    st.sidebar.title("Navigation Bar")
    app_mode = st.sidebar.selectbox("Choose an option:", ["College Info", "Student Marks", "Backlogs Comparison"])
    
    if app_mode == "College Info":
        st.title("🎓 RCEE College Chatbot")
        
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

        # Define suggestions for college chatbot suggestions
        suggestions = [
            "🎓 What programs are offered?",
            "👨‍🏫 Who is the principal?",
            "🌟 What is the vision of RCEE?",
            "🏛 Tell me about the college?",
            "📚 What academic accreditations does RCEE hold?"
        ]
        
        # Create a fixed container for quick suggestions at the top
        with st.container():
            st.subheader("Quick Suggestions:")
            cols = st.columns(len(suggestions))
            
            for i, suggestion in enumerate(suggestions):
                if cols[i].button(suggestion):
                    st.session_state["input"] = suggestion
        
        # Display chat history for college chatbot
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Input box for user query at the bottom right, pre-filled with suggestion if clicked
        prompt = st.chat_input("Ask me anything about RCEE (e.g., 'What programs are offered?')") or st.session_state.input
        
        if prompt:
            # Clear the input session state after question is asked
            st.session_state.input = ""
            
            # Add user message to session state
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message in the chat 
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate content based on user input using college chatbot logic 
            response = user_input(prompt)

            # Display assistant response in the chat 
            with st.chat_message("assistant"):
                st.markdown(response)

            # Add assistant response to session state 
            st.session_state.messages.append({"role": "assistant", "content": response})

    elif app_mode == "Student Marks":
        st.title("Student Batch Wise Results Analysis")
        
        # Dropdown menu for selecting specific batch under Student Marks section 
        batch_selection = st.sidebar.selectbox(
            'Select Batch:',
            ['Select Batch', 'AI & DS 2021-2025', 'AI & DS 2020-2024']
        )
        
        if batch_selection == 'AI & DS 2021-2025':
            st.subheader(f"{batch_selection} Student Marks Chatbot")
            
            # Set fixed paths for CSV files specific to AI & DS 2021-2025 batch 
            csv_paths_ai_ds_2021_2025 = {
                "1-1": r"C:\Users\rishi\Desktop\GCP\2021\1-1sem.csv",
                "1-2": r"C:\Users\rishi\Desktop\GCP\2021\1-2sem.csv",
                "2-1": r"C:\Users\rishi\Desktop\GCP\2021\2-1sem.csv",
                "2-2": r"C:\Users\rishi\Desktop\GCP\2021\2-2sem.csv",
                "3-1": r"C:\Users\rishi\Desktop\GCP\2021\3-1sem.csv",
                "3-2": r"C:\Users\rishi\Desktop\GCP\2021\3-2sem.csv",
            }

            semester_options = ["1-1", "1-2", "2-1", "2-2", "3-1", "3-2"]
            semester = st.sidebar.selectbox('Select Semester:', semester_options)

            combined_data_2021_2025 = extract_csv(csv_paths_ai_ds_2021_2025[semester])
            
            user_question_with_roll_number= st.text_input("Ask a question about student results (include your Roll Number):")
            
            if st.button("🚀 Submit"):
                if user_question_with_roll_number:
                    roll_number_match = re.search(r'\b(20|21)[A-Z]{2}[0-9]{1}[A-Z]{1}[0-9]{4}\b', user_question_with_roll_number)
                    
                    if roll_number_match:
                        roll_number_input= roll_number_match.group(0)  # Extracted roll number 
                        
                        if is_valid_roll_number(roll_number_input, combined_data_2021_2025):
                            model_2021_25= genai.GenerativeModel(
                                model_name="gemini-1.5-pro-latest",
                                generation_config={
                                    "temperature": 1,
                                    "top_p": 0.95,
                                    "top_k": 0,
                                    "max_output_tokens": 8192,
                                },
                                safety_settings=[
                                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                                ]
                            )
                            
                            messages_history_2021_25= [{'role': 'user', 'parts': combined_data_2021_2025}]
                            messages_history_2021_25.append({'role': 'user', 'parts': [user_question_with_roll_number]})
                            
                            response_2021_25= model_2021_25.generate_content(messages_history_2021_25)
                            
                            if hasattr(response_2021_25, 'candidates') and response_2021_25.candidates:
                                candidate_output= response_2021_25.candidates[0]
                                output_text_parts= candidate_output.content.parts
                                output_text_lines= [part.text for part in output_text_parts]
                                output_text_cleaned= "\n".join(output_text_lines).strip()
                                
                                st.write(output_text_cleaned)
                            else:
                                st.warning("No response was generated. Please try again.")
                        else:
                            st.error("""Sorry, the roll number you entered is incorrect. Please double-check and re-enter it carefully.""")
                    else:
                        st.error("""It looks like you forgot to include your roll number in your question. Please make sure to mention it!""")
                else:
                    st.warning("Please enter a question before submitting.")

        elif batch_selection == 'AI & DS 2020-2024':
            st.subheader(f"{batch_selection} Student Marks Chatbot")

            # Set fixed paths for CSV files specific to AI & DS 2020-2024 batch 
            csv_paths_ai_ds_2020_2024 = {
                "1-1": r"C:/Users/rishi/Desktop/GCP/2020/1-1sems.csv",
                "1-2": r"C:/Users/rishi/Desktop/GCP/2020/1-2sems.csv",
                "2-1": r"C:/Users/rishi/Desktop/GCP/2020/2-1sems.csv",
                "2-2": r"C:/Users/rishi/Desktop/GCP/2020/2-2sems.csv",
                "3-1": r"C:/Users/rishi/Desktop/GCP/2020/3-1sems.csv",
                "3-2": r"C:/Users/rishi/Desktop/GCP/2020/3-2sems.csv"
            }

            combined_data_2020_24 = []
            for path in csv_paths_ai_ds_2020_2024.values():
                combined_data_2020_24.extend(extract_csv(path))

            user_question_with_roll_number_to24= st.text_input("Ask a question about student results (include your Roll Number):")
            
            if st.button("🚀 Submit"):
                if user_question_with_roll_number_to24:
                    roll_number_match_to24= re.search(r'\b(20|21)[A-Z]{2}[0-9]{1}[A-Z]{1}[0-9]{4}\b', user_question_with_roll_number_to24)
                    
                    if roll_number_match_to24:
                        roll_number_input_to24= roll_number_match_to24.group(0) 
                        
                        if is_valid_roll_number(roll_number_input_to24, combined_data_2020_24):
                            model_to24= genai.GenerativeModel(
                                model_name="gemini-1.5-pro-latest",
                                generation_config={
                                    "temperature": 1,
                                    "top_p": 0.95,
                                    "top_k": 0,
                                    "max_output_tokens": 8192,
                                },
                                safety_settings=[
                                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                                ]
                            )
                            
                            messages_history_to24= [{'role': 'user', 'parts': combined_data_2020_24}]
                            messages_history_to24.append({'role': 'user', 'parts': [user_question_with_roll_number_to24]})
                            
                            response_to24= model_to24.generate_content(messages_history_to24)

                            if hasattr(response_to24, 'candidates') and response_to24.candidates:
                                candidate_output_to24= response_to24.candidates[0]
                                output_text_parts_to24= candidate_output_to24.content.parts
                                output_text_lines_to24= [part.text for part in output_text_parts_to24]
                                output_text_cleaned_to24= "\n".join(output_text_lines_to24).strip()
                                
                                st.write(output_text_cleaned_to24)
                            else:
                                st.warning("No response was generated. Please try again.")
                        else:
                            st.error("""Sorry, the roll number you entered is incorrect. Please double-check and re-enter it carefully.""")
                    else:
                        st.error("""It looks like you forgot to include your roll number in your question. Please make sure to mention it!""")
                else:
                    st.warning("Please enter a question before submitting.")

    elif app_mode == "Backlogs Comparison":
        st.title("Backlogs Comparison")
        
        # Hardcoded CSV file path for backlog data
        csv_path = r"C:\Users\rishi\Desktop\Vijaya\Backlog.csv"
        
        # Extract CSV data
        csv_data = extract_csv(csv_path)
        
        if csv_data:
            st.subheader("Backlog Data")
            st.write(csv_data)  # Display the CSV data for reference
            
            # Initialize the Gemini model for Backlogs Comparison
            model = genai.GenerativeModel(
                model_name="gemini-1.5-flash-002",
                generation_config={
                    "temperature": 1,
                    "top_p": 0.95,
                    "top_k": 40,
                    "max_output_tokens": 8192,
                },
                safety_settings=[
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                ]
            )
            
            # User Input
            user_question = st.text_input("💬 Ask a question about student backlogs (include Roll Number):")
            
            if st.button("🚀 Submit"):
                if user_question:
                    # Advanced Prompt
                    prompt = f"""
You are an intelligent academic assistant with access to student backlog data. Your task is to analyze and provide detailed information based on the provided CSV data. The data includes student roll numbers, cleared subjects, supply subjects, and grades.

Below is the CSV data:

{csv_data}

TASK:
1. **Identify the Roll Number**: Extract the roll number mentioned in the user's question.
2. **Retrieve Student Data**: Locate the row corresponding to the roll number in the CSV data.
3. **Cleared Subjects**:
   - Count the number of cleared subjects.
   - List the names of the cleared subjects.
   - Provide the grades for each cleared subject.
4. **Supply Subjects**:
   - Count the number of supply subjects.
   - List the names of the supply subjects.
   - Provide the grades for each supply subject (if available).
5. **Summary**:
   - Provide a summary of the student's academic performance.
   - Highlight any critical issues (e.g., too many supply subjects, low grades).
6. **Formatting**:
   - Use clear headings and bullet points for better readability.
   - Ensure the response is well-structured and easy to understand.

ADDITIONAL INSTRUCTIONS:
- If the roll number is not found in the data, explicitly state that the roll number does not exist in the dataset.
- If the data is incomplete or missing, mention this in the response.
- Avoid unnecessary explanations or assumptions. Stick to the facts provided in the CSV data.

QUESTION: {user_question}

RESPONSE FORMAT:
**Roll Number**: [Roll Number]
**Cleared Subjects**:
- [Subject 1]: [Grade]
- [Subject 2]: [Grade]
- ...
**Number of Cleared Subjects**: [Count]
**Supply Subjects**:
- [Subject 1]: [Grade]
- [Subject 2]: [Grade]
- ...
**Number of Supply Subjects**: [Count]
**Summary**: [Brief summary of the student's performance]
"""

                    # Start conversation with CSV data
                    convo = model.start_chat(history=[
                        {"role": "user", "parts": [prompt]}
                    ])

                    response = convo.send_message(user_question)

                    # Display Response
                    if response and hasattr(response, 'candidates') and response.candidates:
                        candidate_output = response.candidates[0]
                        output_text_parts = candidate_output.content.parts
                        output_text_lines = [part.text for part in output_text_parts]
                        output_text_cleaned = "\n".join(output_text_lines).strip()

                        st.success("### ✅ Response:")
                        st.write(output_text_cleaned)
                    else:
                        st.warning("⚠️ No response generated. Please try again.")
                else:
                    st.warning("⚠️ Please enter a question before submitting.")
        else:
            st.error("❌ No data loaded from CSV. Please check the file path.")

# Run the app 
if __name__ == "__main__":
    main()
