# College & Student Marks Chatbot

## Overview
This project is a chatbot application designed to provide information about a college and analyze student marks based on user queries. The application utilizes Google Generative AI for natural language processing and supports interaction through a user-friendly web interface built with Streamlit.

## Features
- **College Information**: Users can ask questions about college programs, faculty, vision, and accreditations.
- **Student Marks Analysis**: Users can inquire about their academic results by providing their roll number.
- **PDF and CSV Integration**: The application extracts information from PDF files for college details and CSV files for student marks.

## Technologies Used
- Python
- Streamlit
- PyPDF2
- Langchain
- Google Generative AI
- FAISS (Facebook AI Similarity Search)
- Regular Expressions (re)
- CSV Handling

## Setup Instructions

### Prerequisites
- Python 3.7 or higher
- Google API Key for Google Generative AI

### Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
2. Install the required packages:
   pip install streamlit PyPDF2 langchain langchain-google-genai google-generativeai
3. GOOGLE_API_KEY = "YOUR_GOOGLE_API_KEY"  # Replace with your actual API key
4. Running the Application
To run the application, execute the following command in your terminal:
streamlit run studentmarks.py


Usage
1.Open your web browser and navigate to the URL provided in the terminal (usually http://localhost:8501).
2.Choose between "College Info" and "Student Marks" from the sidebar.
3.For college information, type your question in the input box or select from quick suggestions.
4.For student marks analysis, select your batch and enter your roll number along with your question.

Roll Number Format
The roll number should follow the format: 20ME1A5403 or 21ME..., where:
The first two digits represent the year of admission.
The next two letters represent the department.
The following digit and letter represent the section.
The last four digits are the unique identifier.

Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue for any enhancements or bug fixes.
