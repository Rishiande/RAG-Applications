import os
import streamlit as st
import google.generativeai as genai

# Set up Google API Key
GOOGLE_API_KEY = "AIzaSyCWmWlwM4R3Otqp0Go51z9EVCNfEgWa2rM"  # Replace with your actual API key
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
genai.configure(api_key=GOOGLE_API_KEY)

# Streamlit Application
def main():
    st.title("Enhancing Medical Assistance: A BERT powered Chatbot")

    # User Input
    symptoms = st.text_input("Enter your symptoms (e.g., headache, fever, fatigue):")

    # Submit Button
    if st.button("Submit Symptoms"):
        if symptoms.strip():
            # Format prompt to include request for probabilities
            prompt = f"""
            You are a highly intelligent medical assistant. A user has described their symptoms as follows:
            Symptoms: {symptoms}

            Please:
            1. Predict the most likely diseases based on the symptoms along with their probability scores.
            2. Provide precautions to prevent worsening of the diseases.
            3. Suggest possible medications (common over-the-counter or prescription medications).
            """

            # Generate response
            st.write("Analyzing your symptoms...")
            try:
                model = genai.GenerativeModel('gemini-pro')  # Use a valid model name here
                response = model.generate_content(prompt)  # Use generate_content instead of generate_text
                
                # Display results
                st.success("Prediction Results:")
                result_text = response.text  # Adjust based on actual response structure
                
                # Displaying the response text
                st.write(result_text)
                
                # Add a warning about medications
                st.warning("**Note:** Medications suggested are optional. Please use cautiously and consult a healthcare professional before starting any new medication.")

            except Exception as e:
                st.error(f"Error generating response: {e}")
        else:
            st.error("Please enter symptoms to proceed.")

if __name__ == "__main__":
    main()
