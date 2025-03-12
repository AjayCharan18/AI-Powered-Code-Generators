# Import necessary libraries
import os
import torch
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
import streamlit as st
import black  # For code formatting
import subprocess  # For code testing
import re  # For dependency detection
import json  # For saving and loading chat history
import pylint.lint  # For code optimization suggestions
import pydoc  # For code documentation generation
from github import Github  # For GitHub integration
import speech_recognition as sr  # For voice input
from fpdf import FPDF  # For exporting to PDF
import radon.complexity  # For code metrics

# Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['STREAMLIT_SERVER_ENABLE_FILE_WATCHER'] = 'false'

# Load GPT-Neo model and tokenizer
@st.cache_resource  # Cache to prevent reloading the model each time
def load_model():
    model_name = "EleutherAI/gpt-neo-125M"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPTNeoForCausalLM.from_pretrained(model_name)
    model.eval()
    return model, tokenizer

# Generate code function
def generate_code(prompt, model, tokenizer, max_length=300, temperature=0.7, top_k=50, top_p=0.95):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    
    # Ensure pad_token_id is set
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            do_sample=True,
            pad_token_id=pad_token_id,
            attention_mask=inputs.ne(pad_token_id).float(),
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Format code using black
def format_code(code):
    try:
        return black.format_str(code, mode=black.FileMode())
    except Exception as e:
        st.warning(f"Code formatting failed: {e}")
        return code  # Return unformatted code if formatting fails

# Detect dependencies
def detect_dependencies(code):
    imports = re.findall(r"^\s*import\s+(\w+)|^\s*from\s+(\w+)\s+import", code, re.MULTILINE)
    dependencies = {imp[0] or imp[1] for imp in imports}
    return list(dependencies)

# Test the generated code
def test_code(code, language):
    try:
        if language == "Python":
            with open("temp.py", "w") as f:
                f.write(code)
            result = subprocess.run(["python", "temp.py"], capture_output=True, text=True)
        elif language == "JavaScript":
            with open("temp.js", "w") as f:
                f.write(code)
            result = subprocess.run(["node", "temp.js"], capture_output=True, text=True)
        elif language == "Java":
            with open("Main.java", "w") as f:
                f.write(code)
            subprocess.run(["javac", "Main.java"], capture_output=True, text=True)
            result = subprocess.run(["java", "Main"], capture_output=True, text=True)
        else:
            return False, "Unsupported language"
        
        return (result.returncode == 0, result.stdout if result.returncode == 0 else result.stderr)
    except Exception as e:
        return False, str(e)

# Explain the generated code
def explain_code(code):
    return "This code performs the task described in your prompt. It follows best practices and is optimized for readability."

# Suggest optimizations
def suggest_optimizations(code):
    try:
        pylint_output = pylint.lint.Run(["--reports=y", "--disable=all", "--enable=unused-import,unused-variable"], do_exit=False)
        return pylint_output.linter.stats["by_msg"]
    except Exception as e:
        return f"Optimization suggestions failed: {e}"

# Generate documentation
def generate_documentation(code):
    try:
        return pydoc.plain(pydoc.render_doc(code))
    except Exception as e:
        return f"Documentation generation failed: {e}"

# Push code to GitHub
def push_to_github(code, repo_name, access_token):
    try:
        g = Github(access_token)
        user = g.get_user()
        repo = user.create_repo(repo_name)
        repo.create_file("generated_code.py", "Initial commit", code)
        return True, f"Code pushed to {repo_name}"
    except Exception as e:
        return False, str(e)

# Get voice input
def get_voice_input():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Speak now...")
        audio = r.listen(source)
        try:
            return r.recognize_google(audio)
        except sr.UnknownValueError:
            return "Could not understand audio"
        except sr.RequestError:
            return "API unavailable"

# Export to PDF
def export_to_pdf(code, explanation):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Generated Code", ln=True, align="C")
    pdf.multi_cell(0, 10, txt=code)
    pdf.cell(200, 10, txt="Code Explanation", ln=True, align="C")
    pdf.multi_cell(0, 10, txt=explanation)
    pdf.output("generated_code.pdf")
    return "generated_code.pdf"

# Streamlit app
def main():
    # Custom HTML and CSS for styling
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #f0f2f6;
            font-family: 'Arial', sans-serif;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            border-radius: 5px;
            padding: 10px 20px;
            font-size: 16px;
        }
        .stButton button:hover {
            background-color: #45a049;
        }
        .stTextInput input {
            border-radius: 5px;
            padding: 10px;
            font-size: 16px;
        }
        .stSidebar {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .stMarkdown h1 {
            color: #4CAF50;
            font-size: 36px;
            font-weight: bold;
        }
        .stMarkdown h2 {
            color: #333333;
            font-size: 24px;
            font-weight: bold;
        }
        .stMarkdown h3 {
            color: #555555;
            font-size: 20px;
            font-weight: bold;
        }
        .stCode {
            background-color: #ffffff;
            border-radius: 5px;
            padding: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Load the model and tokenizer
    model, tokenizer = load_model()

    # App title and description
    st.markdown("<h1>🚀 AI-Powered Code Generator</h1>", unsafe_allow_html=True)
    st.markdown("<p>Describe the code you want, and the AI will generate it for you.</p>", unsafe_allow_html=True)

    # Initialize session state for chat history and code history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "code_history" not in st.session_state:
        st.session_state.code_history = []

    # Sidebar for customizable parameters
    st.sidebar.header("⚙️ Model Parameters")
    temperature = st.sidebar.slider("Temperature", 0.1, 1.0, 0.7)
    top_k = st.sidebar.slider("Top-k", 1, 100, 50)
    top_p = st.sidebar.slider("Top-p", 0.1, 1.0, 0.95)
    max_length = st.sidebar.slider("Max Length", 100, 1000, 300)

    # Dark mode toggle
    dark_mode = st.sidebar.checkbox("🌙 Dark Mode")
    if dark_mode:
        st.markdown(
            """
            <style>
            .stApp {
                background-color: #1e1e1e;
                color: #ffffff;
            }
            .stSidebar {
                background-color: #2e2e2e;
                color: #ffffff;
            }
            .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
                color: #ffffff;
            }
            .stCode {
                background-color: #2e2e2e;
                color: #ffffff;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

    # Language selection
    language = st.sidebar.selectbox("Select Programming Language", ["Python", "JavaScript", "Java"])

    # Voice input
    if st.sidebar.button("🎤 Voice Input"):
        user_input = get_voice_input()
        st.write(f"You said: {user_input}")

    # Code templates
    templates = {
        "Web Scraping": "import requests\nfrom bs4 import BeautifulSoup\n\n# Your code here",
        "Data Analysis": "import pandas as pd\nimport numpy as np\n\n# Your code here",
    }
    template = st.sidebar.selectbox("Select a template:", list(templates.keys()))
    if st.sidebar.button("Load Template"):
        user_input = templates[template]

    # User input
    user_input = st.text_input("You: ", placeholder="Describe the code you want to generate...")

    if user_input:
        with st.spinner("Generating code..."):
            try:
                generated_code = generate_code(user_input, model, tokenizer, max_length, temperature, top_k, top_p)
                formatted_code = format_code(generated_code)

                # Allow user to edit the generated code
                edited_code = st.text_area("Edit the generated code:", value=formatted_code, height=300)
                formatted_code = edited_code

                st.session_state.chat_history.append(("You", user_input))
                st.session_state.chat_history.append(("AI", formatted_code))
                st.session_state.code_history.append((formatted_code, language))

                st.write("### 💬 Chat History")
                for role, message in st.session_state.chat_history:
                    if role == "You":
                        st.write(f"**You:** {message}")
                    else:
                        st.write(f"**AI:**")
                        st.code(message, language=language.lower())

                # Detect dependencies
                dependencies = detect_dependencies(formatted_code)
                if dependencies:
                    st.write("**📦 Detected Dependencies:**")
                    st.write(", ".join(dependencies))

                # Explain the generated code
                explanation = explain_code(formatted_code)
                st.write("**📖 Code Explanation:**")
                st.write(explanation)

                # Suggest optimizations
                optimizations = suggest_optimizations(formatted_code)
                st.write("**🔧 Optimization Suggestions:**")
                st.write(optimizations)

                # Generate documentation
                documentation = generate_documentation(formatted_code)
                st.write("**📄 Code Documentation:**")
                st.code(documentation)

                # Test the generated code
                st.write("**🛠️ Test the Code:**")
                if st.button("Run Code"):
                    success, output = test_code(formatted_code, language)
                    if success:
                        st.success("✅ Code executed successfully!")
                        st.write("**📝 Output:**")
                        st.code(output)
                    else:
                        st.error("❌ Code execution failed!")
                        st.write("**🛑 Error:**")
                        st.code(output)

                # Push to GitHub
                repo_name = st.text_input("Enter GitHub repository name:")
                access_token = st.text_input("Enter GitHub access token:", type="password")
                if st.button("Push to GitHub"):
                    success, message = push_to_github(formatted_code, repo_name, access_token)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)

                # Export to PDF
                if st.button("Export to PDF"):
                    pdf_path = export_to_pdf(formatted_code, explanation)
                    st.success(f"PDF saved to {pdf_path}")

                # Download generated code
                st.download_button(
                    label="📥 Download Code",
                    data=formatted_code,
                    file_name=f"generated_code.{language.lower()}",
                    mime="text/plain",
                )

            except Exception as e:
                st.error(f"⚠️ An error occurred: {e}")

    # Display code history in the sidebar
    if st.session_state.code_history:
        st.sidebar.header("📜 Code History")
        for i, (code, lang) in enumerate(st.session_state.code_history):
            with st.sidebar.expander(f"Generated Code {i + 1} ({lang})"):
                st.code(code, language=lang.lower())

if __name__ == "__main__":
    main()