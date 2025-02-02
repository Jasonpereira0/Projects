# **E-commerce FAQ Chatbot with Hugging Face Integration**

## **Project Overview**
This project implements an E-commerce FAQ chatbot using **Hugging Face Transformers** and **Gradio**. The chatbot provides answers to frequently asked questions (FAQs) about e-commerce operations, such as order tracking, payment methods, and return policies. It combines structured dataset search with AI-generated fallback responses to handle both known and unknown queries.

---

## **Features**
1. **Hybrid Response System:**
   - Searches for answers in a pre-defined FAQ dataset.
   - Generates fallback responses using a pre-trained language model (e.g., GPT-2) when no matching question is found.

2. **User-Friendly Interface:**
   - Built using Gradio, the chatbot provides an intuitive web-based interface for user interaction.

3. **Pre-trained Language Model:**
   - Uses Hugging Face Transformers to integrate a conversational model for generating fallback responses.

4. **Scalable Design:**
   - Easily extendable with larger datasets or more powerful models like GPT-Neo or GPT-J.

---

## **Technologies Used**
- **Python:** Core programming language.
- **Hugging Face Transformers:** For integrating pre-trained language models (e.g., GPT-2).
- **Gradio:** For creating an interactive web-based chatbot interface.
- **Pandas:** For processing the FAQ dataset.
- **PyTorch:** Backend framework for running Hugging Face models.

---

## **Dataset**
The chatbot uses a JSON dataset (`Ecommerce_FAQ_Chatbot_dataset.json`) containing frequently asked questions and their corresponding answers. The dataset is loaded into a pandas DataFrame, where:
- The `question` field is renamed to `Question`.
- The `answer` field is renamed to `Answer`.

### Example Dataset Structure:
{
"questions":
[
{ "question": "How can I create a
account?", "answer": "To create an account, click on the 'Sign Up' button on the top right corner of our website and follow
he
i
structions." }, { "question": "What
ayment methods do you accept?", "answer": "We accept major credit cards, debit cards, and PayPal
s
p
y


---

## **How It Works**

### **1. Dataset Search**
The chatbot first searches the dataset for a matching question:
- If a match is found, it returns the corresponding answer from the dataset.
- This ensures accurate responses for known queries.

### **2. Fallback Response Generation**
If no match is found in the dataset:
- The query is passed to a Hugging Face pre-trained language model (e.g., GPT-2) for generating a response.
- This allows the chatbot to handle unknown or out-of-scope queries.

### **3. Gradio Interface**
The chatbot uses Gradio to provide an interactive web interface:
- Users can type their queries into a text input box.
- The chatbot processes the query and displays the response in a text output box.

---

## **Code Structure**

### Step 1: Load and Process JSON Dataset
The JSON dataset is loaded and converted into a pandas DataFrame:
with open(file_path, 'r', encoding='utf-8') as file:
data = json.load(file)
qa_pairs = data.get('questions', [])
df = pd.DataFrame(qa_pairs)
df.rename(columns={'question': 'Question', 'answer': 'Answer'}, inplace=True)


### Step 2: Set Up Hugging Face Transformers
A pre-trained language model (e.g., GPT-2) is loaded using Hugging Face Transformers:
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

### Step 3: Build Chatbot Logic
The chatbot searches the dataset for matching questions or generates fallback responses using the language model:
def chatbot_response(user_query):
for index, row in df.iterrows():
if user_query.lower() in row["Question"].lower():
return f"Answer: {row['Answer']}"

### Step 4: Create Gradio Interface
Gradio provides an interactive web interface for user interaction:
interface = gr.Interface(
fn=chatbot_interface,
inputs="text",
outputs="text",
title="E-commerce FAQ Chatbot with Hugging Face Integration",
description="Ask me anything about products, orders, or support!"
)
interface.launch()

---

## **Installation**

### Prerequisites
1. Python 3.8 or higher.
2. Required libraries:
   - `transformers`
   - `pandas`
   - `gradio`
   - `torch`

### Install Dependencies
Run the following command to install all required libraries: 
pip install transformers pandas gradio torch

---

## **Usage**

1. Clone this repository:
git clone https://github.com/JasonPereira0/Projects/chatbot_project.git
cd chatbot_project

2. Place your JSON dataset (`Ecommerce_FAQ_Chatbot_dataset.json`) in the `data/` folder.

3. Run the Python script: python chatbot.py


4. Open the Gradio interface in your browser and interact with the chatbot!

---

## **Example Queries**

### Query 1: Dataset Match
**Input:**  
`How can I create an account?`

**Output:**  
`Answer: To create an account, click on the 'Sign Up' button on the top right corner of our website and follow the instructions.`

---

### Query 2: Fallback Response
**Input:**  
`What is your shipping policy?`

**Output:**  
`Our shipping policy depends on your location and selected shipping method.` *(or another AI-generated response)*

---

## **Limitations**

1. **Model Limitations:**
- GPT-2 may generate less coherent fallback responses compared to larger models like GPT-Neo or GPT-J.
- Fine-tuning on domain-specific data can improve performance but requires additional resources.

2. **Dataset Dependency:**
- The accuracy of answers depends on the quality and coverage of the FAQ dataset.

3. **Performance Constraints:**
- Larger models require more computational resources (e.g., GPU).
- Response generation may be slower for complex queries or large datasets.

---

## **Future Improvements**

1. **Fine-Tune Pre-trained Models:**
- Fine-tune GPT-2 or larger models on the FAQ dataset for more accurate fallback responses.

2. **Use Larger Models:**
- Replace GPT-2 with more powerful models like `"EleutherAI/gpt-neo"` or `"GPT-J"` for better response quality.

3. **Deploy Online:**
- Host the chatbot on platforms like Streamlit Cloud, Hugging Face Spaces, or AWS Lambda for broader accessibility.

4. **Add Contextual Memory:**
- Enhance the chatbot by maintaining conversation history using frameworks like Rasa or LangChain.

5. **Improve Dataset Coverage:**
- Expand the FAQ dataset to cover more queries and reduce reliance on fallback responses.

---

## **Acknowledgments**
This project uses open-source tools provided by:
- [Hugging Face](https://huggingface.co/)
- [Gradio](https://gradio.app/)

Special thanks to Hugging Face's community for providing pre-trained models like GPT-2!

---

## **Contact**
For any questions or suggestions, feel free to reach out:

**Name:** Your Name  
**Email:** your.email@example.com  
**GitHub:** [yourusername](https://github.com/yourusername)  


