To run the chatbot, enter “streamlit run app.py” in the terminal in the folder with app.py. 

To change the RAG model, at the very top of the streamlit UI, there is a drop down with FAISS, Cosine, Manhattan, and Euclidean.

Please, note you would need to create you own hugging face token when you clone the REPO. 
HF_TOKEN = "" <-- in app.py
The follow permissions need to be turned on
Inference
Make calls to Inference Providers
Make calls to your Inference Endpoints
Manage your Inference Endpoints
