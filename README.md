# Deep Learning for Comment Toxicity Detection  

🚀 A deep learning–based project that detects toxic comments such as **toxic, severe toxic, obscene, threat, insult, and identity hate**.  
This project uses **TensorFlow/Keras**, trained on the [Jigsaw Toxic Comment Classification Dataset](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge), and deployed with **Streamlit** for interactive testing.  

---

## 📂 Project Structure
Deep-Learning-for-Comment-Toxicity-Detection/
│── app/ # Streamlit app files
│── data/ # Dataset (train/test CSVs)
│── notebooks/ # Jupyter notebooks (EDA, experiments)
│── src/ # Python source code
│── train_model.py # Model training script
│── requirements.txt # Dependencies
│── Deployment_Guide.md # Deployment steps
│── README.md # Project documentation
## ⚡ Features
- Preprocessing of raw comment text (cleaning, tokenization, padding).  
- Deep Learning model using **LSTM/GRU/Embedding layers**.  
- Multi-label classification for 6 toxicity categories.  
- Training pipeline with model saving/loading.  
- **Streamlit app** to test new comments interactively.  

---

## 🛠 Installation  

Clone the repository:  
```bash
git clone https://github.com/MYTHILI05896/Deep-Learning-for-Comment-Toxicity-Detection.git
cd Deep-Learning-for-Comment-Toxicity-Detection/toxicity-detection-streamlit


Create a virtual environment (recommended):

python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows

Install dependencies:

pip install -r requirements.txt

Training the Model

To train the model on your dataset:

python train_model.py

Running the Streamlit App

To launch the web app:

streamlit run app/app.py


-m9808262@gmail.com
Mythili N
