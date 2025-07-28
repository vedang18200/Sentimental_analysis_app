# 🧠 Sentiment Analysis App

An interactive **Streamlit** web app that detects the **sentiment** (positive or negative) of input text using a machine learning model trained on the [Sentiment140 dataset](https://www.kaggle.com/datasets/kazanova/sentiment140).

![Demo](https://user-images.githubusercontent.com/your-demo-link/demo.gif) <!-- Replace with your own gif or screenshot -->

---

## 🚀 Features

- Real-time sentiment prediction
- Cleaned and preprocessed tweets using regular expressions
- Trained with a subset of Sentiment140 (10,000 samples for fast performance)
- Uses TF-IDF + best-performing ML model
- Confidence score shown for predictions (if supported by model)
- Easy to use, clean UI

---

## 🧠 Model Details

- Dataset: [Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140)
- Preprocessing:
  - Removed URLs, mentions, hashtags, special characters
  - Lowercased and stripped text
- Models Tried:
  - Logistic Regression ✅ *(best)*
  - Naive Bayes
  - Linear SVC
- Vectorization: TF-IDF (max features: 5000)

---

## 📂 Project Structure
sentiment_app/
│
├── models/
│ ├── best_sentiment_model.pkl # Trained model
│ └── tfidf_vectorizer.pkl # Vectorizer
│
├── app.py # Streamlit frontend
├── requirements.txt # Dependencies
└── README.md # Project overview

---

## 🛠️ Installation

### 🔹 Clone the Repo

```bash
git clone https://github.com/vedang18200/sentiment-analysis-app.git
cd sentiment-analysis-app
```
### 🔹 Create a Virtual Environment (optional)
```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```
### 🔹 Install Dependencies
```bash
pip install -r requirements.txt
```
### 🔹 Run the App
```bash
streamlit run app.py
```
Then open http://localhost:8501/ in your browser.

### 🧪 Example
Input: -
```bash
I absolutely love this app. It's fast and fun!
```
Output: -
🙂 Positive Sentiment
Confidence: 95.42%

### 📈 Model Training Notebook
The training notebook includes:
Data sampling
Cleaning
EDA visualizations (word clouds, bar plots, tweet length, etc.)
Model training & evaluation
Model saving with joblib

### 📬 Contact
Made with ❤️ by Vedang Deshmukh
Feel free to connect or contribute!
📧 vedangdeshmukh777@gmail.com




