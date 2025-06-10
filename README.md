# 📺 YouTube Popularity Classification and Prediction

This project predicts the popularity of YouTube videos based on their metadata using regression and unsupervised clustering. It also provides a Gradio interface for creators to input video details and receive predicted view counts and a popularity label (Low/ Medium/ High/).

---

## 🔍 Problem Statement

Content creators often publish videos without knowing their potential performance. This project uses machine learning and sentiment analysis to predict the expected view count and classify videos into popularity tiers before they're uploaded.

---

## 🧠 Key Features

- 🔗 **YouTube API**: Fetch video metadata (title, description, stats)
- 💬 **Sentiment Analysis**: Analyze sentiment of titles/descriptions using VADER (NLTK)
- 📊 **Regression Models**: Random Forest, XGBoost, Gradient Boosting
- 📈 **Clustering**: KMeans, HDBSCAN, Agglomerative Clustering
- 🎛️ **Interactive Gradio UI**: Input features → see predictions in real-time

---

## 🛠️ Tech Stack

- **Languages & Libraries**: Python, Pandas, NumPy, Scikit-learn, NLTK, Matplotlib, Seaborn
- **ML Models**: Random Forest, Gradient Boosting, XGBoost (if used)
- **Clustering**: KMeans, HDBSCAN, AgglomerativeClustering
- **Sentiment Analysis**: NLTK VADER
- **Interface**: Gradio
- **API Integration**: Google API Client (YouTube Data API v3)

---

## 🔑 Setup Instructions

### 1. 📦 Install Requirements

```bash
pip install -r requirements.txt
```
### 2. 🔑 Set Up YouTube Data API
- Go to Google Cloud Console
- Create a new project and enable YouTube Data API v3
- Create credentials → API Key
- Save the key in a file (e.g., api_key.txt) or load it via environment variable

### 3. 🔑 Run the Application
 - open notebooks folder
 - run the notebook (`Data_Collection.ipynb`)
 - download the merged.csv file
 - run the notebook (`Regression+Clustering+Gradio.ipynb`)
 - click on the working link in the final output to see the Gradio interface.

## Project Structure

```
├── data/ # Raw and merged data
├── notebooks/ # EDA, modeling and final Gradio Script
├── requirements.txt
├── README.md
```

## 🧠 Future Enhancements

🎥 Include thumbnail image features using computer vision

⏳ Add time-based trend modeling

💬 Incorporate title rewriting recommendations via GPT-style prompts

## 👨‍💻 Author
Madhu Priya P

Graduate Student at UMBC

MPS Data Science

