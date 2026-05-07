# 🎥 YouTube Popularity Predictor

> Predict video view counts and popularity categories using machine learning — Gradient Boosting regression + KMeans clustering trained on real YouTube data.

---

## What It Does

Enter a YouTube video's details (title, description, duration, upload hour, channel, tags) and the app predicts:

- **Estimated view count** — regression output converted from log scale
- **Popularity category** — one of Low / Medium / High / Viral, assigned via KMeans clustering

---

## Pages

| Tab | Description |
|-----|-------------|
| 🔮 Predict | Enter video details and get an instant prediction |
| 📊 Model Performance | R², RMSE, feature importances, actual vs predicted chart, PCA cluster plot |
| 🗂️ Data Explorer | Views by upload hour, cluster distribution, dataset preview |

---

## How It Works

### Features Used
- `title_length` — character count of the title
- `title_sentiment` — VADER compound sentiment score of the title
- `has_description` — binary flag (1 = has description)
- `desc_sentiment` — VADER compound sentiment score of the description
- `tag_count` — number of tags
- `duration_sec` — video duration in seconds (parsed from ISO 8601)
- `hour` — hour of upload (0–23)
- `channel_*` — one-hot encoded channel name (15 channels)

### Models
- **Regression**: `GradientBoostingRegressor` with `StandardScaler` pipeline — predicts `log(viewCount + 1)`
- **Clustering**: `KMeans` (k=4) on the 7 base features + predicted log view count — assigns Low / Medium / High / Viral labels sorted by cluster centroids
- **Sentiment**: NLTK VADER on title and description text

---

## Project Structure

```
youtube_predictor/
├── app.py                  ← Streamlit app
├── requirements.txt        ← Python dependencies
├── .streamlit/
│   └── config.toml         ← Dark theme config
└── data/
    └── merged_file.csv     ← YouTube dataset (you provide this)
```

---

## Running Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Place your dataset
# → data/merged_file.csv

# Run
streamlit run app.py
```

The app trains the models fresh on startup and caches them — first load takes ~30 seconds depending on dataset size.

---

## Deploying to Streamlit Cloud (Free)

1. Push this folder to a GitHub repository — **include `data/merged_file.csv`**
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → connect your repo
4. Set main file to `app.py`
5. Click **Deploy**

> The dataset must be committed to GitHub for the model to train on the cloud.

---

## Dataset Format

`merged_file.csv` should contain these columns:

| Column | Type | Description |
|--------|------|-------------|
| `title` | string | Video title |
| `description` | string | Video description |
| `viewCount` | integer | Total views |
| `likeCount` | integer | Total likes |
| `commentCount` | integer | Total comments |
| `duration` | string | ISO 8601 duration (e.g. `PT5M30S`) |
| `publishedAt` | datetime | Upload timestamp |
| `channelTitle` | string | Channel name |
| `tags` | list/string | Video tags |

---

## Tech Stack

- **Streamlit** — UI framework
- **scikit-learn** — GradientBoosting, KMeans, PCA, StandardScaler
- **NLTK VADER** — Sentiment analysis
- **isodate** — ISO 8601 duration parsing
- **Plotly** — Interactive charts
- **Pandas / NumPy** — Data processing

---

## Author

**Madhu Priya Pulletikurthi**  
© 2025 · Built with Streamlit
