
# 📰 NewsVerify AI — Fake News Detection

NewsVerify AI is an educational machine-learning application that analyzes news headlines and article text and classifies them as **Real-news-like** or **Fake-news-like** based on patterns learned from a labeled dataset.

## Features

- Machine-learning text classification
- TF-IDF feature extraction with unigrams and bigrams
- Logistic Regression classifier
- Interactive Streamlit dashboard
- Separate scores for real-news-like and fake-news-like classifications
- Evaluation using a held-out test dataset

## Technology Stack

- Python
- Pandas
- Scikit-learn
- Streamlit
- TF-IDF Vectorization
- Logistic Regression

## Dataset

The project uses the ISOT Fake News Dataset, containing `Fake.csv` and `True.csv`.

The dataset is combined and shuffled, then split into training and testing subsets using stratified sampling.

## Model Evaluation

The initial experiment achieved **99.17% accuracy** on a held-out test split containing 7,819 articles.

This result measures performance on this particular dataset and split. It should not be interpreted as evidence that the application can reliably identify misinformation in real-world news. Dataset-specific patterns, source overlap, and distribution differences can affect performance.

## Project Structure

```text
fake-news-detection/
├── data/
│   ├── Fake.csv
│   └── True.csv
├── models/
│   └── fake_news_model.pkl
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation and Usage

### 1. Clone the repository

```bash
git clone https://github.com/Jaiswal0318/fake-news-detection.git
cd fake-news-detection
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Prepare the dataset

Place `Fake.csv` and `True.csv` in the `data/` directory.

### 5. Train the model

```powershell
python train_model.py
```

### 6. Launch the application

```powershell
streamlit run app.py
```

## Limitations and Responsible Use

- Predictions are based on patterns learned from the training dataset.
- A model score is not a verified probability that a claim is true or false.
- The classifier can misclassify satire, opinion, breaking news, and unfamiliar writing styles.
- Important claims should be checked against reliable independent sources.
- This educational project does not replace professional fact-checking.

## Author

**Abhishek Jaiswal**

GitHub: [Jaiswal0318](https://github.com/Jaiswal0318)
