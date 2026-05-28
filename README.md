# SMS Spam Classifier

An elegant, high-performance machine learning web application powered by **Streamlit** that processes and classifies SMS or text messages as **Spam** or **Ham (Not Spam)** in real time. It features a advanced natural language processing (NLP) pipeline and a robust ensemble model.

---

## Key Features

* **Interactive Streamlit Interface**: Clean, minimalist UI with custom CSS styling and responsive layout.
* **Robust Text Preprocessing**: Deep text-cleaning pipeline powered by **NLTK** (lowercasing, tokenization, stopword removal, punctuation stripping, and stemming).
* **High-Accuracy Ensemble Classifier**: Powered by a soft-voting ensemble model combining Support Vector Machines (SVM), Naive Bayes, and Extra Trees to maximize precision and recall.
* **Real-time Inference**: Instant classification feedback.

---

## Natural Language Processing (NLP) Pipeline

Before any classification occurs, input text passes through a rigorous multi-stage preprocessing pipeline to remove noise and isolate the core signals:

```
[ Input Raw Text ]
       │
       ▼
 1. Convert to Lowercase   ──> Standardizes casing (e.g., "SPAM" -> "spam")
       │
       ▼
 2. Word Tokenization      ──> Splits the text into individual words/tokens
       │
       ▼
 3. Alphanumeric Filter    ──> Removes emojis, special characters, and noise
       │
       ▼
 4. Stopwords & Punctuation──> Removes common filler words (e.g., "the", "is")
       │
       ▼
 5. Porter Stemming        ──> Reduces words to root form (e.g., "running" -> "run")
       │
       ▼
 6. TF-IDF Vectorization   ──> Converts text into a 3000-dimensional frequency vector
       │
       ▼
[ Ensemble Classifier ]    ──> VotingClassifier predicts Spam (1) vs Ham (0)
```

---

## Model & Architecture Details

The system employs a multi-model architecture built using **scikit-learn**:

### 1. Feature Extractor
* **Type**: `TfidfVectorizer` (Term Frequency-Inverse Document Frequency)
* **Configuration**: Curated vocabulary limited to the top **3,000 features** (`max_features=3000`) based on word frequencies in the corpus, ensuring a compact yet descriptive semantic space.

### 2. Classifier
* **Type**: `VotingClassifier` (Ensemble Model)
* **Voting Mode**: **Soft Voting** (`voting='soft'`), which sums the predicted probabilities of each estimator and classifies based on the highest cumulative confidence.
* **Sub-Estimators**:
  1. **Support Vector Classifier (SVC)**: Configured with `kernel='sigmoid'`, `gamma=1.0`, and probability calibration enabled. Excel at finding high-dimensional decision boundaries.
  2. **Multinomial Naive Bayes (MultinomialNB)**: Extremely efficient and baseline-strong probabilistic classifier tailored for text classification tasks.
  3. **Extra Trees Classifier (ExtraTreesClassifier)**: An ensemble of 500 extremely randomized decision trees (`n_estimators=500`) to capture complex, non-linear relationships and reduce overfitting.

---

## 💻 Installation & Setup

Get the project up and running locally in just a few minutes:

### 1. Clone the Repository
```bash
git clone https://github.com/Parth-Verma0812/sms-spam-classifier.git
cd sms-spam-classifier
```

### 2. Set Up Virtual Environment (Recommended)
Create and activate a virtual environment to isolate the project dependencies:

**On macOS & Linux:**
```bash
python3 -m venv .venv
source .venv/bin/bin/activate
```

**On Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies
Install all required libraries using the provided `requirements.txt` file:
```bash
pip install -r requirements.txt
```

### 4. Download NLTK Datasets
Since the pipeline utilizes tokenizers and stopword lists from the Natural Language Toolkit, download the required packages. You can run this in your terminal or python interpreter:
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

---

## 🎮 Running the Application

Launch the Streamlit web application on your local machine:

```bash
streamlit run app.py
```

Once running, the application will be hosted at:
* **Local URL**: `http://localhost:8501`
* **Network URL**: `http://192.168.x.x:8501`

Simply type or paste a message into the input field and click **Classify this Message** to view the classification result instantly!

---

## 📁 File Structure

```
sms-spam-classifier/
├── .venv/                 # Local python virtual environment (excluded from git)
├── Classifier.pkl         # Trained VotingClassifier ensemble model
├── Vectorizer_.pkl        # Pickled TF-IDF Vectorizer (3,000 features)
├── Vectorizer.pkl         # Backup/alternative Vectorizer
├── app.py                 # Streamlit web application logic
├── requirements.txt       # Pin-pointed Python library dependencies
└── README.md              # Project documentation (this file)
```

---

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to open a pull request or submit an issue to improve the preprocessing pipeline or model architecture.

*https://github.com/Parth-Verma0812*
