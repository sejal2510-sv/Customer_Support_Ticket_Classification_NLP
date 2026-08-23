# Multilingual Customer Support Ticket Classification using NLP

An NLP-based machine learning project for automatically classifying customer support tickets into their correct **ticket type** using text classification techniques, enhanced with additional metadata signals.

## Project Overview

Customer support teams receive a large number of tickets that need to be manually reviewed and categorized before they can be handled. This project uses **Natural Language Processing (NLP)** and **Machine Learning** to automatically classify support tickets based on their textual content combined with ticket metadata.

The model uses the **ticket subject and body**, along with **queue, priority, and language** metadata, as input and predicts the corresponding **ticket type** (`Incident`, `Request`, `Problem`, or `Change`).

## Dataset

The dataset contains **20,000 customer support ticket records** with 15 columns. The data includes multilingual customer queries — English and German — along with ticket metadata.

### Important Columns

| Column            | Description                                                  |
| ----------------- | ------------------------------------------------------------ |
| `subject`         | Subject/title of the support ticket                          |
| `body`            | Main text of the customer request                            |
| `answer`          | Corresponding support response (not used for modeling)       |
| `type`            | Type of ticket — Incident, Request, Problem, or Change       |
| `queue`           | Support department the ticket was routed to                  |
| `priority`        | Ticket priority (low, medium, high)                          |
| `language`        | Ticket language (en, de)                                     |
| `tag_1` – `tag_8` | Ticket-related tags                                          |

The `type` column is used as the **target variable**, while `subject` and `body` are combined to create the main text feature, and `queue`, `priority`, `language` are used as additional metadata features.

## Objective

The main objectives of this project are:

* Clean and preprocess multilingual customer support text data.
* Combine ticket subject and body into a single text feature.
* Convert text into numerical features using **TF-IDF**.
* Incorporate ticket metadata (queue, priority, language) as extra signal.
* Train multiple machine learning classification models.
* Compare model performance using accuracy and classification metrics.
* Identify the best-performing model for automatic ticket-type classification.

## Methodology

### 1. Data Loading

The dataset is loaded using Pandas and initially contains **20,000 rows and 15 columns**.

### 2. Data Preprocessing

The following preprocessing steps are performed:

* Remove records with missing ticket bodies.
* Fill missing `subject`, `queue`, `priority`, and `language` values.
* Combine `subject` and `body` into a single text column.
* Convert text to lowercase.
* Remove URLs and HTML tags.
* Remove punctuation and numbers.
* Remove unnecessary newline characters.
* Remove **English and German stopwords** (using the `stop-words` library, since the dataset is bilingual).

After handling missing values, **19,998 records** remain for modeling.

### 3. Train-Test Split

The data is divided into:

* **80% Training Data**
* **20% Testing Data**
* `random_state = 42`
* Stratified splitting is used to preserve class distribution across ticket types.

This results in **15,998 training records and 4,000 testing records**.

### 4. TF-IDF Feature Extraction + Metadata Encoding

The cleaned ticket text is transformed into numerical features using **TF-IDF (Term Frequency-Inverse Document Frequency)**.

The implementation uses:

* Maximum features: **30,000**
* Unigrams and bigrams: `ngram_range=(1,2)`
* `min_df=2`, `max_df=0.9`, `sublinear_tf=True`

In addition, `queue`, `priority`, and `language` are encoded using **One-Hot Encoding** and combined with the TF-IDF matrix (`hstack`) to give a final feature space of **30,015 features**, allowing the models to use both textual and categorical signals.

### 5. Machine Learning Models

Three classification algorithms are trained and compared:

1. **Logistic Regression**
2. **Complement Naive Bayes**
3. **Linear Support Vector Machine (Linear SVM)**

## Model Performance

| Model                    |   Accuracy |
| ------------------------ | ---------: |
| Logistic Regression      |     81.58% |
| Complement Naive Bayes   |     76.95% |
| **Linear SVM**           | **82.20%** |

Among the tested models, **Linear SVM achieved the highest accuracy of 82.20%** and was selected as the best-performing model.

## Evaluation

Model performance is evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

A confusion matrix is also generated for the best-performing Linear SVM model to analyze classification performance across the four ticket types.

## Visualizations

The project generates the following visualizations:

### Ticket Type Distribution

Shows the distribution of support tickets across the four ticket types (Incident, Request, Problem, Change).

### Model Comparison

Compares the accuracy of Logistic Regression, Complement Naive Bayes, and Linear SVM.

### Confusion Matrix

Displays the classification performance of the best-performing model across ticket types.

## Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Matplotlib**
* **Seaborn**
* **SciPy** (sparse matrix stacking)
* **Regular Expressions**
* **stop-words** (multilingual stopword removal)
* **TF-IDF**
* **Logistic Regression**
* **Complement Naive Bayes**
* **Linear SVM**

## Project Workflow

```text
Customer Support Ticket
          ↓
 Subject + Body
          ↓
    Text Cleaning
   (EN + DE stopwords)
          ↓
  Train-Test Split
          ↓
 TF-IDF Vectorization + Metadata Encoding
   (queue, priority, language)
          ↓
 ┌────────┼──────────────┐
 ↓        ↓              ↓
Logistic  Complement    Linear SVM
Regression Naive Bayes
 └────────┼──────────────┘
          ↓
   Model Evaluation
          ↓
   Best Model: Linear SVM
          ↓
 Automatic Ticket Type Classification
```

## Project Structure

```text
multilingual-support-ticket-classification-nlp/
│
├── Dataset_tickets_multilang.csv
├── nlp_updated.ipynb
├── type_distribution.png
├── model_comparison.png
├── confusion_matrix.png
└── README.md
```

## How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/multilingual-support-ticket-classification-nlp.git
cd multilingual-support-ticket-classification-nlp
```

### 2. Install Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy stop-words
```

### 3. Run the Notebook

Open the notebook in **Google Colab or Jupyter Notebook** and execute the cells sequentially.

Make sure the CSV dataset is placed in the same directory as the notebook (or update `file_path` in the loading cell).

## Key Takeaways

* NLP can be used to automate customer support ticket-type classification.
* TF-IDF combined with ticket metadata (queue, priority, language) improves model performance over text alone.
* Multiple traditional machine learning algorithms can be compared for text classification.
* In this implementation, **Linear SVM performed better than Logistic Regression and Complement Naive Bayes**, achieving **82.20% accuracy**.
* The `Problem` class was the hardest to classify (lower recall), likely due to textual overlap with `Incident` tickets.

## Future Improvements

The project can be further improved by:

* Applying class balancing techniques for underrepresented ticket types (e.g. `Change`).
* Testing different TF-IDF parameters and n-gram ranges.
* Using word embeddings such as Word2Vec or GloVe.
* Exploring transformer-based multilingual models such as mBERT or XLM-R.
* Performing hyperparameter tuning using GridSearchCV or RandomizedSearchCV.
* Extending support to additional languages beyond English and German.
* Building a simple web interface for real-time ticket classification.

## Author

**Sejal Verma**

**M.Sc Statistics**
---

⭐ If you find this project useful, consider giving the repository a star!
