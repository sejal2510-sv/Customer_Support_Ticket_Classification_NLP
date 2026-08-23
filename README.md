# Customer Support Ticket Classification using NLP

An NLP-based machine learning project for automatically classifying customer support tickets into the appropriate support department using text classification techniques.

## Project Overview

Customer support teams receive a large number of tickets that need to be manually reviewed and routed to the appropriate department. This project uses **Natural Language Processing (NLP)** and **Machine Learning** to automatically classify support tickets based on their textual content.

The model uses the **ticket subject and body** as input and predicts the corresponding **support queue/department**.

## Dataset

The dataset contains **20,000 customer support ticket records** with 15 columns. The data includes multilingual customer queries, mainly English and German, along with ticket metadata.

### Important Columns

| Column            | Description                                                  |
| ----------------- | ------------------------------------------------------------ |
| `subject`         | Subject/title of the support ticket                          |
| `body`            | Main text of the customer request                            |
| `answer`          | Corresponding support response                               |
| `type`            | Type of ticket such as Request, Incident, Problem, or Change |
| `queue`           | Target support department                                    |
| `priority`        | Ticket priority                                              |
| `language`        | Ticket language                                              |
| `tag_1` – `tag_8` | Ticket-related tags                                          |

The `queue` column is used as the **target variable**, while the `subject` and `body` columns are combined to create the input text.

## Objective

The main objectives of this project are:

* Clean and preprocess customer support text data.
* Combine ticket subject and body into a single text feature.
* Convert text into numerical features using **TF-IDF**.
* Train multiple machine learning classification models.
* Compare model performance using accuracy and classification metrics.
* Identify the best-performing model for automatic ticket routing.

## Methodology

### 1. Data Loading

The dataset is loaded using Pandas and initially contains **20,000 rows and 15 columns**.

### 2. Data Preprocessing

The following preprocessing steps are performed:

* Remove records with missing ticket bodies.
* Fill missing subjects with empty strings.
* Combine `subject` and `body` into a single text column.
* Convert text to lowercase.
* Remove URLs and HTML tags.
* Remove punctuation and numbers.
* Remove unnecessary newline characters.

After handling missing values, **19,998 records** remain for modeling.

### 3. Train-Test Split

The data is divided into:

* **80% Training Data**
* **20% Testing Data**
* `random_state = 42`
* Stratified splitting is used to preserve class distribution.

This results in **15,998 training records and 4,000 testing records**.

### 4. TF-IDF Feature Extraction

The cleaned ticket text is transformed into numerical features using **TF-IDF (Term Frequency-Inverse Document Frequency)**.

The implementation uses:

* Maximum features: **5,000**
* Unigrams and bigrams: `ngram_range=(1,2)`

This allows the models to capture both individual words and two-word phrases.

### 5. Machine Learning Models

Three classification algorithms are trained and compared:

1. **Logistic Regression**
2. **Multinomial Naive Bayes**
3. **Linear Support Vector Machine (Linear SVM)**

## Model Performance

| Model                   |   Accuracy |
| ----------------------- | ---------: |
| Logistic Regression     |     42.10% |
| Multinomial Naive Bayes |     37.92% |
| **Linear SVM**          | **43.15%** |

Among the tested models, **Linear SVM achieved the highest accuracy of 43.15%** and was selected as the best-performing model.

## Evaluation

Model performance is evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

A confusion matrix is also generated for the best-performing Linear SVM model to analyze classification performance across different support queues.

## Visualizations

The project generates the following visualizations:

### Ticket Distribution

Shows the distribution of support tickets across different departments/queues.

### Model Comparison

Compares the accuracy of Logistic Regression, Multinomial Naive Bayes, and Linear SVM.

### Confusion Matrix

Displays the classification performance of the best-performing model across support departments.

## Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Matplotlib**
* **Seaborn**
* **Regular Expressions**
* **TF-IDF**
* **Logistic Regression**
* **Multinomial Naive Bayes**
* **Linear SVM**

## Project Workflow

```text
Customer Support Ticket
          ↓
 Subject + Body
          ↓
    Text Cleaning
          ↓
  Train-Test Split
          ↓
    TF-IDF Vectorization
          ↓
 ┌────────┼──────────────┐
 ↓        ↓              ↓
Logistic  Naive Bayes   Linear SVM
Regression
 └────────┼──────────────┘
          ↓
   Model Evaluation
          ↓
   Best Model: Linear SVM
          ↓
 Automatic Ticket Classification
```

## Project Structure

```text
customer-support-ticket-classification-nlp/
│
├── dataset-tickets-multi-lang-4-20k.csv
├── customer_support_ticket_classification.ipynb
├── queue_distribution.png
├── model_comparison.png
├── confusion_matrix.png
└── README.md
```

## How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/customer-support-ticket-classification-nlp.git
cd customer-support-ticket-classification-nlp
```

### 2. Install Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 3. Run the Notebook

Open the notebook in **Google Colab or Jupyter Notebook** and execute the cells sequentially.

Make sure the CSV dataset is placed in the same directory as the notebook.

## Key Takeaways

* NLP can be used to automate customer support ticket routing.
* TF-IDF provides an effective representation of textual support tickets.
* Multiple traditional machine learning algorithms can be compared for text classification.
* In this implementation, **Linear SVM performed better than Logistic Regression and Multinomial Naive Bayes**, achieving **43.15% accuracy**.

## Future Improvements

The project can be further improved by:

* Applying class balancing techniques for underrepresented queues.
* Using advanced text preprocessing and stopword removal.
* Testing different TF-IDF parameters.
* Using word embeddings such as Word2Vec or GloVe.
* Exploring transformer-based models such as BERT.
* Performing hyperparameter tuning using GridSearchCV or RandomizedSearchCV.
* Building a simple web interface for real-time ticket classification.

## Author

**Sejal Verma**

M.Sc. Statistics

---

⭐ If you find this project useful, consider giving the repository a star!
