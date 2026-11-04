# AI-Powered Fake Job and Recruitment Scam Detection System

A beginner-friendly prototype that screens pasted job advertisements and recruitment messages using Natural Language Processing (NLP), TF-IDF, and supervised machine learning.

## Project objective

Help job seekers identify recruitment messages that resemble fraudulent postings. The application provides:
- A machine-learning classification estimate
- Separate rule-based warning-sign checks
- Practical safety recommendations

**Important:** This is a screening aid, not a fraud verification service. It can make false-positive and false-negative predictions. Independently verify every employer and job offer.

## Technology stack

- Python
- Pandas and NumPy
- scikit-learn
- TF-IDF
- Logistic Regression and Multinomial Naive Bayes
- Streamlit
- Matplotlib

## Project structure

```text
fake_job_detector/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── data/
│   └── job_posts.csv
├── models/
│   └── (generated after training; not committed)
├── reports/
│   └── (generated after training)
└── screenshots/
    └── (add screenshots of your running app)
```

## Setup on Windows

1. Install Python 3.10 or later.
2. Download or clone this repository.
3. Open Command Prompt in the project folder.
4. Create a virtual environment:

   ```bash
   python -m venv venv
   ```

5. Activate it:

   ```bash
   venv\Scripts\activate
   ```

6. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

7. Train and evaluate the model:

   ```bash
   python train_model.py
   ```

8. Start the web application:

   ```bash
   streamlit run app.py
   ```

The application should open in your browser.

## Dataset

`data/job_posts.csv` currently contains a **small illustrative sample created for demonstrating the code workflow**. It is not a validated real-world dataset and must not be used to claim reliable scam-detection performance.

For a proper academic evaluation, replace or supplement it with a suitable public labeled dataset such as the Employment Scam Aegean Dataset (EMSCAD), only after checking its current access conditions, license, documentation, and column definitions. Prepare a CSV with:
- `text`: the job advertisement or recruitment message text
- `label`: `fraudulent` or `legitimate`

Do not upload private messages, phone numbers, email addresses, or personal identity information.

## Model workflow

1. Load and inspect labeled text data.
2. Split data into training and test sets.
3. Fit TF-IDF only within the training pipeline.
4. Train Logistic Regression and Multinomial Naive Bayes.
5. Compare precision, recall, F1-score, accuracy, and confusion matrix.
6. Save the model with the best fraudulent-class F1-score.
7. Load the trained model in Streamlit.

## Warning-sign checks

The app separately checks for selected phrases relating to upfront payments, unrealistic earnings, urgency, hiring without interviews, and requests for sensitive information. These checks are not the machine-learning model's explanation and do not prove fraud.

## Limitations and responsible use

- Training data may be outdated, biased, or unrepresentative.
- Scam tactics and wording change over time.
- False positives and false negatives are possible.
- A professional-looking message is not proof of legitimacy.
- The app does not verify company identity or live job listings.
- Do not treat its output as a definitive fraud determination.

## Future enhancements

- Evaluate on a larger, licensed, representative dataset.
- Add multilingual recruitment-message support.
- Add carefully sourced company and job-link verification.
- Improve model monitoring and periodic evaluation.

## References

- scikit-learn text feature extraction: https://scikit-learn.org/stable/modules/feature_extraction.html
- scikit-learn text classification tutorial: https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
- scikit-learn model evaluation: https://scikit-learn.org/stable/modules/model_evaluation.html
- Streamlit documentation: https://docs.streamlit.io/
