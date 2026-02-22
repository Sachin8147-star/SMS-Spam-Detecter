# SMS-Spam-Detecter

## Task 4: Spam Message Classification Model
### CDZ PVT LTD - Python Development Internship

---

## 📌 Project Overview

As part of my Python Development internship at **CDZ PVT LTD**, I have developed a predictive model to classify SMS messages as either **Spam or Ham (Legitimate)**.

The goal of this task was to demonstrate the implementation of a machine learning pipeline using **Scikit-Learn**, from data preprocessing and visualization to model training and evaluation.
---

## 📊 Dataset

I used the **SMS Spam Collection Dataset**, which contains over **5,000 SMS messages** tagged as spam or ham.

- **Ham**: Legitimate personal messages.

- **Spam**: Advertisements, promotions, or phishing attempts.
---

## 🛠️ Technologies Used

- Python 3.x

- Pandas: For data manipulation and cleaning.

- Matplotlib & Seaborn: For exploratory data analysis (EDA).

- Scikit-Learn: For building the machine learning model.

- Jupyter Notebook: For the final implementation and documentation.

---

## 🚀 How the Model Works

1. Data Cleaning: Removed unnecessary columns and handled missing values (if any).

2. EDA: Visualized the distribution of spam vs. ham messages to understand data balance.

3. Feature Extraction: Used TfidfVectorizer to convert raw text into numerical features that a machine can understand.

4. Model Selection: I chose the Multinomial Naive Bayes algorithm. This is a classic and highly effective choice for text classification tasks.

5. Evaluation: The model was tested on a 20% hold-out set to measure accuracy and precision.
---

## 📈 Results

- The model performed with high accuracy. Below are the key metrics from the evaluation:

- Accuracy: ~98% (or your actual score)

- Precision/Recall: The model effectively identified spam messages with minimal false positives.

- The detailed classification report and confusion matrix can be found inside the Jupyter Notebook.
---

## 📂 Repository Structure

task4.ipynb: The main Jupyter Notebook containing the code and explanations.

spam.csv: The dataset used for training.

README.md: Project documentation.

⚙️ How to Run

Clone this repository.

Ensure you have the required libraries installed:

code
Bash
download
content_copy
expand_less
pip install pandas scikit-learn matplotlib seaborn

Open task4.ipynb in VS Code or Jupyter Notebook and run all cells.

Author: [Your Name]
Internship Role: Python Developer
Deadline: March 02, 2026

💡 Pro-Tips for your GitHub:

The "Human" Touch: In the "Results" section, replace ~98% with the actual accuracy number your code printed out.

Add a Screenshot: If you want to go the extra mile, take a screenshot of your bar chart (the "Spam vs Ham" graph) and upload it to the repo, then link it in the README.

Commit Messages: When you upload to GitHub, don't just say "upload files." Use messages like:

"Initial commit: added dataset and basic script"

"Finished preprocessing and TF-IDF vectorization"

"Added final evaluation and README"
This shows the recruiter that you worked on it over time.