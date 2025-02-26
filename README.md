Customer Purchase Prediction using Decision Tree Classifier

This project involves building a Decision Tree Classifier to predict whether a customer will purchase a product or service based on their demographic and behavioral data. The dataset used for this task is the Bank Marketing dataset from the UCI Machine Learning Repository.

The Bank Marketing dataset contains information about a bank's marketing campaigns, and the goal is to predict if a customer will subscribe to a term deposit based on various features such as age, job, marital status, education, and contact information.
Dataset Overview

The dataset consists of the following columns:

    Age: The age of the customer.
    Job: The type of job the customer has (e.g., admin, technician, services, etc.).
    Marital: The marital status of the customer (e.g., married, single, divorced).
    Education: The education level of the customer (e.g., primary, secondary, tertiary).
    Default: Whether the customer has credit in default (yes/no).
    Housing: Whether the customer has a housing loan (yes/no).
    Loan: Whether the customer has a personal loan (yes/no).
    Contact: The communication type used for contacting the customer (e.g., cellular, telephone).
    Month: The last contact month of the year (e.g., January, February, etc.).
    Day_of_week: The last contact day of the week (e.g., Monday, Tuesday).
    Duration: The duration of the last contact in seconds.
    Campaign: The number of contacts performed during this campaign.
    Pdays: The number of days since the customer was last contacted during a previous campaign.
    Previous: The number of contacts performed before this campaign.
    Poutcome: The outcome of the previous marketing campaign (e.g., success, failure).
    Target: Whether the customer subscribed to the term deposit (yes/no).

Objective

The objective of this project is to build a machine learning model (a Decision Tree Classifier) to predict whether a customer will subscribe to a term deposit based on their demographic and behavioral data.
Steps in This Project
1. Data Preprocessing

    Load the dataset: The dataset is loaded into a Pandas DataFrame.
    Handle missing values: Any missing values are handled appropriately (e.g., imputation or removal).
    Feature encoding: Categorical features such as job, marital, education, and contact are encoded into numerical values using techniques like label encoding or one-hot encoding.
    Data normalization: Continuous features like age, duration, and campaign are scaled using standard scaling techniques to ensure uniformity across features.

2. Exploratory Data Analysis (EDA)

    Analyze the distribution of the target variable (Target) to understand the class balance.
    Visualize relationships between features and the target using bar plots, histograms, and correlation heatmaps.
    Identify feature importance using decision trees to determine which variables influence the target variable the most.

3. Building the Decision Tree Classifier

    Split the dataset into training and testing sets (e.g., 80% for training and 20% for testing).
    Use the DecisionTreeClassifier from Scikit-learn to build the model.
    Tune the model using hyperparameters like max_depth, min_samples_split, and min_samples_leaf to optimize performance.
    Visualize the trained decision tree using plot_tree to interpret the model and understand the decision-making process.

4. Model Evaluation

    Evaluate the model using common metrics such as accuracy, precision, recall, and F1-score.
    Plot a confusion matrix to visualize the performance of the classifier.
    Use cross-validation to further validate the model's performance and ensure it generalizes well to unseen data.

Libraries Used

    Pandas: For data manipulation and cleaning.
    NumPy: For numerical operations.
    Matplotlib and Seaborn: For data visualization.
    Scikit-learn: For building the decision tree classifier, data splitting, and model evaluation.
    Graphviz: For visualizing the decision tree.

How to Run
1. Clone the repository:

git clone https://github.com/your-username/customer-purchase-prediction.git
cd customer-purchase-prediction

2. Install the required libraries:

pip install -r requirements.txt

3. Run the Jupyter Notebook or Python Script to train and evaluate the model:

jupyter notebook decision_tree_classifier.ipynb

Or

python decision_tree_classifier.py

4. View the decision tree visualization:

Once the model is trained, a decision tree visualization will be generated, showing the decision-making process used by the classifier to predict whether a customer will subscribe to a term deposit.
Example Output

The Decision Tree model's performance can be evaluated as follows:

    Accuracy: 85% (depending on the hyperparameters and data split)
    Precision: 83%
    Recall: 80%
    F1-Score: 81%

The confusion matrix will show how well the model performed on predicting customers who subscribed to the term deposit and those who did not.
Conclusion

By building a Decision Tree Classifier on the Bank Marketing dataset, we were able to predict customer responses to marketing campaigns. The insights derived from the model can be used to enhance targeted marketing efforts, ensuring better resource allocation and improving conversion rates.

This project provides an introduction to applying machine learning algorithms in real-world marketing scenarios, focusing on data preprocessing, model building, and evaluation techniques.
Additional Notes

    Make sure to download the Bank Marketing dataset from the UCI Machine Learning Repository and place it in the appropriate directory, or update the data path in the script.
    Hyperparameter tuning may improve the model's performance. Feel free to experiment with different parameters.

This README provides a step-by-step overview of the project, outlining the data preparation, model-building process, and evaluation, along with clear instructions for users to run the project locally.
