# swe485
group members:
Reem Altamimi
Wedad Alqahtani
Noura Alshahrani
Lina Alsuhaibani
Maram Alamri

-----------------
Project Motivation:

Stroke is among the major causes of deaths and long-term disabilities globally. Early risk prediction of stroke would greatly improve the outcomes in patients by allowing timely medical interventions. The motivation of this project is to use machine learning techniques in analyzing health-related factors and to predict the occurrence of a stroke. By working with this dataset, we try to develop a predictive model to assist healthcare professionals in identifying individuals at high risk.

The key goal of this dataset is to present stroke prediction abilities using health parameters. It offers machine learning models the ability to predict the   likelihood of a stroke based on input features like age, hypertension, heart disease, and smoker status.

Objective of Collecting This Dataset:
This dataset has been gathered to support the research process involved in predictive health analytics. 
Primary objectives include the following:
1. Medical Decision Support: Helping doctors and healthcare professionals make data-driven decisions regarding stroke prevention.
2. Risk Factor Analysis: Understanding how different health parameters contribute to stroke occurrence.
3. Machine Learning Applications: Training and evaluating classification models for stroke prediction.

The Stroke Prediction Dataset, which is accessible on Kaggle, consists of 5,110 records with 12 attributes pertaining to patient demographics and medical data. Based on these input values, the dataset is intended to forecast the probability that a patient will have a stroke.

Attribute Information:
1. id: Unique identifier for each patient.
2. gender: Categorical variable indicating the patient's gender: "Male", "Female", or "Other".
3. age: Numerical value representing the patient's age.
4. hypertension: Binary variable where 0 indicates the patient does not have hypertension, and 1 indicates the patient has hypertension.
5. heart_disease: Binary variable where 0 indicates the patient does not have any heart diseases, and 1 indicates the patient has a heart disease.
6. ever_married: Categorical variable indicating if the patient has ever been married: "No" or "Yes".
7. work_type: Categorical variable describing the patient's type of work: "children", "Govt_job", "Never_worked", "Private", or "Self-employed".
8. Residence_type: Categorical variable indicating the patient's place of residence: "Rural" or "Urban".
9. avg_glucose_level: Numerical value representing the patient's average glucose level in the blood.
10. bmi: Numerical value representing the patient's body mass index.
11. smoking_status: Categorical variable indicating the patient's smoking status: "formerly smoked", "never smoked", "smokes", or "Unknown" (where "Unknown" means that the information is unavailable for this patient).
12. stroke: Binary target variable where 1 indicates the patient had a stroke, and 0 indicates the patient did not have a stroke.
This dataset is valuable for developing machine learning models aimed at predicting stroke occurrences based on various health and demographic factors. It has been utilized in studies focusing on predictive modeling and the identification of key risk factors for stroke.

dataset source link: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset/data



Process technique explanation:

1)	Handling missing value in bmi column
   
df = df.copy()  
df.loc[:, 'bmi'] = df['bmi'].fillna(df['bmi'].median())

the goal:
in bmi column there is missing value (NaN) and these missing values will affect the analysis results
We will replace the missing values with median instead of mean because it is more stable when there are extreme values

3)	Converting Categorical Variables to Numbers Using One-Hot Encoding
   
categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

Some columns contain categorical values (Categorical Variables), such as:
•	gender (male/female)
•	ever_married (married/not married)
•	work_type (type of job)
•	Residence_type (type of residence)
•	smoking_status (smoking status)

These values are converted into numerical values using One-Hot Encoding, which creates additional columns where each option becomes a separate column with binary values (0 or 1).
Why drop_first=True?
To avoid the problem of Multicollinearity, the first category of each column is dropped.

3)	Applying Numerical Variable Normalization

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
numeric_cols = ['age', 'avg_glucose_level', 'bmi']
df_encoded[numeric_cols] = scaler.fit_transform(df_encoded[numeric_cols])

•	Normalize numerical values (age, avg_glucose_level, bmi) to a range between 0 and 1 using MinMaxScaler.

Why MinMaxScaler?
•	It ensures that all variables have the same range, which helps models learn efficiently, especially when using neural networks and algorithms sensitive to different value scales.

4)	Saving Processed Data
   
processed_file_path = "/content/drive/MyDrive/Dataset/processed_stroke_data.csv"
df_encoded.to_csv(processed_file_path, index=False)
print("Processed dataset saved at:", processed_file_path)

Save the processed data in a new CSV file to be used later for analysis or training a machine learning model.

