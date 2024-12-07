import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

data = {
    'Salary': [50000, 60000, 45000, 70000, 80000],
    'Credit_Score': [650, 700, 600, 720, 750],
    'Loan_Amount': [150000, 200000, 120000, 250000, 300000],
    'Age': [25, 30, 22, 35, 40],
    'Employment_Status': [1, 1, 0, 1, 1],  # 1: Employed, 0: Unemployed
    'Loan_Approval': [1, 1, 0, 1, 1]  # 1: Approved, 0: Denied
}

df = pd.DataFrame(data)

X = df.drop('Loan_Approval', axis=1)
y = df['Loan_Approval']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))

def predict_loan_approval(salary, credit_score, loan_amount, age, employment_status):
    input_data = pd.DataFrame([[salary, credit_score, loan_amount, age, employment_status]], columns=X.columns)
    
    input_data_scaled = scaler.transform(input_data)
    
    prediction = model.predict(input_data_scaled)
    
    if prediction == 1:
        return "Loan Approved"
    else:
        return "Loan Denied"

salary = float(input("Enter salary: "))
credit_score = int(input("Enter credit score: "))
loan_amount = float(input("Enter loan amount: "))
age = int(input("Enter age: "))
employment_status = int(input("Enter employment status (1 for employed, 0 for unemployed): "))

result = predict_loan_approval(salary, credit_score, loan_amount, age, employment_status)
print(result)

