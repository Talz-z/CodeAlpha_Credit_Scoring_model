import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


np.random.seed() 
n_samples = 100  


age = np.random.randint(18, 70, n_samples)
income = np.random.randint(20000, 150000, n_samples)
debt = np.random.randint(0, 100000, n_samples)
history = np.random.uniform(0.4, 0.99, n_samples) 


debt_to_income = debt / (income + 1e-6)  
risk = 1 - history



base_rule = (debt_to_income < 0.4) & (history > 0.7)

noise = np.random.random(n_samples) < 0.15
creditworthy = (base_rule ^ noise).astype(int)


data = {
    'Age': age,
    'Income': income,
    'Debt': debt,
    'history': history,
    'debtToIncome': debt_to_income,
    'risk': risk,
    'Creditworthy': creditworthy
}
df = pd.DataFrame(data)

print("Generated Credit Data (First 10 Rows):")
print(df.head(10))
print(f"\nDataset shape: {df.shape}")
print(f"Creditworthy ratio: {df['Creditworthy'].mean():.2f}")


X = df.drop('Creditworthy', axis=1)
y = df['Creditworthy']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)


predictions = model.predict(X_test)

print("\nEvaluation Metrics:")
print(f"Accuracy: {accuracy_score(y_test, predictions):.2f}")
print(f"Precision: {precision_score(y_test, predictions, zero_division=0):.2f}")
print(f"Recall: {recall_score(y_test, predictions):.2f}")
print(f"F1-Score: {f1_score(y_test, predictions):.2f}")

new_age = np.random.randint(18, 65)
new_income = np.random.randint(25000, 120000)
new_debt = np.random.randint(0, 80000)
new_history = np.random.uniform(0.5, 0.95)
new_debt_to_income = new_debt / new_income
new_risk = 1 - new_history

new_applicant = pd.DataFrame([[new_age, new_income, new_debt, new_history, 
                              new_debt_to_income, new_risk]],
                            columns=X.columns)

prediction = model.predict(new_applicant)
prob_approved = model.predict_proba(new_applicant)[0][1]

print("\nNew Applicant Details:")
print(f"Age: {new_age}, Income: ${new_income:,}, Debt: ${new_debt:,}")
print(f"Credit History: {new_history:.2f}, Debt-to-Income: {new_debt_to_income:.2f}")

print("\nCredit Decision:")
print(f"Decision: {'APPROVED' if prediction[0] == 1 else 'REJECTED'}")
print(f"Approval Probability: {prob_approved:.1%}")