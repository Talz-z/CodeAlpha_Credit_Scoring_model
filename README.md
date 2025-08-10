# CodeAlpha_Credit_Scoring_model

## Key Features:
- Synthetic dataset generation with realistic financial features
- Creditworthiness determination using debt-to-income ratio and payment history
- Random Forest classification model training
- Comprehensive model evaluation metrics (accuracy, precision, recall, F1)
- New applicant prediction with approval probability
- Noise injection to simulate real-world decision inconsistencies

## How It Works:
1. Generates synthetic financial data for 100 applicants:
   - Age (18-70), Income ($20k-$150k), Debt ($0-$100k)
   - Payment history (40%-99% on-time payments)
   - Calculates debt-to-income ratio and risk score
2. Determines creditworthiness using business rules:
   - Base rule: DTI < 40% and payment history > 70%
   - Adds 15% random noise to simulate real-world inconsistencies
3. Trains Random Forest classifier on 80% of data
4. Evaluates model performance on 20% test set
5. Simulates new applicant assessment:
   - Predicts approval/rejection status
   - Calculates approval probability percentage

## Code Structure:
- Uses pandas/numpy for data manipulation
- Employs scikit-learn for ML operations:
  - 'RandomForestClassifier' for prediction model
  - 'train_test_split' for data partitioning
  - Metrics (accuracy_score, precision_score, etc.) for evaluation
- Maintains state with:
  - 'df' DataFrame holding all applicant data
  - 'X'/'y' feature/target variables for modeling
  - 'model' storing trained classifier
- Key variables:
  - 'debt_to_income': Debt/(Income + epsilon)
  - 'risk': 1 - payment_history
  - 'creditworthy': Binary target (1 = approved, 0 = rejected)
- Workflow stages:
  1. Data generation and preprocessing
  2. Feature engineering
  3. Model training and evaluation
  4. New applicant simulation
