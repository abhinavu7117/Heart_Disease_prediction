import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def main():
    # 1. Load Data
    print("Loading data from 'heart.csv'...")
    df = pd.read_csv('heart.csv')
    
    X = df.drop('target', axis=1)
    y = df['target']
    
    # 2. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 3. Scale Data (Important for Logistic Regression and Neural Networks)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. Define Models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Feedforward Neural Network': MLPClassifier(random_state=42, max_iter=1500)
    }
    
    # 5. Train and Evaluate
    results = {}
    print("\nTraining and evaluating models...")
    for name, model in models.items():
        # Train
        model.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
            
        # Evaluate
        results[name] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred),
            'ROC-AUC': roc_auc_score(y_test, y_prob)
        }
        
    # Print results
    print("\n" + "="*60)
    print("--- Model Performance ---")
    print("="*60)
    results_df = pd.DataFrame(results).T
    print(results_df.round(4).to_string())
    
    # 6. Feature Importance (from Random Forest)
    print("\n" + "="*60)
    print("--- Key Features (Random Forest Feature Importances) ---")
    print("="*60)
    rf_model = models['Random Forest']
    feature_importances = pd.DataFrame(
        rf_model.feature_importances_,
        index = X.columns,
        columns=['Importance']
    ).sort_values('Importance', ascending=False)
    print(feature_importances.round(4).to_string())
    
    # Save results to CSV for easy viewing later
    results_df.to_csv('model_evaluation_results.csv')
    feature_importances.to_csv('feature_importances.csv')
    print("\nResults saved to 'model_evaluation_results.csv' and 'feature_importances.csv'.")

if __name__ == "__main__":
    main()
