import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn

def load_data(folder_path):
    # Memuat data dari folder hasil preprocessing
    X_train = pd.read_csv(f'{folder_path}/X_train.csv')
    X_test = pd.read_csv(f'{folder_path}/X_test.csv')
    y_train = pd.read_csv(f'{folder_path}/y_train.csv').values.ravel()
    y_test = pd.read_csv(f'{folder_path}/y_test.csv').values.ravel()
    return X_train, X_test, y_train, y_test

def train_with_tuning():
    # 1. Load Data
    print("Loading data...")
    X_train, X_test, y_train, y_test = load_data('telco_churn_preprocessing')

    # Experiment sudah diset dari 'mlflow run --experiment-name' di workflow
    # Tidak perlu set_experiment() lagi di sini

    # 3. Definisi Hyperparameter untuk Tuning
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }

    # Menggunakan run yang sudah dibuat oleh 'mlflow run' command
    # Tidak perlu start_run() baru karena sudah ada active run
    
    mlflow.set_tag("mlflow.runName", "Hyperparameter_Tuning_RF")

    print("Mulai Hyperparameter Tuning...")
    rf = RandomForestClassifier(random_state=42)
    
    # Menggunakan GridSearchCV
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    # Ambil model terbaik
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    print(f"Model terbaik ditemukan: {best_params}")

    # 5. Evaluasi Model Terbaik
    y_pred = best_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Metrics -> Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}")

    # 6. MANUAL LOGGING (Syarat Wajib Skilled)
    
    # a. Log Parameters
    for param_name, param_value in best_params.items():
        mlflow.log_param(param_name, param_value)
    
    # b. Log Metrics
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)

    # c. Log Model (Menyimpan Artefak)
    mlflow.sklearn.log_model(best_model, name="model")
    
    print("Logging ke MLflow selesai.")

if __name__ == "__main__":
    train_with_tuning()