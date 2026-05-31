import numpy as np
import optuna
import tensorflow as tf

from tensorflow.keras import layers, Model
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import average_precision_score

# Auxiliary functions
def get_cv(n_splits=3, stratified=False):
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42) if stratified else \
           KFold(n_splits=n_splits, shuffle=True, random_state=42)


def cv_score(model_fn, X, y, cv, metric):
    """Reusable CV engine """
    scores = []

    for tr_idx, val_idx in cv.split(X, y if y is not None else None):
        model = model_fn()

        model.fit(X[tr_idx], y[tr_idx]) if y is not None else model.fit(X[tr_idx])

        preds = model.predict_proba(X[val_idx])[:, 1] if y is not None else model.decision_function(X[val_idx])

        scores.append(metric(y[val_idx], preds) if y is not None else np.std(preds))

    return np.mean(scores)


#Tunning functions
def tune_xgboost(X_train, y_train, scale_pos_weight, n_trials=10):

    X, y = np.asarray(X_train), np.asarray(y_train)
    cv = get_cv(n_splits=3, stratified=True)

    def objective(trial):

        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.1),
            "max_depth": trial.suggest_int("max_depth", 3, 6),
            "subsample": trial.suggest_float("subsample", 0.8, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.8, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 5),
            "gamma": trial.suggest_float("gamma", 0.0, 0.3),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 0.3),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 2.0),
        }

        def model_fn():
            return XGBClassifier(
                **params,
                n_estimators=200,  # reduced (was 400)
                scale_pos_weight=scale_pos_weight,
                tree_method="hist",
                device="cuda",
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1
            )

        return cv_score(model_fn, X, y, cv, average_precision_score)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    return study.best_params


def tune_rf(X_train, y_train, n_trials=5):

    X, y = np.asarray(X_train), np.asarray(y_train)
    cv = get_cv(n_splits=3, stratified=True)

    def objective(trial):

        def model_fn():
            return RandomForestClassifier(
                n_estimators=trial.suggest_int("n_estimators", 100, 300),
                max_depth=trial.suggest_int("max_depth", 5, 20),
                min_samples_split=trial.suggest_int("min_samples_split", 2, 15),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 8),
                max_features="sqrt",
                class_weight="balanced",
                n_jobs=-1,
                random_state=42
            )

        return cv_score(model_fn, X, y, cv, average_precision_score)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    return study.best_params


def tune_if(X_train, n_trials=5):

    X = np.asarray(X_train)
    cv = get_cv(n_splits=3)

    def objective(trial):

        def model_fn():
            return IsolationForest(
                n_estimators=trial.suggest_int("n_estimators", 50, 120),
                max_samples=trial.suggest_categorical("max_samples", ["auto", 0.7]),
                max_features=trial.suggest_float("max_features", 0.8, 1.0),
                contamination=trial.suggest_float("contamination", 0.01, 0.1),
                random_state=42,
                n_jobs=-1
            )

        return cv_score(model_fn, X, y=None, cv=cv, metric=None)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    return study.best_params


def tune_autoencoder(X_train, n_trials=20):

    X = np.asarray(X_train)[:5000]
    input_dim = X.shape[1]

    def objective(trial):

        tf.keras.backend.clear_session()

        latent_dim = trial.suggest_int("latent_dim", 4, 24)
        lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [128, 256])

        n1 = trial.suggest_int("n1", 32, 128)
        n2 = trial.suggest_int("n2", 16, n1)
        dropout = trial.suggest_float("dropout", 0.0, 0.3)

        inp = layers.Input(shape=(input_dim,))

        x = layers.Dense(n1, activation="relu")(inp)
        x = layers.Dropout(dropout)(x)

        x = layers.Dense(n2, activation="relu")(x)
        x = layers.Dropout(dropout)(x)

        z = layers.Dense(latent_dim, activation="relu")(x)

        x = layers.Dense(n2, activation="relu")(z)
        x = layers.Dense(n1, activation="relu")(x)
        out = layers.Dense(input_dim)(x)

        model = Model(inp, out)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss="mae"
        )

        model.fit(
            X, X,
            epochs=5,   # reduced (was 10)
            batch_size=batch_size,
            validation_split=0.1,
            verbose=0
        )

        recon = model.predict(X, verbose=0)
        error = np.mean(np.abs(X - recon), axis=1)

        return (np.percentile(error, 95) - np.percentile(error, 5)) + 0.1 * np.mean(error)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    return study.best_params
