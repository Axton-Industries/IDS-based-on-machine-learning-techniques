import numpy as np
import optuna
import tensorflow as tf
import math

from tensorflow.keras import layers, Model
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import average_precision_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from optuna.integration import TFKerasPruningCallback


# Silence noisy trial outputs to maximize printing performance
optuna.logging.set_verbosity(optuna.logging.WARNING)


# --- 1. AUXILIARY FUNCTIONS ---
def get_cv(n_splits=2, stratified=True):  # Fast baseline setup: 2 folds
    return (StratifiedKFold if stratified else KFold)(
        n_splits=n_splits, shuffle=True, random_state=42
    )


# --- 2. TUNNING FUNCTIONS FOR THE MODELS ---
def tune_xgboost(X_train, y_train, scale_pos_weight, n_trials=30, n_splits=3):
    """
    Research-Grade XGBoost Hyperparameter Tuning Engine.

    Optimized for maximizing classification performance boundaries using a
    regularized search space and robust cross-validation within a practical,
    GPU-accelerated timeframe.
    """
    # Enforce float32 immediately to protect VRAM limits
    X, y = np.asarray(X_train, np.float32), np.asarray(y_train)
    cv = get_cv(n_splits, stratified=True)

    # --- ADVANCED RESEARCH SEARCH SPACE ---
    # Broadened ranges paired with explicit regularization to prevent overfitting
    space = {
        "learning_rate": (0.01, 0.2, True),       # Logarithmic scale prioritizing granular optimization steps
        "max_depth": (3, 8),                      # Higher ceiling captures highly complex, nested interactions
        "subsample": (0.6, 1.0),                  # Row bagging bounds to maintain ensemble stochasticity
        "colsample_bytree": (0.6, 1.0),           # Feature column bagging bounds to prevent specific component dominance
        "min_child_weight": (1, 10),              # Controls minimal node split weight constraints
        "gamma": (1e-3, 5.0, True),               # Pseudo-logarithmic regularization (minimum loss reduction required to split)
        "alpha": (1e-8, 10.0, True),              # L1 Regularization (Lasso) to naturally zero out weak feature noise
        "reg_lambda": (1e-3, 10.0, True)          # L2 Regularization (Ridge) to stabilize extreme localized weight updates
    }

    def objective(trial):
        params = {
            k: trial.suggest_float(k, v[0], v[1], log=v[2] if len(v) == 3 else False)
            if isinstance(v[0], float) or (len(v) == 3 and v[2])
            else trial.suggest_int(k, *v)
            for k, v in space.items()
        }

        scores = []
        for fold, (tr, va) in enumerate(cv.split(X, y)):
            # Instantiate an isolated, independent model wrapper per fold split
            model = XGBClassifier(
                **params,
                # --- RIGOROUS TRAINING CAPACITY ---
                n_estimators=1500,        # Large tree budget; optimization is handled dynamically by early stopping
                early_stopping_rounds=50,  # Generous lookahead window to prevent premature stagnation termination
                scale_pos_weight=scale_pos_weight,
                tree_method="hist",
                device="cuda",             # Hardware acceleration binding
                eval_metric="logloss",
                verbosity=0,
                random_state=42,
                n_jobs=-1,
            )

            model.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False)

            # Extract full probability maps explicitly tracking optimal early-stopped iteration bounds
            preds = model.get_booster().inplace_predict(
                X[va],
                predict_type="probability",
                iteration_range=(0, model.best_iteration + 1)
            )

            # Use out-of-fold average precision score for evaluation stability
            score = average_precision_score(y[va], preds)
            scores.append(score)

        # Compute cross-validated metric distribution mean
        final_cv_score = np.mean(scores) if scores else 0.0
        trial.report(final_cv_score, step=1)

        return final_cv_score

    # Balanced Pruner: Collects 5 complete baseline trial trends before cutting unpromising candidates
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    return study.best_params


def tune_rf(X_train, y_train, n_trials=10, n_splits=2):
    X, y = np.asarray(X_train, dtype=np.float32), np.asarray(y_train)
    cv = get_cv(n_splits=n_splits, stratified=True)

    def objective(trial):
        # --- BALANCED PERFORMANCE SEARCH SPACE ---
        # Expanded ranges for better results, capped slightly to prevent CPU stalls.
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 150),          # Balanced: Enough trees for stability (Later: 300)
            "max_depth": trial.suggest_int("max_depth", 8, 20),                   # Balanced: Deep enough for complex patterns (Later: 25)
            "max_features": trial.suggest_float("max_features", 0.2, 0.6),        # Feature percentage sampled per split
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 15),   # Controls structural splitting constraints
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),      # Prevents micro-overfitting on small noise pockets
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
            "class_weight": "balanced",
            "random_state": 42,
        }

        scores = []
        for fold, (tr, va) in enumerate(cv.split(X, y)):
            model = RandomForestClassifier(**params, n_jobs=-1)
            model.fit(X[tr], y[tr])

            # Binary evaluation: isolate probability vectors for the positive class (1)
            preds = model.predict_proba(X[va])[:, 1]
            score = average_precision_score(y[va], preds)
            scores.append(score)

        return np.mean(scores) if scores else 0.0

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    return study.best_params


def tune_if(X_train, n_trials=20):
    X = np.asarray(X_train, dtype=np.float32)

    # --- DATA SUBSAMPLING ---
    # Downsample large datasets to 50,000 rows to keep training times fast
    # while retaining enough statistical variance to capture complex attack profiles.
    row_cap = 50000
    if X.shape[0] > row_cap:
        np.random.seed(42)
        X_tune = X[np.random.choice(X.shape[0], row_cap, replace=False)]
    else:
        X_tune = X

    def objective(trial):
        # --- TUNING SEARCH SPACE ---
        params = {
            # More trees yield smoother anomaly scores and prevent individual split bias
            "n_estimators": trial.suggest_int("n_estimators", 100, 250),

            # The percentage of rows used to build each individual isolation tree
            "max_samples": trial.suggest_float("max_samples", 0.5, 1.0),

            # The percentage of features evaluated; crucial for catching stealthy multi-column attacks
            "max_features": trial.suggest_float("max_features", 0.5, 1.0),

            # The expected proportion of outliers/attacks present in the dataset
            "contamination": trial.suggest_float("contamination", 0.005, 0.15),

            "random_state": 42,
            "n_jobs": -1  # Maximize CPU core usage across independent trees
        }

        # --- MODEL FITTING & PREDICTION ---
        model = IsolationForest(**params)
        preds = model.fit_predict(X_tune)         # Outputs: 1 for normal, -1 for anomaly
        scores = model.decision_function(X_tune)  # Raw anomaly score: higher means more normal

        # --- TRIAL GUARD RAIL ---
        # Rejects the trial completely if the parameter choice forces all traffic into a single class
        if len(np.unique(preds)) < 2:
            return -1.0

        # --- UNSUPERVISED METRIC ENGINE ---
        # Isolate score arrays based on the model's internal class splits
        normal_scores = scores[preds == 1]
        anomaly_scores = scores[preds == -1]

        # Calculate how far apart the normal cluster center is from the anomaly cluster center
        score_separation = np.mean(normal_scores) - np.mean(anomaly_scores)

        # Calculate the density/variance of the normal data points
        normal_variance = np.var(normal_scores)

        # GOAL: Maximize cluster distance (separation) while keeping normal data tightly grouped (low variance).
        # We add 1e-6 to the denominator to prevent math division errors.
        unsupervised_score = score_separation / (normal_variance + 1e-6)

        return unsupervised_score

    # Run the optimizer to find parameters that yield the highest unsupervised separation score
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    return study.best_params


def tune_autoencoder(X_train, n_trials=20):
    X = np.asarray(X_train, dtype=np.float32)
    input_dim = X.shape[1]

    # --- DATA SUBSAMPLING & SPLITTING ---
    # Cap training/validation to 50,000 rows to keep epoch training times fast
    # while ensuring a clean validation split to protect against overfitting.
    row_cap = 50000
    if X.shape[0] > row_cap:
        np.random.seed(42)
        X_tune = X[np.random.choice(X.shape[0], row_cap, replace=False)]
    else:
        X_tune = X

    split_idx = int(len(X_tune) * 0.8)
    X_tr, X_val = X_tune[:split_idx], X_tune[split_idx:]

    def objective(trial):
        tf.keras.backend.clear_session()  # Prevent GPU/RAM memory leaks

        # --- TUNING SEARCH SPACE ---
        # A deep, bottlenecked representation designed to learn tightly bound normal profiles.
        latent_dim = trial.suggest_int("latent_dim", 4, 32)
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])
        n1 = trial.suggest_int("n1", 64, 256)
        n2 = trial.suggest_int("n2", 32, n1)  # Tapering constraint forces compression
        activation = trial.suggest_categorical("activation", ["leaky_relu", "swish"])
        dropout = trial.suggest_float("dropout", 0.0, 0.15)
        reg = tf.keras.regularizers.l2(trial.suggest_float("reg_l2", 1e-6, 1e-3, log=True))

        # Simulated contamination expectation for post-reconstruction anomaly thresholding
        contamination = trial.suggest_float("contamination", 0.005, 0.15)

        # --- ARCHITECTURE ---
        inp = layers.Input(shape=(input_dim,))

        # Encoder (Advanced activations preserve RobustScaler continuous ranges)
        x = layers.Dense(n1, activation=activation, kernel_regularizer=reg)(inp)
        if dropout > 0: x = layers.Dropout(dropout)(x)
        x = layers.Dense(n2, activation=activation, kernel_regularizer=reg)(x)
        if dropout > 0: x = layers.Dropout(dropout)(x)

        # Bottleneck: Linear activation completely protects negative features
        z = layers.Dense(latent_dim, activation="linear")(x)

        # Decoder: Flips the encoder architecture to map hidden states back to inputs
        x = layers.Dense(n2, activation=activation, kernel_regularizer=reg)(z)
        x = layers.Dense(n1, activation=activation, kernel_regularizer=reg)(x)
        out = layers.Dense(input_dim, activation="linear")(x)

        # --- MODEL FITTING & PREDICTION ---
        model = Model(inp, out)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mae")

        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            optuna.integration.TFKerasPruningCallback(trial, "val_loss")  # Cut unpromising trials mid-run
        ]

        history = model.fit(
            X_tr, X_tr, epochs=30, batch_size=batch_size,
            validation_data=(X_val, X_val), callbacks=callbacks, verbose=0
        )

        # --- TRIAL GUARD RAIL ---
        # Rejects the trial completely if the architecture explodes, collapses, or fails to log errors safely
        val_loss_hist = history.history.get("val_loss", [])
        if not val_loss_hist or math.isnan(val_loss_hist[-1]) or val_loss_hist[-1] > 5.0:
            return -1.0

        # --- UNSUPERVISED METRIC ENGINE ---
        # Generate reconstruction scores (Mean Absolute Error per sample) on validation data
        recon = model.predict(X_val, verbose=0)
        scores = np.mean(np.abs(X_val - recon), axis=1)

        # Isolate score arrays based on the trial's expected contamination boundary
        threshold = np.percentile(scores, 100 * (1 - contamination))

        # Split traffic mimicking the Isolation forest's predictions (1 = normal, -1 = anomaly)
        normal_scores = scores[scores <= threshold]
        anomaly_scores = scores[scores > threshold]

        # Calculate how far apart the normal cluster center is from the anomaly cluster center.
        # For Autoencoders, anomaly reconstruction scores are expected to be much higher.
        score_separation = np.mean(anomaly_scores) - np.mean(normal_scores)

        # Calculate the density/variance of the normal data points
        normal_variance = np.var(normal_scores)

        # GOAL: Maximize reconstruction contrast (separation) while keeping normal data tightly grouped (low variance).
        # We add 1e-6 to the denominator to prevent math division errors.
        unsupervised_score = score_separation / (normal_variance + 1e-6)

        return float(unsupervised_score)

    # Run the optimizer to find parameters that yield the highest unsupervised separation score
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    return study.best_params




# --- Multiclas functions ---

def tune_xgboost_multiclass(X_train, y_train, n_trials=10, n_splits=2):
    """XGBoost hyperparameter tuning engine for Multiclass tasks."""
    X, y = np.asarray(X_train, np.float32), np.asarray(y_train)
    cv = get_cv(n_splits, stratified=True)

    space = {
        "learning_rate": (0.05, 0.1, False),
        "max_depth": (3, 5),
        "subsample": (0.8, 1.0),
        "colsample_bytree": (0.8, 1.0),
        "min_child_weight": (1, 5),
    }

    def objective(trial):
        params = {
            k: trial.suggest_float(k, v[0], v[1], log=v[2] if len(v) == 3 else False)
            if isinstance(v[0], float) or (len(v) == 3 and v[2])
            else trial.suggest_int(k, *v)
            for k, v in space.items()
        }

        scores = []
        for fold, (tr, va) in enumerate(cv.split(X, y)):
            # Dynamically compute balanced class weights for the training fold
            classes = np.unique(y[tr])
            weights = compute_class_weight(class_weight='balanced', classes=classes, y=y[tr])
            w_dict = dict(zip(classes, weights))
            fold_sample_weights = np.array([w_dict[cls] for cls in y[tr]])

            model = XGBClassifier(
                **params, n_estimators=100, early_stopping_rounds=15, tree_method="hist",
                device="cuda", eval_metric="mlogloss", verbosity=0, random_state=42, n_jobs=-1,
            )

            model.fit(X[tr], y[tr], sample_weight=fold_sample_weights, eval_set=[(X[va], y[va])], verbose=False)

            # In multiclass, inplace_predict yields a matrix of shape (n_samples, n_classes)
            probs = model.get_booster().inplace_predict(X[va], predict_type="probability", iteration_range=(0, model.best_iteration + 1))

            # Convert probabilities to hard class predictions
            preds = np.argmax(probs, axis=1)

            # Calculate Macro F1 Score (robust choice for multiclass)
            score = f1_score(y[va], preds, average="macro")
            scores.append(score)

        final_cv_score = np.mean(scores) if scores else 0.0
        trial.report(final_cv_score, step=1)
        return final_cv_score

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=1),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    return study.best_params


def tune_rf_multiclass(X_train, y_train, n_trials=5):
    """Random Forest hyperparameter tuning engine for Multiclass tasks."""
    X, y = np.asarray(X_train, dtype=np.float32), np.asarray(y_train)
    cv = get_cv(n_splits=2, stratified=True)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 100),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "min_samples_split": trial.suggest_int("min_samples_split", 10, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 15),
            "max_features": "sqrt",
            "class_weight": "balanced",
            "random_state": 42,
        }

        scores = []
        for fold, (tr, va) in enumerate(cv.split(X, y)):
            model = RandomForestClassifier(**params, n_jobs=-1)
            model.fit(X[tr], y[tr])

            # Get hard predictions directly for class assignment
            preds = model.predict(X[va])

            # Calculate Macro F1 score
            score = f1_score(y[va], preds, average="macro")
            scores.append(score)

        # Calculate final CV score first, then report it to Optuna safely at the end of the trial
        final_cv_score = np.mean(scores) if scores else 0.0
        trial.report(final_cv_score, step=1)

        return final_cv_score

    # Added pruning mechanics to align with your XGBoost pipeline speed strategies
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=1),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    return study.best_params
