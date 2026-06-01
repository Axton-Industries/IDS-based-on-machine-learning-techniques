import numpy as np
import optuna
import tensorflow as tf

from tensorflow.keras import layers, Model
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import average_precision_score
from optuna.integration import TFKerasPruningCallback

# Auxiliary functions
# Silence noisy trial outputs to maximize printing performance
optuna.logging.set_verbosity(optuna.logging.WARNING)


# --- 1. OPTIMIZED CV ENGINE ---
def get_cv(n_splits=2, stratified=True):  # Fast baseline setup: 2 folds
    return (StratifiedKFold if stratified else KFold)(
        n_splits=n_splits, shuffle=True, random_state=42
    )


def cv_score(model_fn, X, y=None, cv=None, metric=None):
    cv = cv or get_cv()
    if y is not None:
        return np.mean(
            [
                metric(y[va], model_fn().fit(X[tr], y[tr]).predict_proba(X[va])[:, 1])
                for tr, va in cv.split(X, y)
            ]
        )
    return np.mean(
        [
            np.std(model_fn().fit(X[tr]).decision_function(X[va]))
            for tr, va in cv.split(X)
        ]
    )


# --- 2. TUNNING FUNCTIONS FOR THE MODELS ---
def tune_xgboost(X_train, y_train, scale_pos_weight, n_trials=10, n_splits=2):
    """XGBoost hyperparameter tuning engine.
    """
    # Enforce float32 immediately to protect  VRAM limits
    X, y = np.asarray(X_train, np.float32), np.asarray(y_train)
    cv = get_cv(n_splits, stratified=True)

    # --- FAST TUNING SEARCH SPACE ---
    # Broad parameters are narrow here to run fast. Deeper ranges are commented out below.
    space = {
        "learning_rate": (0.05, 0.1, False),  # Fast: High base step size. (Later: 0.01, 0.2, True)
        "max_depth": (3, 5),                  # Fast: Small tree structures. (Later: 3, 7 or 8)
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
            model = XGBClassifier(
                **params,
                # --- SPEED SWITCHES ---
                n_estimators=100,         # FAST RUNS: Use 100 trees. | PRODUCTION LATER: Use 1000 or 2000
                early_stopping_rounds=15, # FAST RUNS: Stop quickly if flat. | PRODUCTION LATER: Use 50 or 100
                scale_pos_weight=scale_pos_weight,
                tree_method="hist",
                device="cuda",             # Enforces MX150 CUDA training execution
                eval_metric="logloss",
                verbosity=0,
                random_state=42,
                n_jobs=-1,
            )

            model.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False)

            preds = model.get_booster().inplace_predict(X[va], predict_type="probability")
            score = average_precision_score(y[va], preds)
            scores.append(score)

            # Optuna Pruning: Kills a trial immediately after fold 0 if it looks bad
            trial.report(score, fold)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return np.mean(scores) if scores else 0.0

    # Aggressive Pruning: Triggers after only 1 warmup trial to keep speed elevated
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=1), # FAST: 1 trial warmup | LATER: 5 trials
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    return study.best_params



def tune_rf(X_train, y_train, n_trials=5):
    """Random Forest hyperparameter tuning engine.

    Stripped down for immediate, lightning-fast execution on the CPU.
    """
    X, y = np.asarray(X_train), np.asarray(y_train)
    cv = get_cv(n_splits=2, stratified=True)

    def objective(trial):
        # --- FAST TUNING SEARCH SPACE ---
        # Broad parameters are narrow here to run fast. Deeper ranges are commented out below.
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 100),       # Fast: Fewer trees for rapid testing (Later: 100, 300)
            "max_depth": trial.suggest_int("max_depth", 3, 8),                # Fast: Shallow depth avoids heavy CPU computations (Later: 5, 20)
            "min_samples_split": trial.suggest_int("min_samples_split", 10, 20), # Fast: Larger splits stop tree growth early (Later: 2, 15)
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 15),   # Fast: Keeps leaves larger to decrease train times (Later: 1, 8)
            "max_features": "sqrt",
            "class_weight": "balanced",
            "n_jobs": -1,                                                     # Enforces full CPU multi-core parallelization
            "random_state": 42
        }

        scores = []
        for fold, (tr, va) in enumerate(cv.split(X, y)):
            model = RandomForestClassifier(**params)
            model.fit(X[tr], y[tr])

            preds = model.predict_proba(X[va])[:, 1]
            score = average_precision_score(y[va], preds)
            scores.append(score)

        return np.mean(scores) if scores else 0.0

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    return study.best_params



def tune_if(X_train, n_trials=5):

    X = np.asarray(X_train)
    cv = get_cv(n_splits=2)

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




def tune_autoencoder(X_train, n_trials=5): # FAST RUNS: 5 trials to check everything works | PRODUCTION: 20 to 50 trials
    """Autoencoder tuning engine optimized for maximum discrimination contrast.

    Configured for rapid execution. Production parameters are commented below.
    """
    # Enforce float32 immediately to protect GPU memory limits
    X = np.asarray(X_train, np.float32)[:5000] # FAST RUNS: Sample 5000 rows | PRODUCTION: Use full X_train
    input_dim = X.shape[1]

    def objective(trial):
        tf.keras.backend.clear_session()

        # --- LATENT (BOTTLENECK) DIMENSION ---
        latent_dim = trial.suggest_int("latent_dim", 4, 8)
        # PRODUCTION LATER: trial.suggest_int("latent_dim", 4, 24)

        # --- LEARNING RATE ---
        lr = trial.suggest_float("lr", 1e-3, 3e-3, log=True)
        # PRODUCTION LATER: trial.suggest_float("lr", 1e-4, 3e-3, log=True)

        # --- BATCH SIZE ---
        batch_size = trial.suggest_categorical("batch_size", [256])
        # PRODUCTION LATER: trial.suggest_categorical("batch_size", [128, 256])

        # --- NETWORK ARCHITECTURE REPRESENTATION ---
        n1 = trial.suggest_int("n1", 32, 64)                # FAST RUNS: Narrow width (Production: 32, 128)
        n2 = trial.suggest_int("n2", 16, n1)                # FAST RUNS: Enforce bottleneck architecture
        dropout = trial.suggest_float("dropout", 0.0, 0.1)  # FAST RUNS: Low dropout to speed up convergence (Production: 0.0, 0.3)

        # --- MODEL BUILD ---
        inp = layers.Input(shape=(input_dim,))

        # Encoder Pipeline
        x = layers.Dense(n1, activation="relu")(inp)
        x = layers.Dropout(dropout)(x)
        x = layers.Dense(n2, activation="relu")(x)
        x = layers.Dropout(dropout)(x)

        # Bottleneck Latent Space
        z = layers.Dense(latent_dim, activation="relu")(x)

        # Decoder Pipeline
        x = layers.Dense(n2, activation="relu")(z)
        x = layers.Dense(n1, activation="relu")(x)
        out = layers.Dense(input_dim)(x)

        model = Model(inp, out)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss="mae"
        )

        # --- FAST TRAINING REGULATORS ---
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=1,                  # FAST RUNS: Stop immediately if no drop (Production: 3 or 5)
                restore_best_weights=True
            ),
            TFKerasPruningCallback(trial, "val_loss")
        ]

        # Fit using a clean validation split
        history = model.fit(
            X, X,
            epochs=3,                        # FAST RUNS: 3 epochs to test pipeline speed (Production: 15 to 30 epochs)
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=0
        )

        # Safeguard against gradient explosion errors
        val_mae = history.history["val_loss"][-1]
        if np.isnan(val_mae) or val_mae > 10.0:
            return -float("inf")

        # --- CONTRAST MAXIMIZATION METRIC ---
        recon = model.predict(X, verbose=0)
        error = np.mean(np.abs(X - recon), axis=1)

        # Relative Contrast Ratio optimization metric
        discrimination_power = (np.percentile(error, 95) - np.percentile(error, 5)) / (np.mean(error) + 1e-6)

        return discrimination_power

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=2)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    return study.best_params
