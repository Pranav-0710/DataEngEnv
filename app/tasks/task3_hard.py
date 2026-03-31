import numpy as np
import pandas as pd


def _make_data(rng: np.random.Generator, n: int) -> tuple[pd.DataFrame, np.ndarray]:
    """Generate a dataset with moderate separability so that proper training yields ~0.72-0.78 accuracy.
    Deterministic given rng.
    Returns (df, y).
    """
    # Binary labels 0/1
    y = rng.integers(0, 2, size=n)

    # Features: class-dependent means with substantial noise
    # This yields a problem that is learnable but not near-perfect.
    shift = (y * 2 - 1).reshape(-1, 1)  # -1 for class 0, +1 for class 1
    base = rng.normal(0, 1.5, size=(n, 5))
    means = np.array([0.8, 0.6, 0.5, 0.3, 0.0])
    X = base + shift * means  # class-separable signal

    # Add one weakly informative interaction-like feature
    x_extra = (X[:, 0] * 0.3 + X[:, 1] * -0.2 + rng.normal(0, 0.7, size=n)).reshape(-1, 1)
    X = np.concatenate([X, x_extra], axis=1)

    df = pd.DataFrame(X, columns=[
        "feat_1",
        "feat_2",
        "feat_3",
        "feat_4",
        "feat_5",
        "feat_6",
    ])

    df["target"] = y
    return df, y


def generate_scenario() -> dict:
    """
    Task 3 (Hard): Data leakage via scaling before split.

    - Data: ~1000 rows; model accuracy appears suspiciously high when StandardScaler is fit on the full dataset
    - Bug: StandardScaler.fit_transform applied before train_test_split (data leakage)
    - True accuracy after fixing scaling (fit on train only) should drop to ~0.72-0.78
    - Also returns held_out_X, held_out_y (NOT to be exposed externally)
    - Deterministic via seed=42
    """

    rng = np.random.default_rng(42)

    # Main dataset for the episode
    df, y = _make_data(rng, 1000)

    # Held-out set for grader3 only
    held_out_df, held_out_y = _make_data(rng, 200)
    held_out_X = held_out_df.drop(columns=["target"]).to_numpy()

    broken_script = (
        "# Broken: fits StandardScaler on FULL dataset before splitting (data leakage)\n"
        "import numpy as np\n"
        "from sklearn.preprocessing import StandardScaler\n"
        "from sklearn.linear_model import LogisticRegression\n"
        "from sklearn.model_selection import train_test_split\n"
        "\n"
        "X = df.drop(columns=['target']).to_numpy()\n"
        "y = df['target'].to_numpy()\n"
        "scaler = StandardScaler()\n"
        "X_scaled = scaler.fit_transform(X)  # data leakage: fit on full data\n"
        "X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)\n"
        "clf = LogisticRegression(max_iter=1000, random_state=42)\n"
        "clf.fit(X_train, y_train)\n"
        "acc = clf.score(X_test, y_test)\n"
        "print('Accuracy:', round(acc, 4))\n"
    )

    # No error on run; it will print an accuracy that looks inflated.
    initial_error_log = ""

    task_description = (
        "The script performs StandardScaler.fit_transform on the entire dataset before splitting into "
        "train/test, leaking information. This yields a suspiciously high reported accuracy."
    )

    expected_fix_hint = (
        "Move scaler.fit to occur AFTER train_test_split and fit ONLY on X_train, then transform X_train and X_test."
    )

    return {
        "broken_script": broken_script,
        "dataframe": df,
        "initial_error_log": initial_error_log,
        "task_description": task_description,
        "expected_fix_hint": expected_fix_hint,
        # Hidden extras for grader3 (never exposed via API)
        "held_out_X": held_out_X,
        "held_out_y": held_out_y,
        "full_df": df,
    }

