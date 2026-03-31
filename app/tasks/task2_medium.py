import numpy as np
import pandas as pd


def generate_scenario() -> dict:
    """
    Task 2 (Medium): NaNs and outliers destabilize scaling.

    - Data: ~500 rows with columns: age, salary, credit_score, loan_amount, employment_years, target
    - Bug: StandardScaler.fit_transform applied to X with NaNs -> ValueError: Input contains NaN
    - Also includes extreme outlier salary values (9999999.0)
    - Deterministic via seed=42
    """

    rng = np.random.default_rng(42)
    n = 500

    age = rng.integers(18, 71, size=n).astype(float)
    salary = np.clip(rng.normal(70000, 20000, size=n), 20000, None)
    credit_score = np.clip(rng.normal(680, 80, size=n), 300, 850)
    loan_amount = np.clip(rng.normal(20000, 10000, size=n), 1000, None)
    employment_years = np.clip(rng.normal(8, 5, size=n), 0, 40).astype(float)

    # Binary target influenced by features with noise
    logits = (
        (age - 40) * -0.03
        + (salary - 70000) / 50000
        + (credit_score - 650) * 0.004
        - (loan_amount - 20000) / 40000
        + (employment_years - 8) * 0.05
        + rng.normal(0, 0.8, size=n)
    )
    prob = 1 / (1 + np.exp(-logits))
    target = (prob > 0.5).astype(int)

    df = pd.DataFrame(
        {
            "age": age,
            "salary": salary,
            "credit_score": credit_score,
            "loan_amount": loan_amount,
            "employment_years": employment_years,
            "target": target,
        }
    )

    # Inject ~15% rows with NaNs across a subset of columns
    nan_rows = rng.choice(np.arange(n), size=int(0.15 * n), replace=False)
    for col in ["age", "credit_score", "employment_years"]:
        df.loc[nan_rows, col] = np.nan

    # Inject ~10 extreme salary outliers
    outlier_rows = rng.choice(np.setdiff1d(np.arange(n), nan_rows), size=10, replace=False)
    df.loc[outlier_rows, "salary"] = 9_999_999.0

    broken_script = (
        "# Broken: scales before cleaning NaNs/outliers, causing ValueError on NaNs\n"
        "from sklearn.preprocessing import StandardScaler\n"
        "from sklearn.linear_model import LogisticRegression\n"
        "from sklearn.model_selection import train_test_split\n"
        "\n"
        "X = df[['age','salary','credit_score','loan_amount','employment_years']].copy()\n"
        "y = df['target']\n"
        "scaler = StandardScaler()\n"
        "X_scaled = scaler.fit_transform(X)  # ValueError: Input contains NaN\n"
        "X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)\n"
        "clf = LogisticRegression(max_iter=1000, random_state=42)\n"
        "clf.fit(X_train, y_train)\n"
        "print('Accuracy:', clf.score(X_test, y_test))\n"
    )

    initial_error_log = (
        "Traceback (most recent call last):\n"
        "  File '<script>', line 1, in <module>\n"
        "ValueError: Input contains NaN\n"
    )

    task_description = (
        "The dataset contains ~15% NaNs and extreme salary outliers. The script scales before "
        "handling missing values, causing StandardScaler to crash with a NaN error."
    )

    expected_fix_hint = (
        "Inspect to find NaNs and outlier ranges. Clean with dropna/fillna and clip/quantile capping, "
        "then fit the scaler on the cleaned training data after the train_test_split."
    )

    return {
        "broken_script": broken_script,
        "dataframe": df,
        "initial_error_log": initial_error_log,
        "task_description": task_description,
        "expected_fix_hint": expected_fix_hint,
    }

