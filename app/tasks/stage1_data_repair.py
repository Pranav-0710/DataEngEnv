import numpy as np
import pandas as pd


def generate_scenario() -> dict:
    """
    Stage 1 — Data Repair (combines old Task 1 + Task 2).

    Two-bug cascade:
      Bug 1: Script references 'age_years' but DataFrame column is 'age' → KeyError
      Bug 2: NaN values in data cause StandardScaler to crash → ValueError

    The agent must make TWO edit_script actions to fix both bugs sequentially.

    Data: 500 rows, seed=42, with ~15% NaN injection and 10 salary outliers.
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

    # Inject ~10 extreme salary outliers (non-overlapping with NaN rows)
    outlier_rows = rng.choice(
        np.setdiff1d(np.arange(n), nan_rows), size=10, replace=False
    )
    df.loc[outlier_rows, "salary"] = 9_999_999.0

    broken_script = (
        "import pandas as pd\n"
        "from sklearn.preprocessing import StandardScaler\n"
        "from sklearn.linear_model import LogisticRegression\n"
        "from sklearn.model_selection import train_test_split\n"
        "\n"
        "X = df[['age_years', 'salary', 'credit_score', 'loan_amount', 'employment_years']].copy()\n"
        "y = df['target']\n"
        "scaler = StandardScaler()\n"
        "X_scaled = scaler.fit_transform(X)\n"
        "X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)\n"
        "from sklearn.linear_model import LogisticRegression\n"
        "clf = LogisticRegression(max_iter=1000, random_state=42)\n"
        "clf.fit(X_train, y_train)\n"
        "print('Accuracy:', clf.score(X_test, y_test))\n"
    )

    initial_error_log = (
        "Traceback (most recent call last):\n"
        "  File '<script>', line 6, in <module>\n"
        "KeyError: 'age_years'\n"
    )

    task_description = (
        "Fix the broken pipeline: there are multiple bugs. "
        "Find and fix all of them."
    )

    expected_fix_hint = "fix age_years→age AND add dropna/clip before scaler"

    return {
        "broken_script": broken_script,
        "dataframe": df,
        "initial_error_log": initial_error_log,
        "task_description": task_description,
        "expected_fix_hint": expected_fix_hint,
        "stage_number": 1,
    }
