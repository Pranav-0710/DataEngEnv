import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "age",
    "salary",
    "credit_score",
    "loan_amount",
    "employment_years",
]


def _make_data(rng: np.random.Generator, n: int) -> pd.DataFrame:
    age = rng.integers(21, 70, size=n).astype(float)
    salary = np.clip(
        rng.normal(72_000, 22_000, size=n) + (age - 40) * 600,
        25_000,
        170_000,
    )
    credit_score = np.clip(
        rng.normal(675, 70, size=n) + (salary - 72_000) / 2_500 - (age - 40) * 0.6,
        350,
        850,
    )
    loan_amount = np.clip(
        rng.normal(24_000, 9_000, size=n) - (credit_score - 675) * 25 + (age - 40) * 120,
        3_000,
        65_000,
    )
    employment_years = np.clip((age - 22) * 0.45 + rng.normal(0, 4, size=n), 0, 42)

    logits = (
        (credit_score - 665) * 0.012
        + (salary - 72_000) / 35_000
        - (loan_amount - 24_000) / 18_000
        + employment_years * 0.045
        - (age - 42) * 0.018
        + rng.normal(0, 1.15, size=n)
    )
    target = (logits > 0.15).astype(int)

    return pd.DataFrame(
        {
            "age": age,
            "salary": salary,
            "credit_score": credit_score,
            "loan_amount": loan_amount,
            "employment_years": employment_years,
            "target": target,
        }
    )


def generate_scenario() -> dict:
    rng = np.random.default_rng(42)
    df = _make_data(rng, 1000)
    held_out_df = _make_data(rng, 200)

    held_out_X = held_out_df[FEATURE_COLUMNS].to_numpy()
    held_out_y = held_out_df["target"].to_numpy()

    broken_script = (
        "import pandas as pd\n"
        "from sklearn.ensemble import GradientBoostingClassifier\n"
        "from sklearn.model_selection import train_test_split\n"
        "from sklearn.preprocessing import StandardScaler\n"
        "\n"
        "X = df[['age', 'salary', 'credit_score', 'loan_amount', 'employment_years']].copy()\n"
        "y = df['target']\n"
        "\n"
        "scaler = StandardScaler()\n"
        "X_scaled = scaler.fit_transform(X)\n"
        "X_train, X_test, y_train, y_test = train_test_split(\n"
        "    X_scaled, y, test_size=0.2, random_state=42, stratify=y\n"
        ")\n"
        "\n"
        "clf = GradientBoostingClassifier(random_state=42)\n"
        "clf.fit(X_train, y_train)\n"
        "print('Accuracy: 0.98')\n"
    )

    return {
        "broken_script": broken_script,
        "dataframe": df,
        "held_out_X": held_out_X,
        "held_out_y": held_out_y,
        "full_df": df,
        "initial_error_log": "",
        "task_description": (
            "Model accuracy is 0.98. No errors. But something is wrong. "
            "Investigate the evaluation pipeline and fix any logical bugs."
        ),
        "expected_fix_hint": "move scaler.fit() to after train_test_split()",
        "stage_number": 3,
    }
