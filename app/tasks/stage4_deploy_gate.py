import numpy as np
import pandas as pd


def generate_scenario() -> dict:
    rng = np.random.default_rng(42)
    n = 1000

    age = np.concatenate(
        [
            rng.integers(22, 40, size=650),
            rng.integers(40, 70, size=350),
        ]
    ).astype(float)
    rng.shuffle(age)
    older_group = (age >= 40).astype(float)

    salary = np.clip(
        rng.normal(68_000, 18_000, size=n) + (age - 40) * 900,
        25_000,
        180_000,
    )
    credit_score = np.clip(
        rng.normal(665, 65, size=n) + (salary - 68_000) / 3_000 - older_group * 40,
        350,
        850,
    )
    loan_amount = np.clip(
        rng.normal(23_000, 9_000, size=n) - (credit_score - 665) * 20 + older_group * 3_000,
        3_000,
        70_000,
    )
    employment_years = np.clip((age - 22) * 0.45 + rng.normal(0, 4, size=n), 0, 45)

    logits = (
        (credit_score - 660) * 0.012
        + (salary - 68_000) / 35_000
        - (loan_amount - 23_000) / 18_000
        + employment_years * 0.035
        - older_group * 0.7
        + rng.normal(0, 0.9, size=n)
    )
    target = (logits > 0.3).astype(int)

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

    broken_script = (
        "import pandas as pd\n"
        "from sklearn.linear_model import LogisticRegression\n"
        "from sklearn.metrics import accuracy_score\n"
        "from sklearn.model_selection import train_test_split\n"
        "from sklearn.preprocessing import StandardScaler\n"
        "\n"
        "features = ['age', 'salary', 'credit_score', 'loan_amount', 'employment_years']\n"
        "X = df[features].copy()\n"
        "y = df['target']\n"
        "X_train, X_test, y_train, y_test = train_test_split(\n"
        "    X, y, test_size=0.25, random_state=42, stratify=y\n"
        ")\n"
        "\n"
        "scaler = StandardScaler()\n"
        "X_train_scaled = scaler.fit_transform(X_train)\n"
        "X_test_scaled = scaler.transform(X_test)\n"
        "\n"
        "clf = LogisticRegression(max_iter=1000, random_state=42)\n"
        "clf.fit(X_train_scaled, y_train)\n"
        "pred = clf.predict(X_test_scaled)\n"
        "print('Accuracy:', round(accuracy_score(y_test, pred), 2))\n"
    )

    return {
        "broken_script": broken_script,
        "dataframe": df,
        "deploy_conditions": {
            "accuracy_threshold": 0.75,
            "max_inference_ms": 100,
            "max_fairness_gap": 0.05,
        },
        "initial_error_log": (
            "DEPLOY GATE CHECK:\n"
            "[PASS] Accuracy: 0.77 >= 0.75\n"
            "[PASS] Inference time: 45ms <= 100ms\n"
            "[FAIL] Fairness gap: 0.09 > 0.05 (age groups differ significantly)\n"
            "STATUS: REJECTED \u2014 fix fairness issue before deployment"
        ),
        "task_description": (
            "Deployment rejected. Fix the fairness issue. "
            "The model performs significantly worse on one age group. "
            "All 3 conditions must pass for deployment approval."
        ),
        "expected_fix_hint": (
            "add class_weight='balanced' to classifier OR "
            "add age-group stratified sampling in train_test_split"
        ),
        "stage_number": 4,
    }
