import numpy as np
import pandas as pd


def generate_scenario() -> dict:
    """
    Stage 2 — Training Monitor.

    Data: 500 rows, cleanly formatted. No NaNs, no extreme outliers.
    Bug: MLPClassifier diverges (loss goes NaN) because features are wildly unscaled
         (e.g., salary around 70,000 mixed with age around 40).
    
    The agent must diagnose the issue from the logs and add a StandardScaler.
    """

    rng = np.random.default_rng(42)
    n = 500

    age = rng.integers(18, 71, size=n).astype(float)
    salary = np.clip(rng.normal(70000, 20000, size=n), 20000, 150000)
    credit_score = np.clip(rng.normal(680, 80, size=n), 300, 850)
    loan_amount = np.clip(rng.normal(20000, 10000, size=n), 1000, 50000)
    employment_years = np.clip(rng.normal(8, 5, size=n), 0, 40).astype(float)

    # Binary target influenced by features
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

    broken_script = (
        "import pandas as pd\n"
        "from sklearn.neural_network import MLPClassifier\n"
        "from sklearn.model_selection import train_test_split\n"
        "\n"
        "X = df[['age', 'salary', 'credit_score', 'loan_amount', 'employment_years']].copy()\n"
        "y = df['target']\n"
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n"
        "clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=10,\n"
        "                    random_state=42, verbose=True)\n"
        "clf.fit(X_train, y_train)\n"
        "print('Accuracy:', clf.score(X_test, y_test))\n"
    )

    initial_error_log = (
        "Epoch 1/10 - loss: 0.693147\n"
        "Epoch 2/10 - loss: nan\n"
        "Epoch 3/10 - loss: nan\n"
        "Training diverged. Check feature scaling."
    )

    task_description = (
        "Training is diverging. Loss went NaN after epoch 1. "
        "Diagnose the issue and fix the training script."
    )

    expected_fix_hint = "add StandardScaler before MLPClassifier"

    return {
        "broken_script": broken_script,
        "dataframe": df,
        "initial_error_log": initial_error_log,
        "task_description": task_description,
        "expected_fix_hint": expected_fix_hint,
        "stage_number": 2,
        "actor_trigger_step": 3,
    }
