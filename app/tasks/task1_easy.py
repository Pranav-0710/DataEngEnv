import numpy as np
import pandas as pd


def generate_scenario() -> dict:
    """
    Task 1 (Easy): Column rename bug.

    - Data: ~200 synthetic employee rows with columns: age, salary, education_level, target
    - Bug: Script references 'age_years' but DataFrame has 'age' -> KeyError
    - Deterministic via seed=42
    """

    rng = np.random.default_rng(42)

    n = 200
    ages = rng.integers(18, 66, size=n)
    salary = np.clip(rng.normal(60000, 15000, size=n), 25000, None)
    education_levels = np.array(["HS", "Bachelors", "Masters", "PhD"]) 
    education_level = rng.choice(education_levels, size=n, replace=True)

    # Simple synthetic target influenced by age and salary (with noise)
    logits = (ages - 40) * 0.05 + (salary - 60000) / 40000 + rng.normal(0, 0.5, size=n)
    prob = 1 / (1 + np.exp(-logits))
    target = (prob > 0.5).astype(int)

    df = pd.DataFrame(
        {
            "age": ages,
            "salary": salary.astype(float),
            "education_level": education_level,
            "target": target,
        }
    )

    broken_script = (
        "# Broken: references 'age_years' (does not exist) instead of 'age'\n"
        "import pandas as pd\n"
        "from sklearn.model_selection import train_test_split\n"
        "from sklearn.preprocessing import StandardScaler\n"
        "from sklearn.linear_model import LogisticRegression\n"
        "\n"
        "X = df[['age_years', 'salary']].copy()\n"
        "y = df['target']\n"
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n"
        "scaler = StandardScaler()\n"
        "X_train = scaler.fit_transform(X_train)\n"
        "X_test = scaler.transform(X_test)\n"
        "clf = LogisticRegression(max_iter=1000, random_state=42)\n"
        "clf.fit(X_train, y_train)\n"
        "print('Accuracy:', clf.score(X_test, y_test))\n"
    )

    initial_error_log = (
        "Traceback (most recent call last):\n"
        "  File '<script>', line 1, in <module>\n"
        "  File '<script>', line 1, in <module>\n"
        "  File '<script>', line 1, in <module>\n"
        "KeyError: 'age_years'\n"
    )

    task_description = (
        "The dataset column is named 'age' but the script tries to use 'age_years', "
        "causing a KeyError when selecting features."
    )

    expected_fix_hint = (
        "Update the script to use the existing 'age' column instead of 'age_years'. "
        "Verify schema with check_schema before selecting columns."
    )

    return {
        "broken_script": broken_script,
        "dataframe": df,
        "initial_error_log": initial_error_log,
        "task_description": task_description,
        "expected_fix_hint": expected_fix_hint,
    }

