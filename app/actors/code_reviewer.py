import re


class CodeReviewer:
    """
    Simulated actor bot that reviews the agent's code edits 
    for Stage 2 and Stage 3 to detect data leakage and missing scaler.
    """

    def review(self, script: str, stage: int) -> str:
        script = script or ""
        
        has_scaler = "StandardScaler" in script
        # Check if X_scaled = fit_transform(X) happens before train_test_split
        # or if scaler is applied to the raw dataframe before splitting
        leakage_detected = False
        
        if has_scaler:
            if re.search(r"scaler\.fit_transform\(\s*X\s*\)", script):
                leakage_detected = True
            elif re.search(r"scaler\.fit\(\s*X\s*\)", script):
                leakage_detected = True

        if stage == 2:
            if not has_scaler:
                return (
                    "REJECT: Feature scaling is missing. MLPClassifier is "
                    "highly sensitive to unscaled features. Add StandardScaler "
                    "before fitting."
                )
            if leakage_detected:
                return (
                    "REJECT: You're fitting the scaler on the full dataset. "
                    "This causes data leakage. Fit only on X_train."
                )
            return (
                "LGTM: Scaler is correctly fitted on training data only. "
                "This looks good to merge."
            )

        if stage == 3:
            if leakage_detected or not has_scaler:
                return (
                    "REJECT: Critical data leakage detected. Scaler must be "
                    "fit on training data only. This invalidates your evaluation."
                )
            return (
                "LGTM: No data leakage detected. Evaluation pipeline "
                "looks correct."
            )
            
        return "I have no feedback for this stage."
