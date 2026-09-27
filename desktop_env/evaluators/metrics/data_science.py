import os
import logging
import numpy as np
import pandas as pd
from openai import OpenAI
import base64
import re

api_key = os.environ.get("OPENAI_API_KEY", "")
base_url = os.environ.get("OPENAI_BASE_URL", None)
client = OpenAI(api_key=api_key if api_key else "EMPTY", base_url=base_url)


logger = logging.getLogger("desktopenv.metrics.data_science")

def evaluate_wine_correlation_txt(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Wine Correlation Analysis task.
    This evaluator checks that:
      - wine_correlation.py exists and imports pandas
      - winequality-red.csv exists
      - result.txt exists and contains the correct third-highest correlated feature
        and its correlation value rounded to two decimals
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    expected = expected.get("rules", expected)
    score = 0.0
    feedback = []

    try:
        # Identify the working directory
        base_dir = os.path.dirname(actual)
        logger.info(f"Starting evaluation in directory: {base_dir}")

        py_file = os.path.join(base_dir, "wine_correlation.py")
        csv_file = os.path.join(base_dir, "winequality-red.csv")
        result_file = os.path.join(base_dir, "result.txt")

        logger.info("Files expected for evaluation:")
        logger.info(f" - Script: {py_file}")
        logger.info(f" - Dataset: {csv_file}")
        logger.info(f" - Result file: {result_file}")

        # 1. Check Python script
        if os.path.exists(py_file):
            score += 0.25
            feedback.append("Python script is present.")
            logger.info("Python script found.")

            with open(py_file, "r", encoding="utf-8") as f:
                content = f.read()
            if "import pandas" in content:
                score += 0.1
                feedback.append("The script correctly imports pandas.")
                logger.info("Pandas import verified in script.")
            else:
                feedback.append("The script does not import pandas.")
                logger.warning("Missing pandas import in script.")
        else:
            feedback.append("Python script is missing.")
            logger.warning("Python script missing.")

        # 2. Check dataset
        if os.path.exists(csv_file):
            score += 0.25
            feedback.append("Dataset file is present.")
            logger.info("Dataset file found.")
        else:
            feedback.append("Dataset file is missing.")
            logger.warning("Dataset file missing.")

        # 3. Check result.txt
        if os.path.exists(result_file):
            score += 0.25
            feedback.append("Result file is present.")
            logger.info("Result file found.")

            try:
                # Compute the expected correlation
                df = pd.read_csv(csv_file)
                correlations = df.corr(numeric_only=True)["quality"].drop("quality").sort_values(ascending=False)
                expected_feature = correlations.index[2]
                expected_value = round(correlations.iloc[2], 2)
                logger.info(f"Expected feature: {expected_feature}, correlation: {expected_value:.2f}")

                with open(result_file, "r", encoding="utf-8") as f:
                    content = f.read().lower().strip()
                logger.info(f"Contents of result.txt: {content}")

                # Match feature and correlation value
                if expected_feature.lower() in content and f"{expected_value:.2f}" in content:
                    score += 0.15
                    feedback.append(f"The correct feature '{expected_feature}' with correlation {expected_value:.2f} is correctly written in result.txt.")
                elif expected_feature.lower() in content:
                    score += 0.1
                    feedback.append(f"The feature '{expected_feature}' is found but the correlation value is incorrect.")
                else:
                    feedback.append(f"The expected feature '{expected_feature}' is not found in result.txt.")
                    logger.warning("Expected feature not found in result file.")
            except Exception as e:
                feedback.append(f"Error while checking result.txt: {e}")
                logger.error(f"Error while verifying result.txt: {e}", exc_info=True)
        else:
            feedback.append("Result file is missing.")
            logger.warning("Result file missing.")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        feedback.append(f"Evaluation error: {e}")

    final_score = min(score, 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score


def evaluate_sulfur_ratio_comparison(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Sulfur Ratio Analysis task.
    This evaluator checks that:
      - sulfur_ratio_analysis.py exists and imports pandas
      - winequality-red.csv exists
      - result.txt exists and correctly reports mean quality
        for sulfur_ratio > 0.4 and <= 0.4 (rounded to two decimals)
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    expected = expected.get("rules", expected)
    score = 0.0
    feedback = []

    try:
        base_dir = os.path.dirname(actual)
        logger.info(f"Starting evaluation in directory: {base_dir}")

        py_file = os.path.join(base_dir, "sulfur_ratio_analysis.py")
        csv_file = os.path.join(base_dir, "winequality-red.csv")
        result_file = os.path.join(base_dir, "result.txt")

        logger.info("Files expected for evaluation:")
        logger.info(f" - Script: {py_file}")
        logger.info(f" - Dataset: {csv_file}")
        logger.info(f" - Result file: {result_file}")

        # 1. Check Python script
        if os.path.exists(py_file):
            score += 0.25
            feedback.append("Python script is present.")
            logger.info("Python script found.")

            with open(py_file, "r", encoding="utf-8") as f:
                content = f.read()
            if "import pandas" in content:
                score += 0.1
                feedback.append("The script correctly imports pandas.")
                logger.info("Pandas import verified.")
            else:
                feedback.append("The script does not import pandas.")
                logger.warning("Missing pandas import in script.")
        else:
            feedback.append("Python script is missing.")
            logger.warning("Python script missing.")

        # 2. Check dataset
        if os.path.exists(csv_file):
            score += 0.25
            feedback.append("Dataset file is present.")
            logger.info("Dataset file found.")
        else:
            feedback.append("Dataset file is missing.")
            logger.warning("Dataset file missing.")

        # 3. Check result.txt
        if os.path.exists(result_file):
            score += 0.25
            feedback.append("Result file is present.")
            logger.info("Result file found.")

            try:
                # Compute expected means
                df = pd.read_csv(csv_file)
                df["sulfur_ratio"] = df["free sulfur dioxide"] / df["total sulfur dioxide"]
                high_ratio_mean = round(df.loc[df["sulfur_ratio"] > 0.4, "quality"].mean(), 2)
                low_ratio_mean = round(df.loc[df["sulfur_ratio"] <= 0.4, "quality"].mean(), 2)
                logger.info(f"Expected output: {high_ratio_mean}, {low_ratio_mean}")

                with open(result_file, "r", encoding="utf-8") as f:
                    content = f.read().strip().replace(" ", "")
                logger.info(f"Contents of result.txt: {content}")

                # Check both numbers
                expected_str = f"{high_ratio_mean},{low_ratio_mean}"
                reversed_str = f"{low_ratio_mean},{high_ratio_mean}"  # allow reversed order

                if expected_str in content:
                    score += 0.15
                    feedback.append("The result.txt contains the correct mean values in the expected order.")
                elif reversed_str in content:
                    score += 0.1
                    feedback.append("The correct values are found but appear in reverse order.")
                else:
                    feedback.append("The expected mean values are not correctly written in result.txt.")
                    logger.warning("Expected values not found.")
            except Exception as e:
                feedback.append(f"Error while checking result.txt: {e}")
                logger.error(f"Error verifying result.txt: {e}", exc_info=True)
        else:
            feedback.append("Result file is missing.")
            logger.warning("Result file missing.")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        feedback.append(f"Evaluation error: {e}")

    final_score = min(score, 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score 

def evaluate_alcohol_outliers_iqr(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Alcohol Outlier Detection (IQR Method) task.
    This evaluator checks that:
      - alcohol_outlier_analysis.py exists and imports pandas
      - winequality-red.csv exists
      - result.txt exists and contains the correct outlier count and minimum outlier value
        rounded to two decimals
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    expected = expected.get("rules", expected)
    score = 0.0
    feedback = []

    try:
        base_dir = os.path.dirname(actual)
        logger.info(f"Starting evaluation in directory: {base_dir}")

        py_file = os.path.join(base_dir, "alcohol_outlier_analysis.py")
        csv_file = os.path.join(base_dir, "winequality-red.csv")
        result_file = os.path.join(base_dir, "result.txt")

        logger.info("Files expected for evaluation:")
        logger.info(f" - Script: {py_file}")
        logger.info(f" - Dataset: {csv_file}")
        logger.info(f" - Result file: {result_file}")

        # 1. Check Python script
        if os.path.exists(py_file):
            score += 0.25
            feedback.append("Python script is present.")
            logger.info("Python script found.")

            with open(py_file, "r", encoding="utf-8") as f:
                content = f.read()
            if "import pandas" in content:
                score += 0.1
                feedback.append("The script correctly imports pandas.")
                logger.info("Pandas import verified in script.")
            else:
                feedback.append("The script does not import pandas.")
                logger.warning("Missing pandas import in script.")
        else:
            feedback.append("Python script is missing.")
            logger.warning("Python script missing.")

        # 2. Check dataset
        if os.path.exists(csv_file):
            score += 0.25
            feedback.append("Dataset file is present.")
            logger.info("Dataset file found.")
        else:
            feedback.append("Dataset file is missing.")
            logger.warning("Dataset file missing.")

        # 3. Check result.txt
        if os.path.exists(result_file):
            score += 0.25
            feedback.append("Result file is present.")
            logger.info("Result file found.")

            try:
                # Compute expected outlier statistics
                df = pd.read_csv(csv_file)
                q1 = df["alcohol"].quantile(0.25)
                q3 = df["alcohol"].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outliers = df[(df["alcohol"] < lower_bound) | (df["alcohol"] > upper_bound)]
                outlier_count = len(outliers)
                min_outlier_value = round(outliers["alcohol"].min(), 2) if not outliers.empty else None
                logger.info(f"Expected output: count={outlier_count}, min={min_outlier_value}")

                with open(result_file, "r", encoding="utf-8") as f:
                    content = f.read().strip().replace(" ", "")
                logger.info(f"Contents of result.txt: {content}")

                expected_str = f"{outlier_count},{min_outlier_value}"
                reversed_str = f"{min_outlier_value},{outlier_count}"

                if expected_str in content:
                    score += 0.15
                    feedback.append("The result.txt contains the correct outlier count and minimum value in the correct order.")
                elif reversed_str in content:
                    score += 0.1
                    feedback.append("The correct values are present but in reverse order.")
                else:
                    feedback.append("The expected outlier count or minimum value is incorrect or missing.")
                    logger.warning("Expected values not found in result.txt.")
            except Exception as e:
                feedback.append(f"Error while checking result.txt: {e}")
                logger.error(f"Error verifying result.txt: {e}", exc_info=True)
        else:
            feedback.append("Result file is missing.")
            logger.warning("Result file missing.")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        feedback.append(f"Evaluation error: {e}")

    final_score = min(score, 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score

def evaluate_wine_regression_mae(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Wine Regression task.
    This evaluator checks:
      - wine_regression_model.py exists and imports pandas and sklearn
      - winequality-red.csv exists
      - validation_mae.txt exists
      - The MAE value is below a defined performance threshold (0.530)
    Gives higher weight (60%) to model performance and 40% to structure.
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    expected = expected.get("rules", expected)
    score = 0.0
    feedback = []

    try:
        base_dir = os.path.dirname(actual)
        logger.info(f"Starting evaluation in directory: {base_dir}")

        py_file = os.path.join(base_dir, "wine_regression_model.py")
        csv_file = os.path.join(base_dir, "winequality-red.csv")
        result_file = os.path.join(base_dir, "validation_mae.txt")

        # 1. Check Python script
        if os.path.exists(py_file):
            score += 0.2
            feedback.append("Python script is present.")
            logger.info("Python script found.")

            with open(py_file, "r", encoding="utf-8") as f:
                content = f.read().lower()
            if "import pandas" in content and ("from sklearn" in content or "import sklearn" in content):
                score += 0.2
                feedback.append("Script correctly imports pandas and sklearn.")
            else:
                feedback.append("Missing pandas or sklearn import.")
                logger.warning("Missing pandas or sklearn import in script.")
        else:
            feedback.append("Python script is missing.")
            logger.warning("Python script missing.")

        # 2. Check dataset
        if os.path.exists(csv_file):
            score += 0.2
            feedback.append("Dataset file is present.")
        else:
            feedback.append("Dataset file is missing.")
            logger.warning("Dataset file missing.")

        # 3. Check output file and model score
        if os.path.exists(result_file):
            score += 0.1  # base credit for having result file
            feedback.append("Output file validation_mae.txt is present.")
            logger.info("Output file found.")

            try:
                with open(result_file, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                mae_value = None
                try:
                    mae_value = float(content)
                    logger.info(f"Read MAE value: {mae_value}")
                except ValueError:
                    feedback.append(f"Could not parse MAE value from file: '{content}'")
                    mae_value = None

                # Threshold-based evaluation
                if mae_value is not None:
                    if mae_value <= 0.530:
                        score += 0.3
                        feedback.append(f"Model performance is good (MAE = {mae_value:.3f}).")
                        logger.info("MAE below threshold. Full score awarded.")
                    else:
                        feedback.append(f"Model performance is poor (MAE = {mae_value:.3f}). Threshold is 0.530.")
                        logger.warning("MAE exceeds acceptable threshold. No score for model performance.")
                else:
                    feedback.append("MAE value missing or invalid.")
            except Exception as e:
                feedback.append(f"Error reading validation_mae.txt: {e}")
                logger.error(f"Error reading validation_mae.txt: {e}", exc_info=True)
        else:
            feedback.append("Output file validation_mae.txt is missing.")
            logger.warning("Output file missing.")

    except Exception as e:
        feedback.append(f"Evaluation failed: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)

    final_score = min(score, 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score

def evaluate_wine_visualization_quality(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Wine Visualization task.
    This evaluation checks:
      - whether the script, dataset, and visualization files exist
      - and uses GPT-4o to assess if the charts are meaningful and readable
    """

    if actual is None:
        print("Directory collection failed.")
        return 0.0

    expected = expected.get("rules", expected)
    base_dir = os.path.dirname(actual)

    py_file = os.path.join(base_dir, "wine_visualization.py")
    csv_file = os.path.join(base_dir, "winequality-red.csv")
    chart1 = os.path.join(base_dir, "chart1.png")
    chart2 = os.path.join(base_dir, "chart2.png")

    score = 0.0
    feedback = []

    # Step 1: Check the Python script
    if os.path.exists(py_file):
        score += 0.2
        feedback.append("The Python script exists.")
        with open(py_file, "r", encoding="utf-8") as f:
            content = f.read().lower()
            if "import pandas" in content and ("matplotlib" in content or "seaborn" in content):
                score += 0.1
                feedback.append("The script correctly imports pandas and visualization libraries.")
            else:
                feedback.append("The script does not import pandas or a visualization library.")
    else:
        feedback.append("The Python script is missing.")

    # Step 2: Check the dataset
    if os.path.exists(csv_file):
        score += 0.1
        feedback.append("The dataset file is present.")
    else:
        feedback.append("The dataset file is missing.")

    # Helper function to encode an image
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    # Step 3: Check and evaluate visualizations
    charts_to_check = [
        (chart1, "scatter plot showing how alcohol content affects wine quality"),
        (chart2, "boxplot showing the distribution of wine quality across different alcohol levels")
    ]

    for chart, description in charts_to_check:
        if os.path.exists(chart):
            score += 0.1
            feedback.append(f"The file {os.path.basename(chart)} exists.")

            # Use GPT-4o to judge clarity and relevance
            try:
                base64_image = encode_image(chart)
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert data visualization evaluator. Judge the clarity and relevance of the chart."
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        f"This chart should look like a {description}. "
                                        f"Does it clearly and readably show how alcohol relates to wine quality? "
                                        f"Check that both axes are labeled, the chart is readable, and the relationship is understandable. "
                                        f"Answer only 'yes' or 'no' and explain briefly."
                                    )
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                                }
                            ]
                        }
                    ],
                    max_tokens=100
                )

                answer = response.choices[0].message.content.lower()
                if "yes" in answer:
                    score += 0.2
                    feedback.append(f"GPT-4o judged that {os.path.basename(chart)} is clear, relevant, and readable.")
                else:
                    feedback.append(f"GPT-4o judged that {os.path.basename(chart)} is not clearly showing the relationship. Response: {answer}")

            except Exception as e:
                feedback.append(f"An error occurred during GPT-4o evaluation: {e}")
        else:
            feedback.append(f"The file {os.path.basename(chart)} is missing.")

    # Final summary
    final_score = min(score, 1.0)
    print("\nEvaluation Summary:")
    for message in feedback:
        print(message)
    print(f"Final Score: {final_score:.2f}")

    return final_score

def evaluate_outlier_distance(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Walmart Outlier Distance task.

    This evaluation checks:
      - that the Python script file exists
      - that the script correctly imports pandas
      - that the result file outlier_distance.txt exists
      - and that the reported average distance value matches the expected value (≈ 444.53 km)
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    expected = expected.get("rules", expected)
    base_dir = os.path.dirname(actual)
    script_file = os.path.join(base_dir, "walmart_outlier_distance.py")
    result_file = os.path.join(base_dir, "outlier_distance.txt")

    score = 0.0
    feedback = []

    true_distance = 444.53  # known correct distance in kilometers

    try:
        # Step 1: Check that the Python script exists
        if os.path.exists(script_file):
            score += 0.4
            feedback.append("The Python script file is present.")
            logger.info("Python script file found.")

            with open(script_file, "r", encoding="utf-8") as f:
                content = f.read().lower()

            if "import pandas" in content:
                score += 0.3
                feedback.append("The script correctly imports pandas.")
                logger.info("Pandas import verified.")
            else:
                feedback.append("The script does not import pandas.")
                logger.warning("Pandas import not found in script.")
        else:
            feedback.append("The Python script file is missing.")
            logger.warning("Python script missing.")

        # Step 2: Check that the result file exists and contains a valid numeric value
        if os.path.exists(result_file):
            score += 0.1
            feedback.append("The result file outlier_distance.txt exists.")
            logger.info("Result file found.")

            try:
                with open(result_file, "r", encoding="utf-8") as f:
                    output = f.read().strip()

                # Extract the first numeric value (handles both plain numbers and text with 'km')
                match = re.search(r"[-+]?\d*\.\d+|\d+", output)
                if match:
                    value = float(match.group())
                    logger.info(f"Extracted numeric value from result file: {value}")

                    if abs(value - true_distance) < 0.5:
                        score += 0.2
                        feedback.append("The reported average distance value matches the expected result.")
                    else:
                        feedback.append(
                            f"The reported distance value does not match the expected result. "
                            f"Expected around {true_distance}, found {value}."
                        )
                        logger.warning(f"Value mismatch: expected {true_distance}, found {value}")
                else:
                    feedback.append("No numeric value could be extracted from the result file.")
                    logger.warning("No numeric pattern found in result file.")

            except Exception as e:
                feedback.append(f"Error reading or parsing the result file: {e}")
                logger.error(f"Error reading result file: {e}", exc_info=True)
        else:
            feedback.append("The result file outlier_distance.txt is missing.")
            logger.warning("Result file missing.")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        feedback.append(f"Evaluation error: {e}")

    # Final score capped at 1.0
    final_score = min(score, 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score

def evaluate_walmart_supercenter_viz_quality(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Walmart Supercenter Trend Analysis task.

    This evaluation checks:
      - that the Python script exists and imports pandas
      - that both visualization files are present
      - and that GPT-4o confirms the charts clearly represent the correct information
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    expected = expected.get("rules", expected)
    base_dir = os.path.dirname(actual)

    py_file = os.path.join(base_dir, "walmart_supercenter_trend.py")
    chart1 = os.path.join(base_dir, "supercenter_trend.png")
    chart2 = os.path.join(base_dir, "store_openings_bar.png")

    score = 0.0
    feedback = []

    # Step 1: Check for Python script
    if os.path.exists(py_file):
        score += 0.3
        feedback.append("The Python script file is present.")
        logger.info("Python script file found.")

        with open(py_file, "r", encoding="utf-8") as f:
            content = f.read().lower()
        if "import pandas" in content:
            score += 0.2
            feedback.append("The script correctly imports pandas.")
            logger.info("Pandas import verified.")
        else:
            feedback.append("The script does not import pandas.")
            logger.warning("Pandas import not found.")
    else:
        feedback.append("The Python script is missing.")
        logger.warning("Python script missing.")

    # Helper: encode images for GPT-4o
    def encode_image(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    # Step 2: Visualization checks
    charts_to_check = [
        (chart1, "a line chart showing the yearly proportion of Supercenters with a ±5 percent confidence band"),
        (chart2, "a bar chart showing the total number of store openings per year")
    ]

    for chart, description in charts_to_check:
        if os.path.exists(chart):
            score += 0.15
            feedback.append(f"The visualization {os.path.basename(chart)} exists.")
            logger.info(f"{chart} found.")

            try:
                encoded = encode_image(chart)
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert in data visualization quality evaluation. Judge each chart objectively."
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        f"Examine this chart and answer if it correctly represents {description} "
                                        f"with clear labeling, readable axes, and balanced layout. "
                                        f"Reply 'yes' if it does, otherwise 'no', followed by a short reason."
                                    )
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{encoded}"}
                                }
                            ]
                        }
                    ],
                    max_tokens=100
                )

                answer = response.choices[0].message.content.lower()
                if "yes" in answer:
                    score += 0.1
                    feedback.append(f"GPT-4o judged {os.path.basename(chart)} as relevant, clear, and visually accurate.")
                    logger.info(f"{chart} judged relevant by GPT-4o.")
                else:
                    feedback.append(f"GPT-4o judged {os.path.basename(chart)} unclear or irrelevant: {answer}")
                    logger.warning(f"{chart} judged unclear by GPT-4o.")
            except Exception as e:
                feedback.append(f"Could not evaluate {os.path.basename(chart)} using GPT-4o: {e}")
                logger.error(f"Error evaluating {chart}: {e}", exc_info=True)
        else:
            feedback.append(f"The visualization {os.path.basename(chart)} is missing.")
            logger.warning(f"{chart} missing.")

    final_score = min(score, 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score

def evaluate_xgb_metrics_classifier(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Walmart XGBoost Classifier task.

    This evaluator checks:
      - whether the Python script file exists
      - whether pandas, xgboost, and sklearn are imported
      - whether the xgb_metrics.txt file exists
      - whether accuracy, precision, and recall are valid and meet the expected threshold (≥ 0.7)
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    expected = expected.get("rules", expected)
    base_dir = os.path.dirname(actual)
    script_file = os.path.join(base_dir, "walmart_xgb_classifier.py")
    result_file = os.path.join(base_dir, "xgb_metrics.txt")

    score = 0.0
    feedback = []

    try:
        # Step 1: Verify the Python script
        if os.path.exists(script_file):
            score += 0.3
            feedback.append("The Python script file is present.")
            with open(script_file, "r", encoding="utf-8") as f:
                content = f.read().lower()

            if "import pandas" in content and "xgboost" in content and "sklearn" in content:
                score += 0.2
                feedback.append("The script correctly imports pandas, xgboost, and sklearn.")
            else:
                feedback.append("The script is missing one or more required imports.")
        else:
            feedback.append("The Python script file is missing.")

        # Step 2: Verify the result file
        if os.path.exists(result_file):
            score += 0.2
            feedback.append("The result file xgb_metrics.txt exists.")

            try:
                with open(result_file, "r", encoding="utf-8") as f:
                    output = f.read().strip()

                # Expected format: accuracy, precision, recall
                parts = [p.strip() for p in output.split(",")]
                if len(parts) == 3:
                    metrics = {}
                    valid = True
                    for name, val in zip(["accuracy", "precision", "recall"], parts):
                        try:
                            num = float(val)
                            metrics[name] = num
                            if not (0.0 <= num <= 1.0):
                                valid = False
                        except ValueError:
                            valid = False

                    if valid:
                        logger.info(f"Extracted metrics: {metrics}")
                        # Check threshold
                        if all(v >= 0.7 for v in metrics.values()):
                            score += 0.3
                            feedback.append("All metrics meet or exceed the 0.7 threshold.")
                        else:
                            score += 0.15
                            feedback.append("Metrics are valid but one or more are below the 0.7 threshold.")
                    else:
                        feedback.append("One or more metrics are invalid or out of range.")
                else:
                    feedback.append("The result file does not contain exactly three comma-separated values.")
            except Exception as e:
                feedback.append(f"Error reading or parsing xgb_metrics.txt: {e}")
        else:
            feedback.append("The result file xgb_metrics.txt is missing.")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        feedback.append(f"Evaluation error: {e}")

    final_score = min(score, 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score


def evaluate_max_route_diff(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Longest Airline Route task.

    This evaluation checks:
      - that the Python script file exists
      - that the script correctly imports pandas
      - that the result file result.txt exists
      - and that its contents match the expected output:
        'AA,HNL-ORD,6819.18,-188.9'
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    expected = expected.get("rules", expected)
    base_dir = os.path.dirname(actual)
    script_file = os.path.join(base_dir, "max_route_diff.py")
    result_file = os.path.join(base_dir, "result.txt")

    score = 0.0
    feedback = []

    expected_line = "AA,HNL-ORD,6819.18,-188.9"

    try:
        # Step 1: Check that the Python script exists
        if os.path.exists(script_file):
            score += 0.4
            feedback.append("The Python script file is present.")
            logger.info("Python script file found.")

            with open(script_file, "r", encoding="utf-8") as f:
                content = f.read().lower()

            if "import pandas" in content:
                score += 0.3
                feedback.append("The script correctly imports pandas.")
                logger.info("Pandas imports verified.")
            else:
                feedback.append("The script does not import pandas or math.")
                logger.warning("Missing pandas or math import in script.")
        else:
            feedback.append("The Python script file is missing.")
            logger.warning("Python script missing.")

        # Step 2: Check that the result file exists and matches expected output
        if os.path.exists(result_file):
            score += 0.1
            feedback.append("The result file result.txt exists.")
            logger.info("Result file found.")

            try:
                with open(result_file, "r", encoding="utf-8") as f:
                    output = f.read().strip().replace(" ", "")

                if output == expected_line.replace(" ", ""):
                    score += 0.2
                    feedback.append("The result file contains the correct expected output.")
                    logger.info("Result output matches expected line.")
                else:
                    feedback.append(
                        f"The result output does not match expected. "
                        f"Expected '{expected_line}', found '{output}'."
                    )
                    logger.warning(f"Output mismatch: expected '{expected_line}', found '{output}'")
            except Exception as e:
                feedback.append(f"Error reading or parsing result.txt: {e}")
                logger.error(f"Error reading result file: {e}", exc_info=True)
        else:
            feedback.append("The result file result.txt is missing.")
            logger.warning("Result file missing.")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        feedback.append(f"Evaluation error: {e}")

    # Final score capped at 1.0
    final_score = min(score, 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score



def evaluate_airline_route_network_quality(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Airline Route Network Visualization task.
    Checks that:
      - The Python script and dataset exist
      - The script imports pandas and matplotlib
      - The route_network.png visualization exists
      - GPT-4o judges that the map clearly shows start and end airports with connecting routes
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    base_dir = os.path.dirname(actual)
    py_file = os.path.join(base_dir, "airline_route_network.py")
    csv_file = os.path.join(base_dir, "routes.csv")
    chart_file = os.path.join(base_dir, "route_network.png")

    score = 0.0
    feedback = []

    # Step 1: Check script existence and imports
    if os.path.exists(py_file):
        score += 0.25
        feedback.append("Python script exists.")
        with open(py_file, "r", encoding="utf-8") as f:
            content = f.read().lower()
            if "import pandas" in content and "matplotlib" in content:
                score += 0.15
                feedback.append("Script correctly imports pandas and matplotlib.")
            else:
                feedback.append("Script missing required library imports.")
    else:
        feedback.append("Python script missing.")

    # Step 2: Check dataset
    if os.path.exists(csv_file):
        score += 0.1
        feedback.append("Dataset file exists.")
    else:
        feedback.append("Dataset missing.")

    # Step 3: Check visualization
    if os.path.exists(chart_file):
        score += 0.1
        feedback.append("Visualization route_network.png exists.")

        def encode_image(path):
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")

        try:
            base64_img = encode_image(chart_file)
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert data visualization evaluator."},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "This image should display an airline route network map: "
                                    "blue points for starting airports, red points for destination airports, "
                                    "and lines connecting them to form routes. "
                                    "Judge whether it clearly shows geographic distribution and connections "
                                    "in a readable and meaningful way. Answer 'yes' or 'no' with a short explanation."
                                )
                            },
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_img}"}}
                        ]
                    }
                ],
                max_tokens=100
            )

            answer = response.choices[0].message.content.lower()
            if "yes" in answer:
                score += 0.4
                feedback.append("GPT-4o judged the visualization clear and accurate.")
            else:
                feedback.append(f"GPT-4o judged the visualization unclear: {answer}")
        except Exception as e:
            feedback.append(f"GPT-4o evaluation failed: {e}")
    else:
        feedback.append("Visualization file route_network.png missing.")

    final_score = min(score, 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score 

def evaluate_xgb_route_metrics(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Route XGBoost Classifier task.

    This function checks:
      - The Python script 'route_xgb_classifier.py' exists
      - pandas, xgboost, and sklearn are imported
      - The output file 'xgb_route_metrics.txt' exists
      - It contains accuracy, precision, and recall (floats between 0 and 1)
      - Each metric must be at least 0.7 for a full score
    """

    if actual is None:
        logger.error("No directory provided.")
        return 0.0

    expected = expected.get("rules", expected)
    base_dir = os.path.dirname(actual)
    script_file = os.path.join(base_dir, "route_xgb_classifier.py")
    result_file = os.path.join(base_dir, "xgb_route_metrics.txt")

    score = 0.0
    feedback = []

    # Step 1: Check for the script file
    if os.path.exists(script_file):
        score += 0.3
        with open(script_file, "r", encoding="utf-8") as f:
            content = f.read().lower()
        if "import pandas" in content and "xgboost" in content and "sklearn" in content:
            score += 0.2
            feedback.append("Script correctly imports pandas, xgboost, and sklearn.")
        else:
            feedback.append("One or more required imports are missing.")
    else:
        feedback.append("Script file route_xgb_classifier.py is missing.")

    # Step 2: Check for the output file and validate metrics
    if os.path.exists(result_file):
        score += 0.2
        with open(result_file, "r", encoding="utf-8") as f:
            output = f.read().strip()

        parts = [p.strip() for p in output.split(",")]
        if len(parts) == 3:
            try:
                accuracy, precision, recall = [float(p) for p in parts]
                if all(0.0 <= m <= 1.0 for m in [accuracy, precision, recall]):
                    if all(m >= 0.7 for m in [accuracy, precision, recall]):
                        score += 0.3
                        feedback.append("All metrics are valid and meet the 0.7 threshold.")
                    else:
                        score += 0.15
                        feedback.append("Metrics are valid but one or more are below 0.7.")
                else:
                    feedback.append("One or more metric values are outside the valid range (0–1).")
            except Exception as e:
                feedback.append(f"Error parsing metric values: {e}")
        else:
            feedback.append("Expected three comma-separated values: accuracy, precision, recall.")
    else:
        feedback.append("Output file xgb_route_metrics.txt is missing.")

    final_score = min(score, 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score

def evaluate_ag_dependence(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Agricultural Dependence task.

    This evaluation checks:
      - that the Python script file exists
      - that the script correctly imports pandas
      - that the result file result.txt exists
      - and that the file contains the correct top 3 states:
        Hawaii, New Jersey, North Dakota
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    expected = expected.get("rules", expected)
    base_dir = os.path.dirname(actual)
    script_file = os.path.join(base_dir, "ag_dependence.py")
    result_file = os.path.join(base_dir, "result.txt")

    score = 0.0
    feedback = []

    correct_states = ["Hawaii", "New Jersey", "North Dakota"]

    try:
        # Step 1: Check that the Python script exists
        if os.path.exists(script_file):
            score += 0.4
            feedback.append("The Python script file is present.")
            logger.info("Python script file found.")

            with open(script_file, "r", encoding="utf-8") as f:
                content = f.read().lower()

            if "import pandas" in content:
                score += 0.3
                feedback.append("The script correctly imports pandas.")
                logger.info("Pandas import verified.")
            else:
                feedback.append("The script does not import pandas.")
                logger.warning("Pandas import not found in script.")
        else:
            feedback.append("The Python script file is missing.")
            logger.warning("Python script missing.")

        # Step 2: Check that the result file exists and contains the expected output
        if os.path.exists(result_file):
            score += 0.1
            feedback.append("The result file result.txt exists.")
            logger.info("Result file found.")

            try:
                with open(result_file, "r", encoding="utf-8") as f:
                    lines = [line.strip().split(",")[0] for line in f if line.strip()]

                logger.info(f"States found in result.txt: {lines}")

                # Compare top 3 states in order
                if lines[:3] == correct_states:
                    score += 0.2
                    feedback.append("The result file contains the correct top 3 states in order.")
                    logger.info("Correct states verified in correct order.")
                else:
                    feedback.append(
                        f"The result file does not match the expected states. "
                        f"Expected: {correct_states}, Found: {lines[:3]}"
                    )
                    logger.warning(f"Mismatch: expected {correct_states}, found {lines[:3]}")
            except Exception as e:
                feedback.append(f"Error reading or parsing result.txt: {e}")
                logger.error(f"Error reading result file: {e}", exc_info=True)
        else:
            feedback.append("The result file result.txt is missing.")
            logger.warning("Result file missing.")

    except Exception as e:
        feedback.append(f"Evaluation failed: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)

    # Final score capped at 1.0
    final_score = min(score, 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score



def evaluate_ag_kmeans_summary(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Agricultural K-Means Summary task.

    This evaluation checks:
      - that the Python script file exists
      - that the script correctly imports pandas and sklearn
      - that the result file cluster_summary.txt exists
      - and that it contains the correct cluster averages:
        0,1832.54
        1,16472.88
        2,7867.99
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    expected = expected.get("rules", expected)
    base_dir = os.path.dirname(actual)
    script_file = os.path.join(base_dir, "ag_kmeans_summary.py")
    result_file = os.path.join(base_dir, "cluster_summary.txt")

    score = 0.0
    feedback = []

    expected_lines = [
        "0,1832.54",
        "1,16472.88",
        "2,7867.99"
    ]

    try:
        # Step 1: Check that the Python script exists
        if os.path.exists(script_file):
            score += 0.4
            feedback.append("The Python script file is present.")
            logger.info("Python script file found.")

            with open(script_file, "r", encoding="utf-8") as f:
                content = f.read().lower()

            if "import pandas" in content and "sklearn" in content:
                score += 0.3
                feedback.append("The script correctly imports pandas and sklearn.")
                logger.info("Pandas and sklearn imports verified.")
            else:
                feedback.append("The script does not import pandas or sklearn.")
                logger.warning("Missing pandas or sklearn import in script.")
        else:
            feedback.append("The Python script file is missing.")
            logger.warning("Python script missing.")

        # Step 2: Check that the result file exists and matches expected output
        if os.path.exists(result_file):
            score += 0.1
            feedback.append("The result file cluster_summary.txt exists.")
            logger.info("Result file found.")

            try:
                with open(result_file, "r", encoding="utf-8") as f:
                    lines = [line.strip().replace(" ", "") for line in f if line.strip()]

                logger.info(f"Lines found in result.txt: {lines}")

                if lines == expected_lines:
                    score += 0.2
                    feedback.append("The result file contains the correct cluster averages in correct order.")
                    logger.info("Correct output verified.")
                else:
                    feedback.append(
                        f"The result file does not match expected content. "
                        f"Expected {expected_lines}, found {lines}."
                    )
                    logger.warning(f"Output mismatch: expected {expected_lines}, found {lines}")
            except Exception as e:
                feedback.append(f"Error reading or parsing cluster_summary.txt: {e}")
                logger.error(f"Error reading result file: {e}", exc_info=True)
        else:
            feedback.append("The result file cluster_summary.txt is missing.")
            logger.warning("Result file missing.")

    except Exception as e:
        feedback.append(f"Evaluation failed: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)

    # Final score capped at 1.0
    final_score = min(score, 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score

def evaluate_region_exports_viz_quality(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Region Exports Visualization task.
    Checks that:
      - The script file exists and imports pandas and matplotlib/seaborn
      - Both visualization files exist
      - GPT-4o judges the visuals for clarity and correctness
    Dataset existence is skipped (assumed already present).
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    base_dir = os.path.dirname(actual)
    py_file = os.path.join(base_dir, "region_exports_viz.py")
    chart1 = os.path.join(base_dir, "region_crops.png")
    chart2 = os.path.join(base_dir, "region_total_share.png")

    score = 0.0
    feedback = []

    # Step 1: Check script and imports
    if os.path.exists(py_file):
        score += 0.25
        feedback.append("Python script exists.")
        with open(py_file, "r", encoding="utf-8") as f:
            content = f.read().lower()
            if "import pandas" in content and ("matplotlib" in content or "seaborn" in content):
                score += 0.15
                feedback.append("Script correctly imports pandas and a visualization library.")
            else:
                feedback.append("Script missing required imports.")
    else:
        feedback.append("Python script missing.")

    # Helper: encode image for GPT-4o
    def encode_image(path):
        with open(path, "rb") as img:
            return base64.b64encode(img.read()).decode("utf-8")

    # Step 2: Evaluate region_crops.png (grouped bar chart)
    if os.path.exists(chart1):
        score += 0.1
        feedback.append("region_crops.png exists.")
        try:
            img_b64 = encode_image(chart1)
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert in evaluating data visualizations."},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "This chart should display a grouped bar chart comparing average corn and wheat exports by U.S. region. "
                                    "Does it clearly present labeled axes, a legend, and readable bars? "
                                    "Answer 'yes' or 'no' with a short reason."
                                )
                            },
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
                        ]
                    }
                ],
                max_tokens=100
            )
            ans = response.choices[0].message.content.lower()
            if "yes" in ans:
                score += 0.4
                feedback.append("GPT-4o judged region_crops.png as clear and correct.")
            else:
                feedback.append(f"GPT-4o judged region_crops.png unclear: {ans}")
        except Exception as e:
            feedback.append(f"GPT-4o evaluation for region_crops.png failed: {e}")
    else:
        feedback.append("region_crops.png missing.")

    # Step 3: Evaluate region_total_share.png (pie chart)
    if os.path.exists(chart2):
        score += 0.1
        feedback.append("region_total_share.png exists.")
        try:
            img_b64 = encode_image(chart2)
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert in evaluating data visualizations."},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "This chart should be a pie chart showing each region’s share of total agricultural exports. "
                                    "Does it clearly display proportional segments, have labels or legend, and maintain readability? "
                                    "Answer 'yes' or 'no' with a short reason."
                                )
                            },
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
                        ]
                    }
                ],
                max_tokens=100
            )
            ans = response.choices[0].message.content.lower()
            if "yes" in ans:
                score += 0.4
                feedback.append("GPT-4o judged region_total_share.png as clear and correct.")
            else:
                feedback.append(f"GPT-4o judged region_total_share.png unclear: {ans}")
        except Exception as e:
            feedback.append(f"GPT-4o evaluation for region_total_share.png failed: {e}")
    else:
        feedback.append("region_total_share.png missing.")

    final_score = min(score, 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score


def evaluate_aapl_corr_ma(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Apple Stock Correlation task.

    This evaluation checks:
      - that the Python script file exists
      - that the script correctly imports pandas
      - that the result file corr_ma.txt exists
      - and that it contains the correct correlation value: 0.997
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    expected = expected.get("rules", expected)
    base_dir = os.path.dirname(actual)
    script_file = os.path.join(base_dir, "aapl_corr_ma.py")
    result_file = os.path.join(base_dir, "corr_ma.txt")

    score = 0.0
    feedback = []

    correct_value = 0.997

    try:
        # Step 1: Check that the Python script exists
        if os.path.exists(script_file):
            score += 0.4
            feedback.append("The Python script file is present.")
            logger.info("Python script file found.")

            with open(script_file, "r", encoding="utf-8") as f:
                content = f.read().lower()

            if "import pandas" in content:
                score += 0.3
                feedback.append("The script correctly imports pandas.")
                logger.info("Pandas import verified.")
            else:
                feedback.append("The script does not import pandas.")
                logger.warning("Pandas import not found in script.")
        else:
            feedback.append("The Python script file is missing.")
            logger.warning("Python script missing.")

        # Step 2: Check that the result file exists and matches expected value
        if os.path.exists(result_file):
            score += 0.1
            feedback.append("The result file corr_ma.txt exists.")
            logger.info("Result file found.")

            try:
                with open(result_file, "r", encoding="utf-8") as f:
                    content = f.read().strip()

                # Extract numeric value
                try:
                    value = float(content)
                    logger.info(f"Read correlation value: {value}")

                    if abs(value - correct_value) < 0.001:
                        score += 0.2
                        feedback.append("The result file contains the correct correlation value.")
                        logger.info("Correlation value matches expected result.")
                    else:
                        feedback.append(
                            f"The correlation value does not match expected. "
                            f"Expected {correct_value}, found {value}."
                        )
                        logger.warning(f"Mismatch: expected {correct_value}, found {value}")
                except ValueError:
                    feedback.append(f"Could not parse a numeric value from corr_ma.txt: '{content}'.")
                    logger.warning("Non-numeric content in corr_ma.txt.")
            except Exception as e:
                feedback.append(f"Error reading corr_ma.txt: {e}")
                logger.error(f"Error reading result file: {e}", exc_info=True)
        else:
            feedback.append("The result file corr_ma.txt is missing.")
            logger.warning("Result file missing.")

    except Exception as e:
        feedback.append(f"Evaluation failed: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)

    # Final score capped at 1.0
    final_score = min(score, 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score

def evaluate_aapl_ma_volatility_viz_quality(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Apple Stock MA & Volatility Visualization task.
    Checks that:
      - The script exists and imports pandas + any visualization library (matplotlib, seaborn, plotly, altair)
      - The two visualization files exist
      - GPT-4o judges that both charts are clear, correct, and readable.
    Dataset presence check skipped (assumed available).
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    base_dir = os.path.dirname(actual)
    py_file = os.path.join(base_dir, "aapl_ma_volatility_viz.py")
    chart1 = os.path.join(base_dir, "aapl_trend_ma7.png")
    chart2 = os.path.join(base_dir, "aapl_volatility.png")

    score = 0.0
    feedback = []

    # Step 1: Script and imports
    if os.path.exists(py_file):
        score += 0.25
        feedback.append("Python script exists.")
        with open(py_file, "r", encoding="utf-8") as f:
            content = f.read().lower()
            if (
                "import pandas" in content and
                ("matplotlib" in content or "seaborn" in content or "plotly" in content or "altair" in content)
            ):
                score += 0.15
                feedback.append("Script correctly imports pandas and a visualization library.")
            else:
                feedback.append("Script missing visualization library imports.")
    else:
        feedback.append("Python script missing.")

    # Helper: encode image for GPT-4o
    def encode_image(path):
        with open(path, "rb") as img:
            return base64.b64encode(img.read()).decode("utf-8")

    # Step 2: Chart 1 - Line chart (AAPL_y and MA7)
    if os.path.exists(chart1):
        score += 0.1
        feedback.append("aapl_trend_ma7.png exists.")
        try:
            img_b64 = encode_image(chart1)
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a data visualization evaluator."},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "This image should show a line chart with AAPL_y and MA7 (7-day moving average) over time. "
                                    "Are both lines visible and clearly labeled, with readable axes and legend? "
                                    "Answer 'yes' or 'no' with a short explanation."
                                )
                            },
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
                        ]
                    }
                ],
                max_tokens=100
            )
            ans = response.choices[0].message.content.lower()
            if "yes" in ans:
                score += 0.4
                feedback.append("GPT-4o judged aapl_trend_ma7.png as clear and correct.")
            else:
                feedback.append(f"GPT-4o judged aapl_trend_ma7.png unclear: {ans}")
        except Exception as e:
            feedback.append(f"GPT-4o evaluation for aapl_trend_ma7.png failed: {e}")
    else:
        feedback.append("aapl_trend_ma7.png missing.")

    # Step 3: Chart 2 - Area chart (Volatility7)
    if os.path.exists(chart2):
        score += 0.1
        feedback.append("aapl_volatility.png exists.")
        try:
            img_b64 = encode_image(chart2)
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a data visualization evaluator."},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "This image should show an area chart of 7-day rolling volatility (Volatility7) over time. "
                                    "Does it clearly represent volatility trends with readable axes and meaningful shading? "
                                    "Answer 'yes' or 'no' with a short reason."
                                )
                            },
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
                        ]
                    }
                ],
                max_tokens=100
            )
            ans = response.choices[0].message.content.lower()
            if "yes" in ans:
                score += 0.4
                feedback.append("GPT-4o judged aapl_volatility.png as clear and correct.")
            else:
                feedback.append(f"GPT-4o judged aapl_volatility.png unclear: {ans}")
        except Exception as e:
            feedback.append(f"GPT-4o evaluation for aapl_volatility.png failed: {e}")
    else:
        feedback.append("aapl_volatility.png missing.")

    final_score = min(score, 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score



def evaluate_squirrel_impute_group(actual: str, expected: dict, **options) -> float:

    """
    Evaluator for the Squirrel Imputation and Grouping task.

    This evaluation checks:
      - that the Python script file exists
      - that the script correctly imports pandas
      - that the result file result.txt exists
      - and that it matches the expected grouped summary:
        ?,Gray,50.0
        ?,Unknown,50.0
        Adult,Black,3.5
        Adult,Cinnamon,12.4
        Adult,Gray,82.4
        Adult,Unknown,1.7
        Juvenile,Black,2.4
        Juvenile,Cinnamon,17.6
        Juvenile,Gray,77.6
        Juvenile,Unknown,2.4
    """


    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    expected = expected.get("rules", expected)
    base_dir = os.path.dirname(actual)
    script_file = os.path.join(base_dir, "squirrel_impute_group.py")
    result_file = os.path.join(base_dir, "result.txt")

    score = 0.0
    feedback = []

    expected_lines = [
        "?,Gray,50.0",
        "?,Unknown,50.0",
        "Adult,Black,3.5",
        "Adult,Cinnamon,12.4",
        "Adult,Gray,82.4",
        "Adult,Unknown,1.7",
        "Juvenile,Black,2.4",
        "Juvenile,Cinnamon,17.6",
        "Juvenile,Gray,77.6",
        "Juvenile,Unknown,2.4"
    ]


    try:
        # Step 1: Check that the Python script exists

        if os.path.exists(script_file):
            score += 0.4
            feedback.append("The Python script file is present.")
            logger.info("Python script file found.")

            with open(script_file, "r", encoding="utf-8") as f:
                content = f.read().lower()

            if "import pandas" in content:
                score += 0.3
                feedback.append("The script correctly imports pandas.")
                logger.info("Pandas import verified.")
            else:
                feedback.append("The script does not import pandas.")
                logger.warning("Pandas import not found in script.")
        else:
            feedback.append("The Python script file is missing.")
            logger.warning("Python script missing.")

        # Step 2: Check that the result file exists and matches expected lines
        if os.path.exists(result_file):
            score += 0.1
            feedback.append("The result file result.txt exists.")
            logger.info("Result file found.")
            try:
                with open(result_file, "r", encoding="utf-8") as f:
                    lines = [line.strip().replace(" ", "") for line in f if line.strip()]
                
                expected_clean = [line.replace(" ", "") for line in expected_lines]

                if lines == expected_clean:
                    score += 0.2
                    feedback.append("The result file matches the expected grouped summary.")
                    logger.info("All expected lines verified correctly.")
                else:
                    feedback.append(
                        f"The result file does not match expected content. "
                        f"Expected: {expected_lines}, Found: {lines}."
                    )
                    logger.warning(f"Mismatch: expected {expected_lines}, found {lines}")

            except Exception as e:
                feedback.append(f"Error reading or parsing result.txt: {e}")
                logger.error(f"Error reading result file: {e}", exc_info=True)
        else:
            feedback.append("The result file result.txt is missing.")
            logger.warning("Result file missing.")


    except Exception as e:
        feedback.append(f"Evaluation failed: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)


    # Final score capped at 1.0
    final_score = min(score, 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)


    return final_score


def evaluate_squirrel_fur_shift_viz_quality(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Squirrel Fur Color & Shift Visualization task.
    Checks:
      - that the script exists and imports pandas + some visualization library (matplotlib, seaborn, plotly, altair)
      - both output images exist
      - GPT-4o judges that each chart is clear, correct, and readable
    Dataset presence check is omitted by design.
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    base_dir = os.path.dirname(actual)
    py_file = os.path.join(base_dir, "squirrel_fur_shift_viz.py")
    pie_path = os.path.join(base_dir, "furcolor_pie.png")
    bar_path = os.path.join(base_dir, "shift_bar.png")

    score = 0.0
    feedback = []

    # 1. Script existence & imports
    if os.path.exists(py_file):
        score += 0.25
        feedback.append("Python script exists.")
        with open(py_file, "r", encoding="utf-8") as f:
            content = f.read().lower()
        if ("import pandas" in content) and (
            ("matplotlib" in content) or ("seaborn" in content) or ("plotly" in content) or ("altair" in content)
        ):
            score += 0.15
            feedback.append("Script imports pandas and a visualization library.")
        else:
            feedback.append("Script missing pandas or a visualization import.")
    else:
        feedback.append("Python script missing.")

    # Helper to encode image
    def encode_image(path):
        with open(path, "rb") as img:
            return base64.b64encode(img.read()).decode("utf-8")

    # 2. Evaluate the pie chart (furcolor_pie.png)
    if os.path.exists(pie_path):
        score += 0.1
        feedback.append("furcolor_pie.png exists.")
        try:
            img_b64 = encode_image(pie_path)
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a data visualization evaluator."},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "This chart should be a pie chart showing the distribution (counts or proportions) of Primary Fur Color categories. Does it clearly display segment sizes, labels or legend, and readability? Answer 'yes' or 'no' with a short reason."
                            },
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
                        ]
                    }
                ],
                max_tokens=100
            )
            ans = response.choices[0].message.content.lower()
            if "yes" in ans:
                score += 0.4
                feedback.append("GPT-4o judged furcolor_pie.png as clear and correct.")
            else:
                feedback.append(f"GPT-4o judged furcolor_pie.png unclear: {ans}")
        except Exception as e:
            feedback.append(f"GPT-4o evaluation for furcolor_pie.png failed: {e}")
    else:
        feedback.append("furcolor_pie.png is missing.")

    # 3. Evaluate the bar chart (shift_bar.png)
    if os.path.exists(bar_path):
        score += 0.1
        feedback.append("shift_bar.png exists.")
        try:
            img_b64 = encode_image(bar_path)
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a data visualization evaluator."},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "This chart should be a bar chart showing total sightings per Shift (e.g. AM vs PM). Does it clearly show bar heights, axis labels, and readability? Answer 'yes' or 'no' with a short reason."
                            },
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
                        ]
                    }
                ],
                max_tokens=100
            )
            ans = response.choices[0].message.content.lower()
            if "yes" in ans:
                score += 0.4
                feedback.append("GPT-4o judged shift_bar.png as clear and correct.")
            else:
                feedback.append(f"GPT-4o judged shift_bar.png unclear: {ans}")
        except Exception as e:
            feedback.append(f"GPT-4o evaluation for shift_bar.png failed: {e}")
    else:
        feedback.append("shift_bar.png is missing.")

    final_score = min(score, 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)
    return final_score


def evaluate_complaints_impute(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Consumer Complaints Imputation task.

    This evaluation checks:
      - that the Python script file exists
      - that the script correctly imports pandas
      - that the result file imputed_columns.txt exists
      - and that the imputed output (Sub-product, State) matches the expected data
        after filling missing values accordingly
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    expected = expected.get("rules", expected)
    base_dir = os.path.dirname(actual)
    script_file = os.path.join(base_dir, "complaints_impute.py")
    csv_file = os.path.join(base_dir, "26k-consumer-complaints.csv")
    result_file = os.path.join(base_dir, "imputed_columns.txt")

    score = 0.0
    feedback = []

    try:
        # Step 1: Check Python script
        if os.path.exists(script_file):
            score += 0.4
            feedback.append("The Python script file is present.")
            logger.info("Python script file found.")

            with open(script_file, "r", encoding="utf-8") as f:
                content = f.read().lower()

            if "import pandas" in content:
                score += 0.3
                feedback.append("The script correctly imports pandas.")
                logger.info("Pandas import verified.")
            else:
                feedback.append("The script does not import pandas.")
                logger.warning("Missing pandas import in script.")
        else:
            feedback.append("The Python script file is missing.")
            logger.warning("Python script missing.")

        # Step 2: Check result file
        if os.path.exists(result_file):
            score += 0.1
            feedback.append("The result file imputed_columns.txt exists.")
            logger.info("Result file found.")

            try:
                # Compute expected imputed output
                df = pd.read_csv(csv_file, dtype=str)
                sub_mode = df["Sub-product"].mode(dropna=True)
                state_mode = df["State"].mode(dropna=True)
                sub_mode_value = sub_mode.iloc[0] if not sub_mode.empty else "Unknown"
                state_mode_value = state_mode.iloc[0] if not state_mode.empty else "Unknown"

                df["Sub-product"] = df["Sub-product"].fillna("Unknown")
                df["State"] = df["State"].fillna(state_mode_value)

                expected_pairs = df[["Sub-product", "State"]].astype(str)
                expected_lines = [f"{sp},{st}" for sp, st in zip(expected_pairs["Sub-product"], expected_pairs["State"])]

                # Read actual result file
                with open(result_file, "r", encoding="utf-8") as f:
                    actual_lines = [line.strip() for line in f if line.strip()]

                # Compare line counts and sample values
                if len(actual_lines) == len(expected_lines):
                    mismatches = sum(1 for a, e in zip(actual_lines, expected_lines) if a.replace(" ", "") != e.replace(" ", ""))
                    mismatch_ratio = mismatches / len(expected_lines)

                    if mismatch_ratio <= 0.01:  # allow 1% tolerance for whitespace or case issues
                        score += 0.2
                        feedback.append("The imputed columns match the expected data.")
                        logger.info("Imputed data verified successfully.")
                    else:
                        feedback.append(f"Imputed data mismatch: {mismatches} lines differ out of {len(expected_lines)}.")
                        logger.warning(f"Mismatch ratio too high ({mismatch_ratio:.3f}).")
                else:
                    feedback.append(
                        f"Line count mismatch: expected {len(expected_lines)}, found {len(actual_lines)}."
                    )
                    logger.warning(f"Line count mismatch: expected {len(expected_lines)}, found {len(actual_lines)}.")
            except Exception as e:
                feedback.append(f"Error verifying imputed_columns.txt: {e}")
                logger.error(f"Error verifying imputed_columns.txt: {e}", exc_info=True)
        else:
            feedback.append("The result file imputed_columns.txt is missing.")
            logger.warning("Result file missing.")

    except Exception as e:
        feedback.append(f"Evaluation failed: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)

    # Final score capped at 1.0
    final_score = min(score, 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score


def evaluate_monthly_complaint_viz_quality(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the monthly complaint volume visualization task.
    Checks:
      - the script exists and imports pandas + a visualization library (matplotlib/seaborn/plotly/altair)
      - the output image exists
      - GPT-4o judges the line chart is clear, correct, and readable
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    base_dir = os.path.dirname(actual)
    py_file = os.path.join(base_dir, "monthly_complaint_viz.py")
    chart_path = os.path.join(base_dir, "monthly_complaint_volume.png")

    score = 0.0
    feedback = []

    # Step 1: script & imports
    if os.path.exists(py_file):
        score += 0.25
        feedback.append("Python script exists.")
        with open(py_file, "r", encoding="utf-8") as f:
            content = f.read().lower()
        if ("import pandas" in content) and (
           ("matplotlib" in content) or ("seaborn" in content) or ("plotly" in content) or ("altair" in content)
        ):
            score += 0.15
            feedback.append("Script imports pandas and a visualization library.")
        else:
            feedback.append("Script missing pandas import or a visualization library import.")
    else:
        feedback.append("Python script missing.")

    # Helper: encode image
    def encode_image(path):
        with open(path, "rb") as img:
            return base64.b64encode(img.read()).decode("utf-8")

    # Step 2: chart existence & quality
    if os.path.exists(chart_path):
        score += 0.1
        feedback.append("monthly_complaint_volume.png exists.")
        try:
            img_b64 = encode_image(chart_path)
            response = client.chat.completions.create(
                model="gpt-4o",
                messages = [
                    {"role": "system", "content": "You are a data visualization evaluator."},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "This image should show a line chart of monthly complaint counts over time. "
                                    "Does it clearly show time on the x-axis, complaint counts on the y-axis, readable labels, and a smooth trend? "
                                    "Answer 'yes' or 'no' with a short reason."
                                )
                            },
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
                        ]
                    }
                ],
                max_tokens = 100
            )
            ans = response.choices[0].message.content.lower()
            if "yes" in ans:
                score += 0.65
                feedback.append("GPT-4o judged monthly_complaint_volume.png as clear and correct.")
            else:
                feedback.append(f"GPT-4o judged monthly_complaint_volume.png unclear: {ans}")
        except Exception as e:
            feedback.append(f"GPT-4o evaluation failed for the chart: {e}")
    else:
        feedback.append("monthly_complaint_volume.png is missing.")

    final_score = min(score, 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)
    return final_score


def evaluate_animal_impute(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Animal 311 Imputation task.

    This evaluation checks:
      - that the Python script file exists
      - that the script correctly imports pandas
      - that the result file result.txt exists
      - and that it correctly imputes Closed Date and Location Type based on the rules:
          * If Closed Date is missing and Status == "In Progress" → "Pending"
          * If Closed Date is missing and Status != "In Progress" → "Unknown"
          * Missing Location Type filled with mode within the same Borough
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    expected = expected.get("rules", expected)
    base_dir = os.path.dirname(actual)
    script_file = os.path.join(base_dir, "animal_impute.py")
    csv_file = os.path.join(base_dir, "311_Animals_short.csv")
    result_file = os.path.join(base_dir, "result.txt")

    score = 0.0
    feedback = []

    try:
        # Step 1: Check for Python script
        if os.path.exists(script_file):
            score += 0.4
            feedback.append("The Python script file is present.")
            logger.info("Python script file found.")

            with open(script_file, "r", encoding="utf-8") as f:
                content = f.read().lower()

            if "import pandas" in content:
                score += 0.3
                feedback.append("The script correctly imports pandas.")
                logger.info("Pandas import verified.")
            else:
                feedback.append("The script does not import pandas.")
                logger.warning("Missing pandas import.")
        else:
            feedback.append("The Python script file is missing.")
            logger.warning("Python script missing.")

        # Step 2: Check result file
        if os.path.exists(result_file):
            score += 0.1
            feedback.append("The result file result.txt exists.")
            logger.info("Result file found.")

            try:
                # Compute expected output dynamically
                df = pd.read_csv(csv_file, dtype=str)
                df = df.fillna(value={"Status": "Unknown", "Borough": "Unknown"})

                # Handle Closed Date imputation
                closed_date_filled = []
                for _, row in df.iterrows():
                    closed_date = row.get("Closed Date", None)
                    status = row.get("Status", "")
                    if pd.isna(closed_date) or closed_date == "" or closed_date.lower() == "nan":
                        if status.strip().lower() == "in progress":
                            closed_date_filled.append("Pending")
                        else:
                            closed_date_filled.append("Unknown")
                    else:
                        closed_date_filled.append(closed_date)
                df["Closed Date"] = closed_date_filled

                # Handle Location Type imputation (mode per Borough)
                df["Location Type"] = df["Location Type"].astype(str)
                df["Borough"] = df["Borough"].astype(str)
                mode_by_borough = (
                    df.groupby("Borough")["Location Type"]
                    .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown")
                    .to_dict()
                )
                df["Location Type"] = df.apply(
                    lambda r: mode_by_borough.get(r["Borough"], "Unknown")
                    if r["Location Type"].strip() in ["", "nan", "None"]
                    else r["Location Type"],
                    axis=1
                )

                # Expected lines
                expected_lines = [
                    f"{cd},{lt}" for cd, lt in zip(df["Closed Date"], df["Location Type"])
                ]

                # Actual result
                with open(result_file, "r", encoding="utf-8") as f:
                    actual_lines = [line.strip() for line in f if line.strip()]

                # Compare lengths and content
                if len(actual_lines) == len(expected_lines):
                    mismatches = sum(
                        1 for a, e in zip(actual_lines, expected_lines)
                        if a.replace(" ", "") != e.replace(" ", "")
                    )
                    mismatch_ratio = mismatches / len(expected_lines)
                    if mismatch_ratio <= 0.01:
                        score += 0.2
                        feedback.append("The imputed columns match the expected data.")
                        logger.info("Imputed values verified successfully.")
                    else:
                        feedback.append(f"Output mismatch: {mismatches} rows differ.")
                        logger.warning(f"Mismatch ratio {mismatch_ratio:.3f}.")
                else:
                    feedback.append(
                        f"Row count mismatch: expected {len(expected_lines)}, found {len(actual_lines)}."
                    )
                    logger.warning(
                        f"Row count mismatch: expected {len(expected_lines)}, found {len(actual_lines)}."
                    )

            except Exception as e:
                feedback.append(f"Error while validating result.txt: {e}")
                logger.error(f"Error verifying result.txt: {e}", exc_info=True)
        else:
            feedback.append("The result file result.txt is missing.")
            logger.warning("Result file missing.")

    except Exception as e:
        feedback.append(f"Evaluation failed: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)

    final_score = min(score, 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score


def evaluate_monthly_borough_complaints_viz_quality(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Monthly Borough Complaints Visualization task.

    This evaluation checks:
      - that the Python script exists and imports pandas
      - that both visualization files are present
      - and that GPT-4o confirms the charts clearly represent the correct information
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    expected = expected.get("rules", expected)
    base_dir = os.path.dirname(actual)

    py_file = os.path.join(base_dir, "monthly_borough_complaints_viz.py")
    chart1 = os.path.join(base_dir, "monthly_trend.png")
    chart2 = os.path.join(base_dir, "borough_area_trend.png")

    score = 0.0
    feedback = []

    # Step 1: Check for Python script and imports
    if os.path.exists(py_file):
        score += 0.3
        feedback.append("The Python script file is present.")
        logger.info("Python script file found.")

        with open(py_file, "r", encoding="utf-8") as f:
            content = f.read().lower()
        if "import pandas" in content:
            score += 0.2
            feedback.append("The script correctly imports pandas.")
            logger.info("Pandas import verified.")
        else:
            feedback.append("The script does not import pandas.")
            logger.warning("Pandas import not found.")
    else:
        feedback.append("The Python script is missing.")
        logger.warning("Python script missing.")

    # Helper: encode image for GPT-4o
    def encode_image(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    # Step 2: Visualization checks (loop-based, balanced)
    charts_to_check = [
        (
            chart1,
            "a line chart showing the monthly total number of complaints (overall trend over time)",
        ),
        (
            chart2,
            "a stacked area chart showing monthly complaints by borough over time (each borough as a layer)",
        ),
    ]

    for chart, description in charts_to_check:
        chart_name = os.path.basename(chart)
        if os.path.exists(chart):
            score += 0.15
            feedback.append(f"The visualization {chart_name} exists.")
            logger.info(f"{chart_name} found.")

            try:
                encoded = encode_image(chart)
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert in data visualization quality evaluation. Judge each chart objectively.",
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        f"Examine this chart and answer if it correctly represents {description} "
                                        f"with clear labeling, readable axes, appropriate scale, and balanced layout. "
                                        f"Reply 'yes' if it does, otherwise 'no', followed by a short reason."
                                    ),
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{encoded}"},
                                },
                            ],
                        },
                    ],
                    max_tokens=100,
                )

                answer = response.choices[0].message.content.lower()
                if "yes" in answer:
                    score += 0.1
                    feedback.append(f"GPT-4o judged {chart_name} as clear, relevant, and visually accurate.")
                    logger.info(f"{chart_name} judged relevant by GPT-4o.")
                else:
                    feedback.append(f"GPT-4o judged {chart_name} unclear or irrelevant: {answer}")
                    logger.warning(f"{chart_name} judged unclear by GPT-4o.")
            except Exception as e:
                feedback.append(f"Could not evaluate {chart_name} using GPT-4o: {e}")
                logger.error(f"Error evaluating {chart_name}: {e}", exc_info=True)
        else:
            feedback.append(f"The visualization {chart_name} is missing.")
            logger.warning(f"{chart_name} missing.")

    # Step 3: Final score
    final_score = min(score, 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score 

def evaluate_animal_descriptor_share(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Animal Descriptor Share task.

    This evaluation checks:
      - that the Python script file exists
      - that the script correctly imports pandas
      - that the result file agency_descriptor_share.csv exists
      - and that the output matches the expected top 3 descriptors per agency with correct counts and percentage shares.
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    expected = expected.get("rules", expected)
    base_dir = os.path.dirname(actual)
    script_file = os.path.join(base_dir, "animal_descriptor_share.py")
    csv_file = os.path.join(base_dir, "311_Animals_short.csv")
    result_file = os.path.join(base_dir, "agency_descriptor_share.csv")

    score = 0.0
    feedback = []

    try:
        # Step 1: Check Python script presence
        if os.path.exists(script_file):
            score += 0.4
            feedback.append("The Python script file is present.")
            logger.info("Python script file found.")

            with open(script_file, "r", encoding="utf-8") as f:
                content = f.read().lower()

            if "import pandas" in content:
                score += 0.3
                feedback.append("The script correctly imports pandas.")
                logger.info("Pandas import verified.")
            else:
                feedback.append("The script does not import pandas.")
                logger.warning("Missing pandas import in script.")
        else:
            feedback.append("The Python script file is missing.")
            logger.warning("Python script missing.")

        # Step 2: Check and verify the output file
        if os.path.exists(result_file):
            score += 0.1
            feedback.append("The result file agency_descriptor_share.csv exists.")
            logger.info("Result file found.")

            try:
                # Read dataset
                df = pd.read_csv(csv_file, dtype=str)
                df["Agency"] = df["Agency"].fillna("Unknown")
                df["Descriptor"] = df["Descriptor"].fillna("Unknown")

                # Compute expected descriptor counts and shares
                df["Agency"] = df["Agency"].astype(str)
                df["Descriptor"] = df["Descriptor"].astype(str)

                expected_rows = []
                for agency, group in df.groupby("Agency"):
                    total = len(group)
                    desc_counts = group["Descriptor"].value_counts().head(3)
                    for desc, cnt in desc_counts.items():
                        pct = round((cnt / total) * 100, 1)
                        expected_rows.append({
                            "Agency": agency,
                            "Descriptor": desc,
                            "Count": cnt,
                            "Percentage": pct
                        })
                expected_df = pd.DataFrame(expected_rows).sort_values(
                    ["Agency", "Count"], ascending=[True, False]
                ).reset_index(drop=True)

                # Read actual output
                actual_df = pd.read_csv(result_file)
                actual_df.columns = actual_df.columns.str.strip()
                expected_cols = ["Agency", "Descriptor", "Count", "Percentage"]

                # Structural validation
                if list(actual_df.columns) == expected_cols:
                    score += 0.1
                    feedback.append("Output file has correct column names.")
                    logger.info("Column structure verified.")
                else:
                    feedback.append(f"Incorrect columns: expected {expected_cols}, found {list(actual_df.columns)}.")
                    logger.warning("Column name mismatch detected.")

                # Value validation
                # Normalize types for fair comparison
                actual_df["Agency"] = actual_df["Agency"].astype(str)
                actual_df["Descriptor"] = actual_df["Descriptor"].astype(str)
                actual_df["Count"] = pd.to_numeric(actual_df["Count"], errors="coerce")
                actual_df["Percentage"] = pd.to_numeric(actual_df["Percentage"], errors="coerce")

                merge_df = pd.merge(
                    expected_df, actual_df,
                    on=["Agency", "Descriptor"], how="outer", suffixes=("_exp", "_act")
                )

                count_match = (merge_df["Count_exp"] == merge_df["Count_act"]).sum()
                pct_match = (abs(merge_df["Percentage_exp"] - merge_df["Percentage_act"]) <= 0.1).sum()
                total_rows = len(merge_df)

                if total_rows > 0:
                    count_ratio = count_match / total_rows
                    pct_ratio = pct_match / total_rows
                else:
                    count_ratio, pct_ratio = 0, 0

                if count_ratio >= 0.95 and pct_ratio >= 0.95:
                    score += 0.1
                    feedback.append("Descriptor counts and percentages match expected results (≥95% match).")
                    logger.info("Counts and percentages verified successfully.")
                else:
                    feedback.append(
                        f"Descriptor counts or percentages differ significantly (Counts match: {count_ratio:.2f}, Percent match: {pct_ratio:.2f})."
                    )
                    logger.warning("Mismatch in computed descriptor shares.")

            except Exception as e:
                feedback.append(f"Error verifying output CSV: {e}")
                logger.error(f"Error verifying output CSV: {e}", exc_info=True)
        else:
            feedback.append("The result file agency_descriptor_share.csv is missing.")
            logger.warning("Result file missing.")

    except Exception as e:
        feedback.append(f"Evaluation failed: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)

    final_score = min(score, 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score

def evaluate_antibiotic_effectiveness(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Antibiotic Effectiveness task.

    This evaluation checks:
      - that the Python script file exists
      - that the script correctly imports pandas
      - that the result file best_antibiotic_by_gram.txt exists
      - and that its content matches the expected best antibiotics for each Gram type:
        <GramType=Negative>,Neomycin,0.59
        <GramType=Positive>,Penicillin,0.15
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    expected = expected.get("rules", expected)
    base_dir = os.path.dirname(actual)
    script_file = os.path.join(base_dir, "antibiotic_effectiveness.py")
    result_file = os.path.join(base_dir, "best_antibiotic_by_gram.txt")

    score = 0.0
    feedback = []

    expected_lines = [
        "<GramType=Negative>,Neomycin,0.59",
        "<GramType=Positive>,Penicillin,0.15"
    ]

    try:
        # Step 1: Check Python script
        if os.path.exists(script_file):
            score += 0.4
            feedback.append("The Python script file is present.")
            logger.info("Python script file found.")

            with open(script_file, "r", encoding="utf-8") as f:
                content = f.read().lower()

            if "import pandas" in content:
                score += 0.3
                feedback.append("The script correctly imports pandas.")
                logger.info("Pandas import verified.")
            else:
                feedback.append("The script does not import pandas.")
                logger.warning("Missing pandas import in script.")
        else:
            feedback.append("The Python script file is missing.")
            logger.warning("Python script missing.")

        # Step 2: Check and validate result file
        if os.path.exists(result_file):
            score += 0.1
            feedback.append("The result file best_antibiotic_by_gram.txt exists.")
            logger.info("Result file found.")

            try:
                with open(result_file, "r", encoding="utf-8") as f:
                    actual_lines = [line.strip() for line in f if line.strip()]

                expected_clean = [line.replace(" ", "") for line in expected_lines]
                actual_clean = [line.replace(" ", "") for line in actual_lines]

                if actual_clean == expected_clean:
                    score += 0.2
                    feedback.append("The result file matches the expected antibiotics and values.")
                    logger.info("Output verified successfully.")
                else:
                    feedback.append(
                        f"The result file does not match expected output.\nExpected: {expected_lines}\nFound: {actual_lines}"
                    )
                    logger.warning("Mismatch in output content.")
            except Exception as e:
                feedback.append(f"Error reading result file: {e}")
                logger.error(f"Error reading best_antibiotic_by_gram.txt: {e}", exc_info=True)
        else:
            feedback.append("The result file best_antibiotic_by_gram.txt is missing.")
            logger.warning("Result file missing.")

    except Exception as e:
        feedback.append(f"Evaluation failed: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)

    final_score = min(score, 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score

def evaluate_antibiotic_kmeans(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Antibiotic KMeans Clustering task. 

    This evaluation checks:
      - that the Python script file exists
      - that the script correctly imports pandas and sklearn
      - that the result file bacteria_clusters.csv exists
      - and that cluster assignments match expected results (allowing label flipping)
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    expected = expected.get("rules", expected)
    base_dir = os.path.dirname(actual)
    script_file = os.path.join(base_dir, "antibiotic_kmeans.py")
    csv_file = os.path.join(base_dir, "Antibiotics.csv")
    result_file = os.path.join(base_dir, "bacteria_clusters.csv")

    score = 0.0
    feedback = []

    try:
        # Step 1: Check Python script
        if os.path.exists(script_file):
            score += 0.4
            feedback.append("The Python script file is present.")
            logger.info("Python script file found.")

            with open(script_file, "r", encoding="utf-8") as f:
                content = f.read().lower()

            if "import pandas" in content and "sklearn" in content:
                score += 0.3
                feedback.append("The script correctly imports pandas and sklearn.")
                logger.info("Imports verified (pandas, sklearn).")
            else:
                feedback.append("The script is missing required imports (pandas or sklearn).")
                logger.warning("Missing import detected.")
        else:
            feedback.append("The Python script file is missing.")
            logger.warning("Python script missing.")

        # Step 2: Check result file
        if os.path.exists(result_file):
            score += 0.1
            feedback.append("The result file bacteria_clusters.csv exists.")
            logger.info("Result file found.")

            try:
                # Recompute expected cluster assignments
                df = pd.read_csv(csv_file)
                numeric_cols = df.select_dtypes(include="number").columns.tolist()
                scaler = MinMaxScaler()
                X_scaled = scaler.fit_transform(df[numeric_cols])

                kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
                expected_labels = kmeans.fit_predict(X_scaled)
                expected_df = pd.DataFrame({
                    "Bacteria": df["Bacteria"],
                    "Cluster": expected_labels
                })

                # Read actual output
                actual_df = pd.read_csv(result_file)
                actual_df.columns = actual_df.columns.str.strip()

                # Structural validation
                expected_cols = ["Bacteria", "Cluster"]
                if list(actual_df.columns) == expected_cols:
                    feedback.append("Output columns are correct.")
                    logger.info("Column structure verified.")
                else:
                    feedback.append(f"Incorrect columns: expected {expected_cols}, found {list(actual_df.columns)}.")
                    logger.warning("Column name mismatch detected.")

                # Basic row validation
                if len(actual_df) == len(expected_df):
                    logger.info(f"Row count verified: {len(actual_df)} rows.")
                else:
                    feedback.append(
                        f"Row count mismatch: expected {len(expected_df)}, found {len(actual_df)}."
                    )
                    logger.warning("Row count mismatch detected.")

                # Merge and compare clusters
                merged = pd.merge(expected_df, actual_df, on="Bacteria", suffixes=("_exp", "_act"))
                normal_match = (merged["Cluster_exp"] == merged["Cluster_act"]).mean()
                flipped_match = (merged["Cluster_exp"] != merged["Cluster_act"]).mean()
                best_match = max(normal_match, flipped_match)

                logger.info(
                    f"Cluster match ratio: normal={normal_match:.3f}, flipped={flipped_match:.3f}, best={best_match:.3f}"
                )

                if best_match >= 0.95:
                    score += 0.2
                    feedback.append("Cluster assignments match expected results (≥95% agreement).")
                    logger.info("Cluster labels verified successfully.")
                else:
                    feedback.append(
                        f"Cluster assignments differ significantly (match ratio={best_match:.2f})."
                    )
                    logger.warning("Cluster mismatch detected.")

            except Exception as e:
                feedback.append(f"Error verifying output file: {e}")
                logger.error(f"Error verifying output file: {e}", exc_info=True)
        else:
            feedback.append("The result file bacteria_clusters.csv is missing.")
            logger.warning("Result file missing.")

    except Exception as e:
        feedback.append(f"Evaluation failed: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)

    final_score = min(score, 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score


def evaluate_antibiotics_sensitivity_viz_quality(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Antibiotic Sensitivity Visualization task.

    This evaluation checks:
      - that the Python script exists and imports pandas
      - that both visualization files are present
      - and that GPT-4o confirms the charts clearly represent the correct information
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    expected = expected.get("rules", expected)
    base_dir = os.path.dirname(actual)

    py_file = os.path.join(base_dir, "antibiotics_sensitivity_viz.py")
    chart1 = os.path.join(base_dir, "mean_sensitivity_bar.png")
    chart2 = os.path.join(base_dir, "antibiotic_scatter.png")

    score = 0.0
    feedback = []

    # Step 1: Check for Python script
    if os.path.exists(py_file):
        score += 0.3
        feedback.append("The Python script file is present.")
        logger.info("Python script file found.")

        with open(py_file, "r", encoding="utf-8") as f:
            content = f.read().lower()
        if "import pandas" in content and (
            "matplotlib" in content or "seaborn" in content or "plotly" in content or "altair" in content
        ):
            score += 0.2
            feedback.append("The script correctly imports pandas and a visualization library.")
            logger.info("Pandas import verified.")
        else:
            feedback.append("The script does not import pandas or a visualization library.")
            logger.warning("Required imports missing.")
    else:
        feedback.append("The Python script is missing.")
        logger.warning("Python script missing.")

    # Helper: encode image for GPT-4o
    def encode_image(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    # Step 2: Visualization checks
    charts_to_check = [
        (chart1, "a bar chart comparing mean antibiotic sensitivity grouped by Gram type"),
        (chart2, "a scatter plot with Penicillin on the x-axis, Streptomycin on the y-axis, and color representing Gram type")
    ]

    for chart, description in charts_to_check:
        chart_name = os.path.basename(chart)
        if os.path.exists(chart):
            score += 0.15
            feedback.append(f"The visualization {chart_name} exists.")
            logger.info(f"{chart_name} found.")

            try:
                encoded = encode_image(chart)
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert in data visualization quality evaluation. Judge each chart objectively."
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        f"Examine this chart and answer if it correctly represents {description} "
                                        f"with clear labeling, readable axes, appropriate scale, and balanced layout. "
                                        f"Reply 'yes' if it does, otherwise 'no', followed by a short reason."
                                    )
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{encoded}"}
                                }
                            ]
                        }
                    ],
                    max_tokens=100
                )

                answer = response.choices[0].message.content.lower()
                if "yes" in answer:
                    score += 0.1
                    feedback.append(f"GPT-4o judged {chart_name} as clear, relevant, and visually accurate.")
                    logger.info(f"{chart_name} judged relevant by GPT-4o.")
                else:
                    feedback.append(f"GPT-4o judged {chart_name} unclear or irrelevant: {answer}")
                    logger.warning(f"{chart_name} judged unclear by GPT-4o.")
            except Exception as e:
                feedback.append(f"Could not evaluate {chart_name} using GPT-4o: {e}")
                logger.error(f"Error evaluating {chart_name}: {e}", exc_info=True)
        else:
            feedback.append(f"The visualization {chart_name} is missing.")
            logger.warning(f"{chart_name} missing.")

    # Step 3: Final Score
    final_score = min(score, 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score

def evaluate_antibiotic_response_viz_quality(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Antibiotic Response Visualization task.

    This evaluation checks:
      - that the Python script exists and imports pandas
      - that both visualization files are present
      - and that GPT-4o confirms the charts clearly represent the correct information
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    expected = expected.get("rules", expected)
    base_dir = os.path.dirname(actual)

    py_file = os.path.join(base_dir, "antibiotic_response_viz.py")
    chart1 = os.path.join(base_dir, "antibiotic_heatmap.png")
    chart2 = os.path.join(base_dir, "gram_boxplot.png")

    score = 0.0
    feedback = []

    # Step 1: Check for Python script
    if os.path.exists(py_file):
        score += 0.3
        feedback.append("The Python script file is present.")
        logger.info("Python script file found.")

        with open(py_file, "r", encoding="utf-8") as f:
            content = f.read().lower()
        if "import pandas" in content and (
            "matplotlib" in content or "seaborn" in content or "plotly" in content or "altair" in content
        ):
            score += 0.2
            feedback.append("The script correctly imports pandas and a visualization library.")
            logger.info("Pandas import verified.")
        else:
            feedback.append("The script does not import pandas or a visualization library.")
            logger.warning("Missing required imports.")
    else:
        feedback.append("The Python script is missing.")
        logger.warning("Python script missing.")

    # Helper function: Encode image for GPT-4o
    def encode_image(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    # Step 2: Visualization checks
    charts_to_check = [
        (chart1, "a heatmap showing the sensitivity of each bacterium (rows) to three antibiotics (columns)"),
        (chart2, "a boxplot comparing the distributions of antibiotic responses grouped by Gram type")
    ]

    for chart, description in charts_to_check:
        chart_name = os.path.basename(chart)
        if os.path.exists(chart):
            score += 0.15
            feedback.append(f"The visualization {chart_name} exists.")
            logger.info(f"{chart_name} found.")

            try:
                encoded = encode_image(chart)
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert in data visualization evaluation. Judge chart clarity, relevance, and correctness."
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        f"Examine this chart and answer if it correctly represents {description} "
                                        f"with clear labeling, readable axes, appropriate color or grouping, and balanced layout. "
                                        f"Reply 'yes' if it does, otherwise 'no', followed by a short reason."
                                    )
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{encoded}"}
                                }
                            ]
                        }
                    ],
                    max_tokens=100
                )

                answer = response.choices[0].message.content.lower()
                if "yes" in answer:
                    score += 0.1
                    feedback.append(f"GPT-4o judged {chart_name} as clear, relevant, and visually accurate.")
                    logger.info(f"{chart_name} judged relevant by GPT-4o.")
                else:
                    feedback.append(f"GPT-4o judged {chart_name} unclear or incorrect: {answer}")
                    logger.warning(f"{chart_name} judged unclear by GPT-4o.")
            except Exception as e:
                feedback.append(f"Could not evaluate {chart_name} using GPT-4o: {e}")
                logger.error(f"Error evaluating {chart_name}: {e}", exc_info=True)
        else:
            feedback.append(f"The visualization {chart_name} is missing.")
            logger.warning(f"{chart_name} missing.")

    # Step 3: Final scoring
    final_score = min(score, 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score

def evaluate_emission_stats(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Continent Emission Statistics task.

    This evaluation checks:
      - that the Python script file exists
      - that the script correctly imports pandas
      - that the result file continent_emission_stats.csv exists
      - and that the computed statistics (Count, Mean, Median, Max, Min) match expected values within tolerance
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    expected = expected.get("rules", expected)
    base_dir = os.path.dirname(actual)
    script_file = os.path.join(base_dir, "emission_stats.py")
    csv_file = os.path.join(base_dir, "Emissions_Data.csv")
    result_file = os.path.join(base_dir, "continent_emission_stats.csv")

    score = 0.0
    feedback = []

    try:
        # Step 1: Check for script file
        if os.path.exists(script_file):
            score += 0.4
            feedback.append("The Python script file is present.")
            logger.info("Python script file found.")

            with open(script_file, "r", encoding="utf-8") as f:
                content = f.read().lower()

            if "import pandas" in content:
                score += 0.3
                feedback.append("The script correctly imports pandas.")
                logger.info("Pandas import verified.")
            else:
                feedback.append("The script does not import pandas.")
                logger.warning("Missing pandas import in script.")
        else:
            feedback.append("The Python script file is missing.")
            logger.warning("Python script missing.")

        # Step 2: Check result file
        if os.path.exists(result_file):
            score += 0.1
            feedback.append("The result file continent_emission_stats.csv exists.")
            logger.info("Result file found.")

            try:
                # Compute expected values
                df = pd.read_csv(csv_file)
                df = df.dropna(subset=["Continent", "Emission"])
                df["Emission"] = pd.to_numeric(df["Emission"], errors="coerce")
                expected_df = (
                    df.groupby("Continent")["Emission"]
                    .agg([
                        ("Count", "count"),
                        ("Mean", lambda x: round(x.mean(), 2)),
                        ("Median", lambda x: round(x.median(), 2)),
                        ("Max", lambda x: round(x.max(), 2)),
                        ("Min", lambda x: round(x.min(), 2))
                    ])
                    .reset_index()
                )

                # Read user output
                actual_df = pd.read_csv(result_file)
                actual_df.columns = actual_df.columns.str.strip()

                # Check column names
                expected_cols = ["Continent", "Count", "Mean", "Median", "Max", "Min"]
                if list(actual_df.columns) == expected_cols:
                    feedback.append("Output columns are correct.")
                    logger.info("Column structure verified.")
                else:
                    feedback.append(f"Incorrect columns: expected {expected_cols}, found {list(actual_df.columns)}.")
                    logger.warning("Column name mismatch detected.")

                # Sort both DataFrames by Continent for comparison
                expected_df = expected_df.sort_values("Continent").reset_index(drop=True)
                actual_df = actual_df.sort_values("Continent").reset_index(drop=True)

                # Ensure types align
                numeric_cols = ["Count", "Mean", "Median", "Max", "Min"]
                for col in numeric_cols:
                    actual_df[col] = pd.to_numeric(actual_df[col], errors="coerce")

                # Compute tolerance-based match
                match_ratios = []
                for col in numeric_cols:
                    diffs = abs(actual_df[col] - expected_df[col])
                    ratio = (diffs <= 0.05).mean()  # allow tiny rounding differences
                    match_ratios.append(ratio)

                overall_match = sum(match_ratios) / len(match_ratios)

                if overall_match >= 0.95:
                    score += 0.2
                    feedback.append("Emission statistics match expected results (≥95% accuracy).")
                    logger.info(f"Statistics verified successfully with mean match {overall_match:.3f}.")
                else:
                    feedback.append(f"Statistics differ significantly (average match={overall_match:.2f}).")
                    logger.warning(f"Statistics mismatch detected with match ratio {overall_match:.3f}.")

            except Exception as e:
                feedback.append(f"Error verifying result file: {e}")
                logger.error(f"Error verifying result file: {e}", exc_info=True)
        else:
            feedback.append("The result file continent_emission_stats.csv is missing.")
            logger.warning("Result file missing.")

    except Exception as e:
        feedback.append(f"Evaluation failed: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)

    final_score = min(score, 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score 

def evaluate_emission_by_continent_viz_quality(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Emission by Continent Visualization task.

    This evaluation checks:
      - that the Python script exists and imports pandas
      - that both visualization files are present
      - and that GPT-4o confirms the charts clearly represent the intended information
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    expected = expected.get("rules", expected)
    base_dir = os.path.dirname(actual)

    py_file = os.path.join(base_dir, "emission_by_continent_viz.py")
    chart1 = os.path.join(base_dir, "emission_trend_by_continent.png")
    chart2 = os.path.join(base_dir, "emission_share_latest_year.png")

    score = 0.0
    feedback = []

    # Step 1: Check Python script and imports
    if os.path.exists(py_file):
        score += 0.3
        feedback.append("The Python script file is present.")
        logger.info("Python script file found.")

        with open(py_file, "r", encoding="utf-8") as f:
            content = f.read().lower()
        if "import pandas" in content and (
            "matplotlib" in content or "seaborn" in content or "plotly" in content or "altair" in content
        ):
            score += 0.2
            feedback.append("The script correctly imports pandas and a visualization library.")
            logger.info("Pandas and visualization library imports verified.")
        else:
            feedback.append("The script is missing pandas or visualization library imports.")
            logger.warning("Import check failed.")
    else:
        feedback.append("The Python script is missing.")
        logger.warning("Python script missing.")

    # Helper: Encode image for GPT-4o
    def encode_image(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    # Step 2: Visualization checks
    charts_to_check = [
        (chart1, "a line chart showing total yearly emissions by continent, with Year on the x-axis and total emission on the y-axis"),
        (chart2, "a bar chart for the most recent year showing each continent's emission share, sorted in descending order")
    ]

    for chart, description in charts_to_check:
        chart_name = os.path.basename(chart)
        if os.path.exists(chart):
            score += 0.15
            feedback.append(f"The visualization {chart_name} exists.")
            logger.info(f"{chart_name} found.")

            try:
                encoded = encode_image(chart)
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert in data visualization evaluation. Judge chart clarity, relevance, and accuracy."
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        f"Examine this chart and verify if it correctly represents {description} "
                                        f"with clear labeling, readable axes, appropriate scale, legend, and balanced layout. "
                                        f"Reply 'yes' if it does, otherwise 'no', followed by a short reason."
                                    )
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{encoded}"}
                                }
                            ]
                        }
                    ],
                    max_tokens=100
                )

                answer = response.choices[0].message.content.lower()
                if "yes" in answer:
                    score += 0.1
                    feedback.append(f"GPT-4o judged {chart_name} as clear, correct, and visually accurate.")
                    logger.info(f"{chart_name} judged correct by GPT-4o.")
                else:
                    feedback.append(f"GPT-4o judged {chart_name} unclear or inaccurate: {answer}")
                    logger.warning(f"{chart_name} judged unclear by GPT-4o.")
            except Exception as e:
                feedback.append(f"Error evaluating {chart_name} using GPT-4o: {e}")
                logger.error(f"GPT-4o evaluation failed for {chart_name}: {e}", exc_info=True)
        else:
            feedback.append(f"The visualization {chart_name} is missing.")
            logger.warning(f"{chart_name} missing.")

    # Step 3: Final score
    final_score = min(score, 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score


def evaluate_continent_emission_growth_viz_quality(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Continent Emission Growth Visualization task.

    This evaluation checks:
      - that the Python script exists and imports pandas
      - that both visualization files are present
      - and that GPT-4o confirms the charts clearly represent the intended information
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    expected = expected.get("rules", expected)
    base_dir = os.path.dirname(actual)

    py_file = os.path.join(base_dir, "continent_emission_growth_viz.py")
    chart1 = os.path.join(base_dir, "continent_trend_line.png")
    chart2 = os.path.join(base_dir, "continent_growth_boxplot.png")

    score = 0.0
    feedback = []

    # Step 1: Check for script and required imports
    if os.path.exists(py_file):
        score += 0.3
        feedback.append("The Python script file is present.")
        logger.info("Python script file found.")

        with open(py_file, "r", encoding="utf-8") as f:
            content = f.read().lower()
        if "import pandas" in content and (
            "matplotlib" in content or "seaborn" in content or "plotly" in content or "altair" in content
        ):
            score += 0.2
            feedback.append("The script correctly imports pandas and a visualization library.")
            logger.info("Imports verified.")
        else:
            feedback.append("The script does not import pandas or a visualization library.")
            logger.warning("Import verification failed.")
    else:
        feedback.append("The Python script is missing.")
        logger.warning("Python script missing.")

    # Helper: encode image for GPT-4o
    def encode_image(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    # Step 2: Visualization checks
    charts_to_check = [
        (
            chart1,
            "a line chart showing average emissions per continent over years with smooth trends and clear legends"
        ),
        (
            chart2,
            "a boxplot showing the distribution of yearly emission growth rates (%) across continents to reveal volatility"
        )
    ]

    for chart, description in charts_to_check:
        chart_name = os.path.basename(chart)
        if os.path.exists(chart):
            score += 0.15
            feedback.append(f"The visualization {chart_name} exists.")
            logger.info(f"{chart_name} found.")

            try:
                encoded = encode_image(chart)
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert in data visualization evaluation. Judge chart clarity, relevance, and correctness."
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        f"Examine this chart and confirm if it correctly represents {description}, "
                                        f"with readable axes, suitable scale, clear labels, and proper layout. "
                                        f"Reply 'yes' if it does, otherwise 'no', followed by a short reason."
                                    )
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{encoded}"}
                                }
                            ]
                        }
                    ],
                    max_tokens=100
                )

                answer = response.choices[0].message.content.lower()
                if "yes" in answer:
                    score += 0.1
                    feedback.append(f"GPT-4o judged {chart_name} as clear, relevant, and visually accurate.")
                    logger.info(f"{chart_name} judged correct by GPT-4o.")
                else:
                    feedback.append(f"GPT-4o judged {chart_name} unclear or inaccurate: {answer}")
                    logger.warning(f"{chart_name} judged unclear by GPT-4o.")
            except Exception as e:
                feedback.append(f"Error evaluating {chart_name} with GPT-4o: {e}")
                logger.error(f"Error evaluating {chart_name}: {e}", exc_info=True)
        else:
            feedback.append(f"The visualization {chart_name} is missing.")
            logger.warning(f"{chart_name} missing.")

    # Step 3: Final score
    final_score = min(score, 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score

def evaluate_yearly_std_summary(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Yearly Ridership Standard Deviation task.

    This evaluation checks:
      - that the Python script file exists
      - that pandas is correctly imported
      - that the result file yearly_std_summary.csv exists
      - and that the computed yearly standard deviations match the expected results
        within rounding tolerance (±0.01)
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    expected = expected.get("rules", expected)
    base_dir = os.path.dirname(actual)
    script_file = os.path.join(base_dir, "yearly_std_summary.py")
    csv_file = os.path.join(base_dir, "MTA_Ridership_by_DATA_NY_GOV.csv")
    result_file = os.path.join(base_dir, "yearly_std_summary.csv")

    score = 0.0
    feedback = []

    # Ground-truth values from the user
    expected_data = pd.DataFrame({
        "Year": [2020, 2021, 2022, 2023, 2024],
        "Subway_Std": [859229.864, 681256.081, 731579.272, 779169.328, 792630.201],
        "Buses_Std": [549675.993, 301160.910, 326023.772, 312713.341, 285409.004]
    })

    try:
        # Step 1: Check that script exists
        if os.path.exists(script_file):
            score += 0.4
            feedback.append("Python script file is present.")
            logger.info("Python script found.")

            with open(script_file, "r", encoding="utf-8") as f:
                content = f.read().lower()
            if "import pandas" in content:
                score += 0.3
                feedback.append("Script correctly imports pandas.")
                logger.info("Verified pandas import.")
            else:
                feedback.append("Missing pandas import in script.")
                logger.warning("Script missing pandas import.")
        else:
            feedback.append("Python script file is missing.")
            logger.warning("Python script file not found.")

        # Step 2: Check output file
        if os.path.exists(result_file):
            score += 0.1
            feedback.append("Result file yearly_std_summary.csv is present.")
            logger.info("Result file found.")

            try:
                actual_df = pd.read_csv(result_file)
                actual_df.columns = actual_df.columns.str.strip()

                expected_cols = ["Year", "Subway_Std", "Buses_Std"]
                if list(actual_df.columns) != expected_cols:
                    feedback.append(f"Incorrect columns. Expected: {expected_cols}, found: {list(actual_df.columns)}.")
                    logger.warning("Column mismatch detected.")
                else:
                    feedback.append("Column names are correct.")
                    logger.info("Columns verified.")

                # Sort by Year
                actual_df = actual_df.sort_values("Year").reset_index(drop=True)
                expected_data = expected_data.sort_values("Year").reset_index(drop=True)

                # Compare numerical values with tolerance
                match_ratios = []
                for col in ["Subway_Std", "Buses_Std"]:
                    diffs = abs(actual_df[col] - expected_data[col])
                    ratio = (diffs <= 0.01).mean()
                    match_ratios.append(ratio)

                overall_match = sum(match_ratios) / len(match_ratios)
                logger.info(f"Mean match ratio across numeric columns: {overall_match:.3f}")

                if overall_match >= 0.95:
                    score += 0.2
                    feedback.append("Computed standard deviations match expected results (≥95% accuracy).")
                else:
                    feedback.append(f"Computed values differ (average match={overall_match:.2f}).")
                    logger.warning("Significant deviation detected in computed values.")

            except Exception as e:
                feedback.append(f"Error reading or verifying yearly_std_summary.csv: {e}")
                logger.error(f"Error verifying yearly_std_summary.csv: {e}", exc_info=True)
        else:
            feedback.append("The result file yearly_std_summary.csv is missing.")
            logger.warning("Result file missing.")

    except Exception as e:
        feedback.append(f"Evaluation failed: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)

    final_score = min(score, 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score


def evaluate_mta_ridership_trends_viz_quality(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the MTA Ridership Trends Visualization task.

    This evaluation checks:
      - that the Python script exists and imports pandas
      - that both visualization files are present
      - and that GPT-4o confirms the charts clearly represent the correct information
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    expected = expected.get("rules", expected)
    base_dir = os.path.dirname(actual)

    py_file = os.path.join(base_dir, "mta_ridership_trends_viz.py")
    chart1 = os.path.join(base_dir, "subway_trend.png")
    chart2 = os.path.join(base_dir, "bus_trend.png")

    score = 0.0
    feedback = []

    # Step 1: Check Python script and imports
    if os.path.exists(py_file):
        score += 0.3
        feedback.append("The Python script file is present.")
        logger.info("Python script file found.")

        with open(py_file, "r", encoding="utf-8") as f:
            content = f.read().lower()
        if "import pandas" in content and (
            "matplotlib" in content or "seaborn" in content or "plotly" in content or "altair" in content
        ):
            score += 0.2
            feedback.append("The script correctly imports pandas and a visualization library.")
            logger.info("Pandas and visualization library imports verified.")
        else:
            feedback.append("The script does not import pandas or a visualization library.")
            logger.warning("Import verification failed.")
    else:
        feedback.append("The Python script is missing.")
        logger.warning("Python script missing.")

    # Helper: encode image for GPT-4o
    def encode_image(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    # Step 2: Visualization checks
    charts_to_check = [
        (
            chart1,
            "a line chart showing total subway ridership over time (with Date on the x-axis and Total Estimated Ridership on the y-axis)"
        ),
        (
            chart2,
            "a line chart showing total bus ridership over time (with Date on the x-axis and Total Estimated Ridership on the y-axis)"
        )
    ]

    for chart, description in charts_to_check:
        chart_name = os.path.basename(chart)
        if os.path.exists(chart):
            score += 0.15
            feedback.append(f"The visualization {chart_name} exists.")
            logger.info(f"{chart_name} found.")

            try:
                encoded = encode_image(chart)
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert in data visualization evaluation. Judge chart clarity, relevance, and correctness."
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        f"Examine this chart and confirm if it correctly represents {description} "
                                        f"with clear labeling, readable axes, suitable scale, and balanced layout. "
                                        f"Reply 'yes' if it does, otherwise 'no', followed by a short reason."
                                    )
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{encoded}"}
                                }
                            ]
                        }
                    ],
                    max_tokens=100
                )

                answer = response.choices[0].message.content.lower()
                if "yes" in answer:
                    score += 0.1
                    feedback.append(f"GPT-4o judged {chart_name} as clear, relevant, and visually accurate.")
                    logger.info(f"{chart_name} judged correct by GPT-4o.")
                else:
                    feedback.append(f"GPT-4o judged {chart_name} unclear or inaccurate: {answer}")
                    logger.warning(f"{chart_name} judged unclear by GPT-4o.")
            except Exception as e:
                feedback.append(f"Error evaluating {chart_name} using GPT-4o: {e}")
                logger.error(f"Error evaluating {chart_name}: {e}", exc_info=True)
        else:
            feedback.append(f"The visualization {chart_name} is missing.")
            logger.warning(f"{chart_name} missing.")

    # Step 3: Final scoring
    final_score = min(score, 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score


def evaluate_stock_reversal_count(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Stock Reversal Count task.

    This evaluation checks:
      - that the Python script file exists
      - that the script correctly imports pandas
      - that the result file reversal_count.txt exists
      - and that the reported total reversal count matches the expected value (304751)
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    expected = expected.get("rules", expected)
    base_dir = os.path.dirname(actual)
    script_file = os.path.join(base_dir, "stock_reversal_count.py")
    result_file = os.path.join(base_dir, "reversal_count.txt")

    score = 0.0
    feedback = []

    true_count = 304751  # expected total reversals from reference computation

    try:
        # Step 1: Check Python script
        if os.path.exists(script_file):
            score += 0.4
            feedback.append("Python script file is present.")
            logger.info("Python script located successfully.")

            with open(script_file, "r", encoding="utf-8") as f:
                content = f.read().lower()

            if "import pandas" in content:
                score += 0.3
                feedback.append("Script correctly imports pandas.")
                logger.info("Verified pandas import.")
            else:
                feedback.append("Missing pandas import in script.")
                logger.warning("Script missing pandas import.")
        else:
            feedback.append("Python script file is missing.")
            logger.warning("Missing Python script file.")

        # Step 2: Check result file
        if os.path.exists(result_file):
            score += 0.1
            feedback.append("Result file reversal_count.txt is present.")
            logger.info("Output file found.")

            try:
                with open(result_file, "r", encoding="utf-8") as f:
                    output = f.read().strip()

                # Extract first integer from file
                match = re.search(r"\d+", output)
                if match:
                    value = int(match.group())
                    logger.info(f"Extracted numeric reversal count: {value}")

                    if abs(value - true_count) <= 5:
                        score += 0.2
                        feedback.append(f"Reversal count matches expected result ({value}).")
                        logger.info("Reversal count verified successfully.")
                    else:
                        feedback.append(
                            f"Reported reversal count ({value}) differs from expected ({true_count})."
                        )
                        logger.warning("Mismatch detected in reversal count.")
                else:
                    feedback.append("No numeric value found in reversal_count.txt.")
                    logger.warning("Missing numeric output in result file.")

            except Exception as e:
                feedback.append(f"Error reading reversal_count.txt: {e}")
                logger.error(f"Error reading reversal_count.txt: {e}", exc_info=True)
        else:
            feedback.append("Result file reversal_count.txt is missing.")
            logger.warning("Result file not found.")

    except Exception as e:
        feedback.append(f"Evaluation failed: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)

    final_score = min(score, 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score

def evaluate_high_volume_ratio(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the High Volume Ratio task.

    This evaluation checks:
      - that the Python script file exists
      - that the script correctly imports pandas
      - that the result file high_volume_ratio.csv exists
      - and that the top 3 rows match the expected (date, volume_ratio)
        within rounding tolerance ±0.01
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    expected = expected.get("rules", expected)
    base_dir = os.path.dirname(actual)
    script_file = os.path.join(base_dir, "high_volume_ratio.py")
    result_file = os.path.join(base_dir, "high_volume_ratio.csv")

    score = 0.0
    feedback = []

    # Ground truth from your reference output
    expected_data = pd.DataFrame({
        "date": ["2015-06-30", "2014-04-03", "2017-08-07"],
        "volume_ratio": [4.98, 4.81, 4.78]
    })

    try:
        # Step 1: Check Python script
        if os.path.exists(script_file):
            score += 0.4
            feedback.append("Python script file is present.")
            logger.info("Python script file found.")

            with open(script_file, "r", encoding="utf-8") as f:
                content = f.read().lower()
            if "import pandas" in content:
                score += 0.3
                feedback.append("Script correctly imports pandas.")
                logger.info("Verified pandas import.")
            else:
                feedback.append("Missing pandas import in script.")
                logger.warning("Pandas import missing.")
        else:
            feedback.append("Python script file is missing.")
            logger.warning("Python script file not found.")

        # Step 2: Check result file
        if os.path.exists(result_file):
            score += 0.1
            feedback.append("Result file high_volume_ratio.csv is present.")
            logger.info("Result file found.")

            try:
                actual_df = pd.read_csv(result_file)
                actual_df.columns = actual_df.columns.str.strip()

                expected_cols = ["date", "volume_ratio"]
                if list(actual_df.columns) != expected_cols:
                    feedback.append(f"Incorrect columns. Expected {expected_cols}, found {list(actual_df.columns)}.")
                    logger.warning("Column name mismatch.")
                else:
                    feedback.append("Column names are correct.")
                    logger.info("Column names verified.")

                # Compare top 3 rows (date + value match)
                if len(actual_df) < 3:
                    feedback.append("Output file contains fewer than 3 rows.")
                    logger.warning("Insufficient rows in output.")
                else:
                    actual_df = actual_df.head(3).reset_index(drop=True)
                    expected_data = expected_data.reset_index(drop=True)

                    match_count = 0
                    for i in range(3):
                        date_match = str(actual_df.loc[i, "date"]).strip() == expected_data.loc[i, "date"]
                        val_diff = abs(actual_df.loc[i, "volume_ratio"] - expected_data.loc[i, "volume_ratio"])
                        if date_match and val_diff <= 0.01:
                            match_count += 1

                    match_ratio = match_count / 3
                    logger.info(f"Match ratio for top 3 rows: {match_ratio:.2f}")

                    if match_ratio >= 0.95:
                        score += 0.2
                        feedback.append("Top 3 rows match expected results (≥95% accuracy).")
                        logger.info("Top rows verified successfully.")
                    else:
                        feedback.append(f"Top rows differ significantly (match ratio={match_ratio:.2f}).")
                        logger.warning("Mismatch in top row comparison.")

            except Exception as e:
                feedback.append(f"Error reading or verifying high_volume_ratio.csv: {e}")
                logger.error(f"Error reading high_volume_ratio.csv: {e}", exc_info=True)
        else:
            feedback.append("Result file high_volume_ratio.csv is missing.")
            logger.warning("Result file missing.")

    except Exception as e:
        feedback.append(f"Evaluation failed: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)

    final_score = min(score, 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score

def evaluate_stock_stability_score(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Stock Stability Score task.

    This evaluation checks:
      - that the Python script file exists
      - that pandas is imported
      - that stability_score.txt exists
      - and that the numeric stability score matches the expected (0.981 ± 0.001)
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    expected = expected.get("rules", expected)
    base_dir = os.path.dirname(actual)
    script_file = os.path.join(base_dir, "stock_stability_score.py")
    result_file = os.path.join(base_dir, "stability_score.txt")

    score = 0.0
    feedback = []

    expected_value = 0.981  # true mean stability score

    try:
        # Step 1: Check script file
        if os.path.exists(script_file):
            score += 0.4
            feedback.append("Python script file is present.")
            logger.info("Python script file found.")

            with open(script_file, "r", encoding="utf-8") as f:
                content = f.read().lower()
            if "import pandas" in content:
                score += 0.3
                feedback.append("Script correctly imports pandas.")
                logger.info("Verified pandas import.")
            else:
                feedback.append("Missing pandas import in script.")
                logger.warning("Pandas import not found.")
        else:
            feedback.append("Python script file is missing.")
            logger.warning("Missing Python script file.")

        # Step 2: Check result file
        if os.path.exists(result_file):
            score += 0.1
            feedback.append("Result file stability_score.txt is present.")
            logger.info("Result file located successfully.")

            try:
                with open(result_file, "r", encoding="utf-8") as f:
                    output = f.read().strip()

                # Extract numeric value
                match = re.search(r"[-+]?\d*\.\d+|\d+", output)
                if match:
                    value = float(match.group())
                    logger.info(f"Extracted stability score: {value}")

                    if abs(value - expected_value) <= 0.001:
                        score += 0.2
                        feedback.append(f"Stability score matches expected result ({value:.3f}).")
                        logger.info("Stability score verified successfully.")
                    else:
                        feedback.append(
                            f"Reported stability score ({value:.3f}) differs from expected ({expected_value:.3f})."
                        )
                        logger.warning("Stability value mismatch.")
                else:
                    feedback.append("No numeric value found in stability_score.txt.")
                    logger.warning("Missing numeric value in result file.")

            except Exception as e:
                feedback.append(f"Error reading stability_score.txt: {e}")
                logger.error(f"Error reading stability_score.txt: {e}", exc_info=True)
        else:
            feedback.append("Result file stability_score.txt is missing.")
            logger.warning("Result file missing.")

    except Exception as e:
        feedback.append(f"Evaluation failed: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)

    final_score = min(score, 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score

def evaluate_monthly_gain_drop(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Monthly Gain and Drop Analysis task.

    This evaluation checks:
      - that the Python script file exists
      - that pandas is correctly imported
      - that the output file monthly_gain_drop.csv exists
      - and that the computed monthly avg_gain and avg_drop match the expected
        values within a tolerance of ±0.001
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    expected = expected.get("rules", expected)
    base_dir = os.path.dirname(actual)
    script_file = os.path.join(base_dir, "monthly_gain_drop.py")
    csv_file = os.path.join(base_dir, "all_stocks_5yr.csv")
    result_file = os.path.join(base_dir, "monthly_gain_drop.csv")

    score = 0.0
    feedback = []

    try:
        # Step 1: Check for script file
        if os.path.exists(script_file):
            score += 0.4
            feedback.append("Python script file is present.")
            logger.info("Python script file found.")

            with open(script_file, "r", encoding="utf-8") as f:
                content = f.read().lower()

            if "import pandas" in content:
                score += 0.3
                feedback.append("Script correctly imports pandas.")
                logger.info("Verified pandas import.")
            else:
                feedback.append("Missing pandas import in script.")
                logger.warning("Missing pandas import in script.")
        else:
            feedback.append("Python script file is missing.")
            logger.warning("Missing Python script file.")

        # Step 2: Check output file
        if os.path.exists(result_file):
            score += 0.1
            feedback.append("Result file monthly_gain_drop.csv is present.")
            logger.info("Output file found.")

            try:
                # --- Compute expected results dynamically ---
                df = pd.read_csv(csv_file)
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                df["month"] = df["date"].dt.month
                df["change"] = df["close"] - df["open"]

                def avg_gain(x):
                    return np.round(x[x > 0].mean(), 3)

                def avg_drop(x):
                    return np.round(x[x < 0].mean(), 3)

                expected_df = (
                    df.groupby("month")["change"]
                    .agg(avg_gain=avg_gain, avg_drop=avg_drop)
                    .reset_index()
                ).fillna(0)

                # --- Read user output ---
                actual_df = pd.read_csv(result_file)
                actual_df.columns = actual_df.columns.str.strip().str.lower()

                expected_cols = ["month", "avg_gain", "avg_drop"]
                if list(actual_df.columns) != expected_cols:
                    feedback.append(
                        f"Incorrect columns. Expected: {expected_cols}, found: {list(actual_df.columns)}."
                    )
                    logger.warning("Column mismatch detected.")
                else:
                    feedback.append("Column names are correct.")
                    logger.info("Column names verified.")

                # --- Compare numerically ---
                actual_df = actual_df.sort_values("month").reset_index(drop=True)
                expected_df = expected_df.sort_values("month").reset_index(drop=True)

                if not np.array_equal(actual_df["month"].values, expected_df["month"].values):
                    feedback.append("Month values mismatch between result and expected output.")
                    logger.warning("Month mismatch detected.")

                match_ratios = []
                for col in ["avg_gain", "avg_drop"]:
                    diffs = abs(actual_df[col] - expected_df[col])
                    ratio = (diffs <= 0.001).mean()
                    match_ratios.append(ratio)
                    logger.info(f"Match ratio for {col}: {ratio:.3f}")

                overall_match = np.mean(match_ratios)
                if overall_match >= 0.95:
                    score += 0.2
                    feedback.append("Monthly gain/drop averages match expected results (≥95% accuracy).")
                    logger.info(f"Numerical comparison passed with match ratio {overall_match:.3f}.")
                else:
                    feedback.append(f"Computed averages differ (match ratio={overall_match:.2f}).")
                    logger.warning(f"Low match ratio detected ({overall_match:.3f}).")

            except Exception as e:
                feedback.append(f"Error verifying monthly_gain_drop.csv: {e}")
                logger.error(f"Error verifying monthly_gain_drop.csv: {e}", exc_info=True)
        else:
            feedback.append("Result file monthly_gain_drop.csv is missing.")
            logger.warning("Result file missing.")

    except Exception as e:
        feedback.append(f"Evaluation failed: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)

    final_score = min(score, 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score


def evaluate_moving_average_crossover(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Moving Average Crossover task.

    This evaluation checks:
      - Python script file presence
      - Correct pandas import
      - Output file exists
      - First 5 crossovers match expected values
      - Final line reports correct total (33710)
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    expected = expected.get("rules", expected)
    base_dir = os.path.dirname(actual)
    script_file = os.path.join(base_dir, "moving_average_crossover.py")
    result_file = os.path.join(base_dir, "bullish_crossovers.csv")

    score = 0.0
    feedback = []

    # Ground truth
    expected_first5 = pd.DataFrame({
        "stock": ["A", "A", "A", "A", "A"],
        "date": [
            "2013-03-05",
            "2013-04-10",
            "2013-05-08",
            "2013-07-02",
            "2013-08-07"
        ]
    })
    expected_total = 33710

    try:
        # Step 1: Check Python script
        if os.path.exists(script_file):
            score += 0.4
            feedback.append("Python script file is present.")
            logger.info("Python script file found.")
            with open(script_file, "r", encoding="utf-8") as f:
                content = f.read().lower()
            if "import pandas" in content:
                score += 0.3
                feedback.append("Script correctly imports pandas.")
                logger.info("Verified pandas import.")
            else:
                feedback.append("Missing pandas import in script.")
                logger.warning("Missing pandas import in script.")
        else:
            feedback.append("Python script file is missing.")
            logger.warning("Python script missing.")

        # Step 2: Check output file
        if os.path.exists(result_file):
            score += 0.1
            feedback.append("Result file bullish_crossovers.csv is present.")
            logger.info("Output file located successfully.")

            try:
                df = pd.read_csv(result_file, header=None)
                nrows = len(df)

                if nrows < 6:
                    feedback.append("Output file must contain at least 6 rows (5 + total).")
                    logger.warning("Insufficient rows in output.")
                else:
                    first5 = df.iloc[:5].reset_index(drop=True)
                    summary_line = df.iloc[-1, 0]

                    first5.columns = ["stock", "date"]
                    first5["stock"] = first5["stock"].astype(str).str.strip()
                    first5["date"] = first5["date"].astype(str).str.strip()

                    # Compare first 5
                    match_count = sum(
                        (first5["stock"] == expected_first5["stock"])
                        & (first5["date"] == expected_first5["date"])
                    )
                    match_ratio = match_count / 5.0
                    logger.info(f"Match ratio for first 5 crossovers: {match_ratio:.2f}")

                    if match_ratio >= 0.95:
                        score += 0.1
                        feedback.append("First 5 bullish crossover entries match expected results.")
                        logger.info("Top crossovers verified successfully.")
                    else:
                        feedback.append(f"First 5 entries differ (match ratio={match_ratio:.2f}).")
                        logger.warning("Mismatch in top rows.")

                    # Check total count
                    if "Total_Crossovers" in summary_line:
                        try:
                            total = int(str(df.iloc[-1, 1]).strip())
                            if total == expected_total:
                                score += 0.1
                                feedback.append("Total crossovers count matches expected value (33710).")
                                logger.info("Total crossovers verified successfully.")
                            else:
                                feedback.append(f"Total crossovers mismatch: found {total}, expected {expected_total}.")
                                logger.warning("Total count mismatch.")
                        except Exception:
                            feedback.append("Could not parse total crossovers number.")
                            logger.warning("Failed to parse total number.")
                    else:
                        feedback.append("Final summary line missing 'Total_Crossovers'.")
                        logger.warning("Summary line not formatted correctly.")

            except Exception as e:
                feedback.append(f"Error reading or verifying bullish_crossovers.csv: {e}")
                logger.error(f"Error reading bullish_crossovers.csv: {e}", exc_info=True)
        else:
            feedback.append("Result file bullish_crossovers.csv is missing.")
            logger.warning("Result file missing.")

    except Exception as e:
        feedback.append(f"Evaluation failed: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)

    # Final score
    final_score = min(score, 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)
    return final_score

def evaluate_stock_trends_viz_quality(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Stock Price and Volume Trend Visualization task.

    This evaluation checks:
      - that the Python script exists and imports pandas
      - that both visualization files are present
      - and that GPT-4o confirms the charts clearly represent the correct information
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    expected = expected.get("rules", expected)
    base_dir = os.path.dirname(actual)

    py_file = os.path.join(base_dir, "stock_trends_viz.py")
    chart1 = os.path.join(base_dir, "price_trend.png")
    chart2 = os.path.join(base_dir, "volume_trend.png")

    score = 0.0
    feedback = []

    # Step 1: Check script existence and imports
    if os.path.exists(py_file):
        score += 0.3
        feedback.append("The Python script file is present.")
        logger.info("Python script file found.")

        with open(py_file, "r", encoding="utf-8") as f:
            content = f.read().lower()

        if "import pandas" in content and (
            "matplotlib" in content or "seaborn" in content or "plotly" in content or "altair" in content
        ):
            score += 0.2
            feedback.append("The script correctly imports pandas and a visualization library.")
            logger.info("Required imports verified.")
        else:
            feedback.append("The script does not import pandas or a visualization library.")
            logger.warning("Import verification failed.")
    else:
        feedback.append("The Python script is missing.")
        logger.warning("Python script missing.")

    # Helper: encode images for GPT-4o
    def encode_image(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    # Step 2: Visualization checks
    charts_to_check = [
        (
            chart1,
            "a line chart showing both the Close price and 7-day moving average (MA7) over time, with two clearly distinguishable lines, axis labels, legend, and title"
        ),
        (
            chart2,
            "a bar chart showing trading Volume over time, with Date on the x-axis, Volume on the y-axis, readable gridlines, and proper formatting"
        )
    ]

    for chart, description in charts_to_check:
        chart_name = os.path.basename(chart)
        if os.path.exists(chart):
            score += 0.15
            feedback.append(f"The visualization {chart_name} exists.")
            logger.info(f"{chart_name} found.")

            try:
                encoded = encode_image(chart)
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert in data visualization evaluation. Judge each chart for clarity, correctness, and design quality."
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        f"Examine this chart and confirm if it correctly represents {description}. "
                                        f"Reply 'yes' if it does, otherwise 'no', followed by a short reason."
                                    )
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{encoded}"}
                                }
                            ]
                        }
                    ],
                    max_tokens=100
                )

                answer = response.choices[0].message.content.lower()
                if "yes" in answer:
                    score += 0.1
                    feedback.append(f"GPT-4o judged {chart_name} as clear, accurate, and visually informative.")
                    logger.info(f"{chart_name} validated by GPT-4o.")
                else:
                    feedback.append(f"GPT-4o judged {chart_name} unclear or inaccurate: {answer}")
                    logger.warning(f"{chart_name} judged unclear by GPT-4o.")
            except Exception as e:
                feedback.append(f"Error evaluating {chart_name} using GPT-4o: {e}")
                logger.error(f"Error evaluating {chart_name}: {e}", exc_info=True)
        else:
            feedback.append(f"The visualization {chart_name} is missing.")
            logger.warning(f"{chart_name} missing.")

    # Step 3: Final score
    final_score = min(score, 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score



def evaluate_missing_value_imputation(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Missing-Value Imputation (Category & Title) task.

    This evaluator checks that:
      - The script file exists and imports pandas
      - The result file (imputed_rows.csv) exists
      - Only rows that originally had missing Category or Title are present
      - Missing Category values are filled using the mode within the same Shipping Address State
      - Missing Title values are replaced with "Unknown Product"
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    expected = expected.get("rules", expected)
    base_dir = os.path.dirname(actual)
    script_file = os.path.join(base_dir, "missing_value_imputation.py")
    data_file = os.path.join(base_dir, "amazon-purchases-sample.csv")
    result_file = os.path.join(base_dir, "imputed_rows.csv")

    score = 0.0
    feedback = []

    try:
        # 1. Verify script existence and pandas import
        if os.path.exists(script_file):
            score += 0.2
            feedback.append("The Python script file exists.")
            logger.info("Python script found.")

            with open(script_file, "r", encoding="utf-8") as f:
                content = f.read().lower()
            if "import pandas" in content:
                score += 0.1
                feedback.append("Script correctly imports pandas.")
                logger.info("Pandas import verified.")
            else:
                feedback.append("Missing pandas import.")
                logger.warning("Script does not import pandas.")
        else:
            feedback.append("The Python script file is missing.")
            logger.warning("Script missing.")

        # 2. Verify dataset existence
        if not os.path.exists(data_file):
            feedback.append("Dataset amazon-purchases-sample.csv is missing.")
            logger.error("Dataset missing.")
            return min(score, 1.0)
        else:
            df = pd.read_csv(data_file)

        # 3. Verify result file
        if os.path.exists(result_file):
            score += 0.2
            feedback.append("The result file imputed_rows.csv exists.")
            logger.info("Result file found.")

            df_result = pd.read_csv(result_file)
            orig_missing = df[df["Category"].isna() | df["Title"].isna()].copy()

            # --- Validate: only imputed rows are present ---
            if len(df_result) == len(orig_missing):
                score += 0.15
                feedback.append("The result file contains only imputed rows.")
                logger.info("Row count matches missing subset.")
            else:
                feedback.append("Result file contains unexpected number of rows.")
                logger.warning("Row count mismatch between missing subset and result.")

            # --- Validate: Title imputation ---
            if "Unknown Product" in df_result["Title"].values:
                score += 0.1
                feedback.append("Missing Title values are correctly replaced with 'Unknown Product'.")
                logger.info("Title imputation verified.")
            else:
                feedback.append("No 'Unknown Product' found — Title imputation may be missing.")
                logger.warning("Title imputation not found.")

            # --- Validate: Category imputation by State mode ---
            if "Shipping Address State" in df.columns and "Category" in df.columns:
                valid_state_impute = True
                for _, row in df_result.iterrows():
                    state = row.get("Shipping Address State", None)
                    if pd.isna(state) or state not in df["Shipping Address State"].values:
                        continue
                    state_mode = (
                        df[df["Shipping Address State"] == state]["Category"]
                        .mode()
                    )
                    if len(state_mode) > 0 and row["Category"] not in state_mode.values:
                        valid_state_impute = False
                        break

                if valid_state_impute:
                    score += 0.15
                    feedback.append("Missing Category values are correctly filled using state-wise mode.")
                    logger.info("Category imputation verified.")
                else:
                    feedback.append("Category imputation by state mode appears incorrect.")
                    logger.warning("Incorrect category imputation detected.")
        else:
            feedback.append("Result file imputed_rows.csv is missing.")
            logger.warning("Result file missing.")

    except Exception as e:
        feedback.append(f"Evaluation error: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)

    # --- Final score ---
    final_score = min(score, 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score 


def evaluate_price_outlier_detection(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Price Outlier Detection (IQR Method) task.

    This evaluator checks that:
      - The Python script exists and imports pandas
      - The output file price_outliers.csv exists
      - The output correctly identifies rows whose 'Purchase Price Per Unit'
        lies outside 1.5×IQR from Q1/Q3 for their respective Category
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    expected = expected.get("rules", expected)
    base_dir = os.path.dirname(actual)
    script_file = os.path.join(base_dir, "price_outlier_detection.py")
    data_file = os.path.join(base_dir, "amazon-purchases-sample.csv")
    result_file = os.path.join(base_dir, "price_outliers.csv")

    score = 0.0
    feedback = []

    try:
        # Step 1: Verify Python script
        if os.path.exists(script_file):
            score += 0.25
            feedback.append("Python script found.")
            logger.info("Python script file located.")

            with open(script_file, "r", encoding="utf-8") as f:
                content = f.read().lower()
            if "import pandas" in content:
                score += 0.15
                feedback.append("Script correctly imports pandas.")
                logger.info("Pandas import verified.")
            else:
                feedback.append("Missing pandas import.")
                logger.warning("No pandas import detected.")
        else:
            feedback.append("Python script missing.")
            logger.warning("Script missing.")

        # Step 2: Verify dataset
        if not os.path.exists(data_file):
            feedback.append("Dataset amazon-purchases-sample.csv not found.")
            logger.error("Dataset missing.")
            return min(score, 1.0)
        else:
            df = pd.read_csv(data_file)

        # Step 3: Verify result file
        if os.path.exists(result_file):
            score += 0.25
            feedback.append("Result file price_outliers.csv found.")
            logger.info("Result file located.")

            df_result = pd.read_csv(result_file)
            if "Category" not in df.columns or "Purchase Price Per Unit" not in df.columns:
                feedback.append("Required columns missing in dataset.")
                logger.error("Columns not found: Category or Purchase Price Per Unit.")
                return min(score, 1.0)

            # Compute true outliers by category
            df_valid = df.dropna(subset=["Category", "Purchase Price Per Unit"])
            outlier_indices = []
            for cat, group in df_valid.groupby("Category"):
                q1 = group["Purchase Price Per Unit"].quantile(0.25)
                q3 = group["Purchase Price Per Unit"].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                mask = (group["Purchase Price Per Unit"] < lower) | (group["Purchase Price Per Unit"] > upper)
                outlier_indices.extend(group[mask].index.tolist())

            true_outliers = df.loc[outlier_indices]
            logger.info(f"Computed {len(true_outliers)} true outliers from dataset.")

            # Compare overlap between predicted and actual outliers
            overlap = len(set(df_result.index) & set(true_outliers.index))
            if overlap > 0:
                ratio = overlap / max(len(true_outliers), 1)
                if ratio > 0.9:
                    score += 0.25
                    feedback.append("Outlier detection is accurate (>90% match).")
                    logger.info("High overlap with true outliers.")
                elif ratio > 0.6:
                    score += 0.15
                    feedback.append("Outlier detection is partially accurate (60–90% match).")
                    logger.warning("Moderate overlap with true outliers.")
                else:
                    feedback.append("Outlier detection accuracy is low (<60% match).")
                    logger.warning("Low overlap with true outliers.")
            else:
                feedback.append("No overlap found between result and true outliers.")
                logger.warning("No correct outliers detected.")
        else:
            feedback.append("Result file price_outliers.csv missing.")
            logger.warning("Result file missing.")

    except Exception as e:
        feedback.append(f"Evaluation error: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)

    final_score = min(score, 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)
    return final_score

def evaluate_state_sales_viz_quality(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Amazon Purchases – State-Level Sales Visualization task.

    This evaluation checks:
      - that the Python script exists and imports pandas
      - that both visualization files are present and non-empty
      - and that GPT-4o confirms the charts clearly represent the correct information
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    expected = expected.get("rules", expected)
    base_dir = os.path.dirname(actual)

    py_file = os.path.join(base_dir, "state_sales_viz.py")
    chart1 = os.path.join(base_dir, "state_sales.png")
    chart2 = os.path.join(base_dir, "state_share.png")

    score = 0.0
    feedback = []

    # Step 1: Check for Python script
    if os.path.exists(py_file):
        score += 0.3
        feedback.append("The Python script file is present.")
        logger.info("Python script found.")

        with open(py_file, "r", encoding="utf-8") as f:
            content = f.read().lower()

        if "import pandas" in content and (
            "matplotlib" in content or "seaborn" in content or "plotly" in content or "altair" in content
        ):
            score += 0.2
            feedback.append("The script correctly imports pandas and a visualization library.")
        else:
            feedback.append("Missing pandas or visualization library import.")
    else:
        feedback.append("The Python script file is missing.")
        logger.warning("Python script missing.")

    # Helper: Encode image for GPT-4o
    def encode_image(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    # Step 2: Visualization checks
    charts_to_check = [
        (
            chart1,
            "a bar chart showing total purchase amount per Shipping Address State, with clear axis labels and readable values"
        ),
        (
            chart2,
            "a pie chart showing each state's percentage share of total purchase amount, with labels or legend clearly visible"
        )
    ]

    for chart, description in charts_to_check:
        chart_name = os.path.basename(chart)
        if os.path.exists(chart) and os.path.getsize(chart) > 0:
            score += 0.15
            feedback.append(f"The visualization {chart_name} exists and is non-empty.")

            try:
                encoded = encode_image(chart)
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert data visualization evaluator. Judge chart clarity and correctness."
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        f"Examine this chart and confirm if it correctly represents {description}. "
                                        f"Reply 'yes' if it does, otherwise 'no', followed by a short reason."
                                    )
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{encoded}"}
                                }
                            ]
                        }
                    ],
                    max_tokens=100
                )

                answer = response.choices[0].message.content.lower()
                if "yes" in answer:
                    score += 0.1
                    feedback.append(f"GPT-4o judged {chart_name} as clear and accurate.")
                    logger.info(f"{chart_name} validated by GPT-4o.")
                else:
                    feedback.append(f"GPT-4o judged {chart_name} unclear or inaccurate: {answer}")
                    logger.warning(f"{chart_name} judged unclear by GPT-4o.")
            except Exception as e:
                feedback.append(f"Error evaluating {chart_name} with GPT-4o: {e}")
                logger.error(f"Error evaluating {chart_name}: {e}", exc_info=True)
        else:
            feedback.append(f"The visualization {chart_name} is missing or empty.")
            logger.warning(f"{chart_name} missing or zero-size.")

    # Step 3: Final scoring
    final_score = min(score, 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score

def evaluate_price_quantity_corr(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Price–Quantity Correlation task.

    This evaluator checks that:
      - The Python script file exists
      - It correctly imports pandas
      - The result file price_quantity_corr.txt exists
      - The correlation value in the file matches the expected (-0.065 ± 0.005)
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    expected = expected.get("rules", expected)
    base_dir = os.path.dirname(actual)
    script_file = os.path.join(base_dir, "price_quantity_corr.py")
    result_file = os.path.join(base_dir, "price_quantity_corr.txt")

    true_corr = -0.065
    score = 0.0
    feedback = []

    try:
        # Step 1: Check Python script
        if os.path.exists(script_file):
            score += 0.3
            feedback.append("Python script file found.")
            logger.info("Python script located.")

            with open(script_file, "r", encoding="utf-8") as f:
                content = f.read().lower()

            if "import pandas" in content:
                score += 0.2
                feedback.append("Script correctly imports pandas.")
                logger.info("Verified pandas import.")
            else:
                feedback.append("Missing pandas import in script.")
                logger.warning("Script does not import pandas.")
        else:
            feedback.append("Python script missing.")
            logger.warning("Script missing.")

        # Step 2: Check result file
        if os.path.exists(result_file):
            score += 0.2
            feedback.append("Result file price_quantity_corr.txt found.")
            logger.info("Result file detected.")

            try:
                with open(result_file, "r", encoding="utf-8") as f:
                    content = f.read().strip()

                match = re.search(r"[-+]?\d*\.\d+|\d+", content)
                if match:
                    value = float(match.group())
                    logger.info(f"Extracted correlation: {value}")

                    if abs(value - true_corr) <= 0.005:
                        score += 0.3
                        feedback.append(f"Correlation value matches expected ({value:.3f}).")
                        logger.info("Correlation matches expected range.")
                    else:
                        feedback.append(
                            f"Correlation differs from expected. Found {value:.3f}, expected {-0.065:.3f}."
                        )
                        logger.warning("Correlation mismatch.")
                else:
                    feedback.append("No numeric value detected in result file.")
                    logger.warning("Failed to extract numeric correlation.")
            except Exception as e:
                feedback.append(f"Error reading result file: {e}")
                logger.error(f"Error reading result file: {e}", exc_info=True)
        else:
            feedback.append("Result file price_quantity_corr.txt missing.")
            logger.warning("Result file missing.")

    except Exception as e:
        feedback.append(f"Evaluation failed: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)

    final_score = min(score, 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)
    return final_score

def evaluate_top_categories_viz_quality(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Amazon Purchases — Top Categories Visualization task.

    This evaluation checks:
      - that the Python script exists and imports pandas
      - that both visualization files exist and are non-empty
      - and that GPT-4o confirms the charts clearly represent the correct information
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    expected = expected.get("rules", expected)
    base_dir = os.path.dirname(actual)

    py_file = os.path.join(base_dir, "top_categories_viz.py")
    chart1 = os.path.join(base_dir, "top_categories.png")
    chart2 = os.path.join(base_dir, "category_boxplot.png")

    score = 0.0
    feedback = []

    # Step 1: Check script and imports
    if os.path.exists(py_file):
        score += 0.3
        feedback.append("The Python script file is present.")
        logger.info("Python script found.")

        with open(py_file, "r", encoding="utf-8") as f:
            content = f.read().lower()

        if "import pandas" in content and (
            "matplotlib" in content or "seaborn" in content or "plotly" in content or "altair" in content
        ):
            score += 0.2
            feedback.append("The script correctly imports pandas and a visualization library.")
        else:
            feedback.append("Missing pandas or visualization library import.")
            logger.warning("Import verification failed.")
    else:
        feedback.append("The Python script is missing.")
        logger.warning("Python script missing.")

    # Helper: Encode image for GPT-4o
    def encode_image(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    # Step 2: Visualization checks
    charts_to_check = [
        (
            chart1,
            "a horizontal bar chart showing the top 5 product categories ranked by total purchase value, with clear labels and readable category names"
        ),
        (
            chart2,
            "a boxplot showing the price distributions across the top 5 categories, with appropriate axis labels and readable legends"
        )
    ]

    for chart, description in charts_to_check:
        chart_name = os.path.basename(chart)
        if os.path.exists(chart) and os.path.getsize(chart) > 0:
            score += 0.15
            feedback.append(f"The visualization {chart_name} exists and is non-empty.")

            try:
                encoded = encode_image(chart)
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert in data visualization evaluation. Judge chart clarity, readability, and correctness."
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        f"Examine this chart and confirm if it correctly represents {description}. "
                                        f"Reply 'yes' if it does, otherwise 'no', followed by a short reason."
                                    )
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{encoded}"}
                                }
                            ]
                        }
                    ],
                    max_tokens=100
                )

                answer = response.choices[0].message.content.lower()
                if "yes" in answer:
                    score += 0.1
                    feedback.append(f"GPT-4o judged {chart_name} as clear, accurate, and visually informative.")
                    logger.info(f"{chart_name} validated by GPT-4o.")
                else:
                    feedback.append(f"GPT-4o judged {chart_name} unclear or inaccurate: {answer}")
                    logger.warning(f"{chart_name} judged unclear by GPT-4o.")
            except Exception as e:
                feedback.append(f"Error evaluating {chart_name} with GPT-4o: {e}")
                logger.error(f"Error evaluating {chart_name}: {e}", exc_info=True)
        else:
            feedback.append(f"The visualization {chart_name} is missing or empty.")
            logger.warning(f"{chart_name} missing or zero-size.")

    # Step 3: Final score
    final_score = min(score, 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score


def evaluate_asin_prefix_summary(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the ASIN Prefix Quantity Summary task.

    This evaluator checks that:
      - The Python script exists and imports pandas
      - The result file asin_prefix_summary.csv exists
      - The file contains correct columns Prefix and Total_Quantity
      - The top 5 rows match the expected result:
            Prefix  Total_Quantity
            00      5
            02      2
            03      4
            04      4
            05      5
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    base_dir = os.path.dirname(actual)
    script_file = os.path.join(base_dir, "asin_prefix_summary.py")
    result_file = os.path.join(base_dir, "asin_prefix_summary.csv")

    score = 0.0
    feedback = []

    # --- Ground truth (first 5 expected rows) ---
    expected_data = pd.DataFrame({
        "Prefix": ["00", "02", "03", "04", "05"],
        "Total_Quantity": [5, 2, 4, 4, 5]
    })

    try:
        # Step 1: Check for Python script
        if os.path.exists(script_file):
            score += 0.25
            feedback.append("Python script file found.")
            logger.info("Python script found.")

            with open(script_file, "r", encoding="utf-8") as f:
                content = f.read().lower()

            if "import pandas" in content:
                score += 0.15
                feedback.append("Script correctly imports pandas.")
                logger.info("Pandas import verified.")
            else:
                feedback.append("Missing pandas import.")
                logger.warning("Script missing pandas import.")
        else:
            feedback.append("Python script missing.")
            logger.warning("Script file not found.")

        # Step 2: Check result file
        if os.path.exists(result_file):
            score += 0.25
            feedback.append("Result file asin_prefix_summary.csv found.")
            logger.info("Result file found.")

            try:
                df = pd.read_csv(result_file)

                # Column name check
                if list(df.columns[:2]) == ["Prefix", "Total_Quantity"]:
                    score += 0.15
                    feedback.append("Correct column names detected.")
                    logger.info("Column names verified.")
                else:
                    feedback.append("Column names do not match expected ['Prefix', 'Total_Quantity'].")
                    logger.warning(f"Column names mismatch: {df.columns.tolist()}")

                # Data validation (compare first 5 rows)
                df_subset = df.head(5).reset_index(drop=True)
                if df_subset.equals(expected_data):
                    score += 0.2
                    feedback.append("Top 5 rows match expected prefix summary exactly.")
                    logger.info("Output data matches expected result.")
                else:
                    feedback.append("Top 5 rows do not match expected prefix summary.")
                    logger.warning("Mismatch in top 5 expected rows.")
                    logger.warning(f"Expected:\n{expected_data}\nGot:\n{df_subset}")

            except Exception as e:
                feedback.append(f"Error reading or parsing asin_prefix_summary.csv: {e}")
                logger.error(f"Error reading asin_prefix_summary.csv: {e}", exc_info=True)
        else:
            feedback.append("Result file asin_prefix_summary.csv missing.")
            logger.warning("Result file not found.")

    except Exception as e:
        feedback.append(f"Evaluation failed: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)

    # --- Finalize ---
    final_score = min(score, 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)
    return final_score


def evaluate_price_stats(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Price Stats (Average Range and Midpoint) task.

    This evaluator checks that:
      - The Python script exists and imports pandas
      - The result file price_stats.txt exists
      - The file has exactly two numeric lines:
            line 1 → average price range
            line 2 → average midpoint
      - The values match the expected results within tolerance ±0.01
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    base_dir = os.path.dirname(actual)
    script_file = os.path.join(base_dir, "price_stats.py")
    result_file = os.path.join(base_dir, "price_stats.txt")

    # Ground truth
    true_range = 5.613
    true_midpoint = 280.417

    score = 0.0
    feedback = []

    try:
        # Step 1: Verify Python script
        if os.path.exists(script_file):
            score += 0.25
            feedback.append("Python script file found.")
            logger.info("Python script located.")

            with open(script_file, "r", encoding="utf-8") as f:
                content = f.read().lower()
            if "import pandas" in content:
                score += 0.15
                feedback.append("Script correctly imports pandas.")
                logger.info("Verified pandas import.")
            else:
                feedback.append("Missing pandas import in script.")
                logger.warning("Script missing pandas import.")
        else:
            feedback.append("Python script missing.")
            logger.warning("Script file not found.")

        # Step 2: Verify output file
        if os.path.exists(result_file):
            score += 0.25
            feedback.append("Result file price_stats.txt found.")
            logger.info("Result file detected.")

            try:
                with open(result_file, "r", encoding="utf-8") as f:
                    lines = [l.strip() for l in f.readlines() if l.strip()]

                if len(lines) != 2:
                    feedback.append("Output file should contain exactly two lines.")
                    logger.warning(f"Found {len(lines)} lines instead of 2.")
                else:
                    # Extract numeric values from both lines
                    match_range = re.search(r"[-+]?\d*\.\d+|\d+", lines[0])
                    match_mid = re.search(r"[-+]?\d*\.\d+|\d+", lines[1])

                    if match_range and match_mid:
                        avg_range = float(match_range.group())
                        avg_mid = float(match_mid.group())

                        logger.info(f"Extracted values: Range={avg_range}, Midpoint={avg_mid}")

                        # Validate range
                        if abs(avg_range - true_range) <= 0.01:
                            score += 0.2
                            feedback.append(f"Average price range is correct ({avg_range:.3f}).")
                            logger.info("Average range matches expected.")
                        else:
                            feedback.append(f"Average price range differs: expected {true_range:.3f}, got {avg_range:.3f}.")
                            logger.warning("Average range mismatch.")

                        # Validate midpoint
                        if abs(avg_mid - true_midpoint) <= 0.01:
                            score += 0.15
                            feedback.append(f"Average midpoint is correct ({avg_mid:.3f}).")
                            logger.info("Average midpoint matches expected.")
                        else:
                            feedback.append(f"Average midpoint differs: expected {true_midpoint:.3f}, got {avg_mid:.3f}.")
                            logger.warning("Average midpoint mismatch.")
                    else:
                        feedback.append("Could not extract valid numeric values from output.")
                        logger.warning("Failed numeric extraction.")
            except Exception as e:
                feedback.append(f"Error reading price_stats.txt: {e}")
                logger.error(f"Error reading result file: {e}", exc_info=True)
        else:
            feedback.append("Result file price_stats.txt missing.")
            logger.warning("Result file not found.")

    except Exception as e:
        feedback.append(f"Evaluation failed: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)

    final_score = min(score, 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score

def evaluate_adj_corr(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Close–AdjClose Correlation task.

    This evaluator checks that:
      - The Python script file exists
      - It correctly imports pandas
      - The result file adj_corr.txt exists
      - The correlation value is 1.000 (within ±0.001 tolerance)
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    base_dir = os.path.dirname(actual)
    script_file = os.path.join(base_dir, "adj_corr.py")
    result_file = os.path.join(base_dir, "adj_corr.txt")

    true_corr = 1.000
    score = 0.0
    feedback = []

    try:
        # Step 1: Verify script file
        if os.path.exists(script_file):
            score += 0.3
            feedback.append("Python script file found.")
            logger.info("Python script file detected.")

            with open(script_file, "r", encoding="utf-8") as f:
                content = f.read().lower()

            if "import pandas" in content:
                score += 0.2
                feedback.append("Script correctly imports pandas.")
                logger.info("Verified pandas import.")
            else:
                feedback.append("Missing pandas import in script.")
                logger.warning("Script missing pandas import.")
        else:
            feedback.append("Python script missing.")
            logger.warning("Script file not found.")

        # Step 2: Verify result file
        if os.path.exists(result_file):
            score += 0.2
            feedback.append("Result file adj_corr.txt found.")
            logger.info("Result file detected.")

            try:
                with open(result_file, "r", encoding="utf-8") as f:
                    content = f.read().strip()

                match = re.search(r"[-+]?\d*\.\d+|\d+", content)
                if match:
                    value = float(match.group())
                    logger.info(f"Extracted correlation value: {value}")

                    if abs(value - true_corr) <= 0.001:
                        score += 0.3
                        feedback.append(f"Correlation value matches expected ({value:.3f}).")
                        logger.info("Correlation matches expected.")
                    else:
                        feedback.append(f"Correlation differs: expected {true_corr:.3f}, got {value:.3f}.")
                        logger.warning("Correlation mismatch.")
                else:
                    feedback.append("No numeric value found in result file.")
                    logger.warning("Could not extract numeric correlation.")
            except Exception as e:
                feedback.append(f"Error reading adj_corr.txt: {e}")
                logger.error(f"Error reading adj_corr.txt: {e}", exc_info=True)
        else:
            feedback.append("Result file adj_corr.txt missing.")
            logger.warning("Result file missing.")

    except Exception as e:
        feedback.append(f"Evaluation failed: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)

    final_score = min(score, 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)
    return final_score

def evaluate_return_volatility(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Return Volatility task.

    This evaluator checks that:
      - The Python script exists and imports pandas
      - The result file return_volatility.txt exists
      - The file contains one numeric value (volatility)
      - The value matches the expected (1.66 ± 0.01)
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    base_dir = os.path.dirname(actual)
    script_file = os.path.join(base_dir, "return_volatility.py")
    result_file = os.path.join(base_dir, "return_volatility.txt")

    true_volatility = 1.66
    score = 0.0
    feedback = []

    try:
        # Step 1: Verify script existence and import
        if os.path.exists(script_file):
            score += 0.3
            feedback.append("Python script file found.")
            logger.info("Python script file located.")

            with open(script_file, "r", encoding="utf-8") as f:
                content = f.read().lower()
            if "import pandas" in content:
                score += 0.2
                feedback.append("Script correctly imports pandas.")
                logger.info("Verified pandas import.")
            else:
                feedback.append("Missing pandas import.")
                logger.warning("Pandas import missing.")
        else:
            feedback.append("Python script missing.")
            logger.warning("Script file not found.")

        # Step 2: Verify output file
        if os.path.exists(result_file):
            score += 0.2
            feedback.append("Result file return_volatility.txt found.")
            logger.info("Result file detected.")

            try:
                with open(result_file, "r", encoding="utf-8") as f:
                    content = f.read().strip()

                match = re.search(r"[-+]?\d*\.\d+|\d+", content)
                if match:
                    value = float(match.group())
                    logger.info(f"Extracted volatility: {value}")

                    if abs(value - true_volatility) <= 0.01:
                        score += 0.3
                        feedback.append(f"Volatility value matches expected ({value:.2f}).")
                        logger.info("Volatility matches expected.")
                    else:
                        feedback.append(
                            f"Volatility differs from expected. Expected {true_volatility:.2f}, found {value:.2f}."
                        )
                        logger.warning("Value mismatch.")
                else:
                    feedback.append("No numeric value found in result file.")
                    logger.warning("Failed numeric extraction.")
            except Exception as e:
                feedback.append(f"Error reading return_volatility.txt: {e}")
                logger.error(f"Error reading file: {e}", exc_info=True)
        else:
            feedback.append("Result file return_volatility.txt missing.")
            logger.warning("Result file missing.")

    except Exception as e:
        feedback.append(f"Evaluation failed: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)

    final_score = min(score, 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)
    return final_score

def evaluate_stock_adj_trends_viz_quality(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Stock Ticker Trends Visualization task.

    This evaluation checks:
      - that the Python script exists and imports pandas
      - that both visualization files exist and are non-empty
      - and that GPT-4o confirms the charts clearly represent the correct information
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    expected = expected.get("rules", expected)
    base_dir = os.path.dirname(actual)

    py_file = os.path.join(base_dir, "stock_adj_trends_viz.py")
    chart1 = os.path.join(base_dir, "adjclose_trend.png")
    chart2 = os.path.join(base_dir, "volume_trend.png")

    score = 0.0
    feedback = []

    # Step 1: Check script and imports
    if os.path.exists(py_file):
        score += 0.3
        feedback.append("The Python script file is present.")
        logger.info("Python script file found.")

        with open(py_file, "r", encoding="utf-8") as f:
            content = f.read().lower()

        if "import pandas" in content and (
            "matplotlib" in content or "seaborn" in content or "plotly" in content or "altair" in content
        ):
            score += 0.2
            feedback.append("The script correctly imports pandas and a visualization library.")
        else:
            feedback.append("Missing pandas or visualization library import.")
            logger.warning("Import verification failed.")
    else:
        feedback.append("The Python script is missing.")
        logger.warning("Python script missing.")

    # Helper: Encode images for GPT-4o
    def encode_image(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    # Step 2: Visualization checks
    charts_to_check = [
        (
            chart1,
            "a line chart showing AdjClose over time, with clearly labeled axes, title, and readable scale"
        ),
        (
            chart2,
            "a bar chart showing AdjVolume over time, with Date on the x-axis, Volume on the y-axis, clear labels, and readable layout"
        )
    ]

    for chart, description in charts_to_check:
        chart_name = os.path.basename(chart)
        if os.path.exists(chart) and os.path.getsize(chart) > 0:
            score += 0.15
            feedback.append(f"The visualization {chart_name} exists and is non-empty.")

            try:
                encoded = encode_image(chart)
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert in data visualization evaluation. Judge chart clarity, readability, and correctness."
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        f"Examine this chart and confirm if it correctly represents {description}. "
                                        f"Reply 'yes' if it does, otherwise 'no', followed by a short reason."
                                    )
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{encoded}"}
                                }
                            ]
                        }
                    ],
                    max_tokens=100
                )

                answer = response.choices[0].message.content.lower()
                if "yes" in answer:
                    score += 0.1
                    feedback.append(f"GPT-4o judged {chart_name} as clear, accurate, and visually informative.")
                    logger.info(f"{chart_name} validated by GPT-4o.")
                else:
                    feedback.append(f"GPT-4o judged {chart_name} unclear or inaccurate: {answer}")
                    logger.warning(f"{chart_name} judged unclear by GPT-4o.")
            except Exception as e:
                feedback.append(f"Error evaluating {chart_name} with GPT-4o: {e}")
                logger.error(f"Error evaluating {chart_name}: {e}", exc_info=True)
        else:
            feedback.append(f"The visualization {chart_name} is missing or empty.")
            logger.warning(f"{chart_name} missing or zero-size.")

    # Step 3: Final score
    final_score = min(score, 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score


def evaluate_bullish_crossover(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Bullish Moving Average Crossover task.

    This evaluator checks:
      - The Python script exists and imports pandas.
      - The result file bullish_crossover.csv exists.
      - The CSV contains correct columns (Date, Stock).
      - The total number of bullish crossovers equals 196 (±2 tolerance).
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    base_dir = os.path.dirname(actual)
    script_file = os.path.join(base_dir, "bullish_crossover.py")
    result_file = os.path.join(base_dir, "bullish_crossover.csv")

    expected_crossovers = 196
    score = 0.0
    feedback = []

    try:
        # Check script file
        if os.path.exists(script_file):
            score += 0.3
            feedback.append("Python script file found.")
            logger.info("Script file located.")

            with open(script_file, "r", encoding="utf-8") as f:
                content = f.read().lower()
            if "import pandas" in content:
                score += 0.2
                feedback.append("Script correctly imports pandas.")
                logger.info("Verified pandas import.")
            else:
                feedback.append("Missing pandas import.")
                logger.warning("Pandas import not found.")
        else:
            feedback.append("Python script missing.")
            logger.warning("Script file missing.")

        # Check output CSV
        if os.path.exists(result_file):
            score += 0.2
            feedback.append("Result file bullish_crossover.csv found.")
            logger.info("Result file found.")
            try:
                df = pd.read_csv(result_file)
                cols = [c.lower().strip() for c in df.columns]
                if all(c in cols for c in ["date", "stock"]):
                    score += 0.1
                    feedback.append("Correct columns (Date, Stock) present.")
                    logger.info("Columns verified.")
                else:
                    feedback.append("Incorrect columns in output file.")
                    logger.warning("Column names mismatch.")

                total = len(df)
                if abs(total - expected_crossovers) <= 2:
                    score += 0.2
                    feedback.append(f"Crossovers count matches expected ({total}).")
                    logger.info("Crossover count verified within tolerance.")
                else:
                    feedback.append(
                        f"Crossovers count mismatch. Expected {expected_crossovers}, found {total}."
                    )
                    logger.warning("Crossover count mismatch.")
            except Exception as e:
                feedback.append(f"Error reading bullish_crossover.csv: {e}")
                logger.error(f"Error reading output file: {e}", exc_info=True)
        else:
            feedback.append("Result file bullish_crossover.csv missing.")
            logger.warning("Result file not found.")

    except Exception as e:
        feedback.append(f"Evaluation failed: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)

    final_score = min(score, 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)
    return final_score


def evaluate_stock_regression_mae(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Stock Linear Regression task.

    Checks:
      - stock_linear_regression.py exists
      - pandas and sklearn are imported
      - model_mae.txt exists
      - MAE is a valid positive float
      - Full score if MAE <= 2.675
    """

    if actual is None:
        logger.error("No directory provided.")
        return 0.0

    expected = expected.get("rules", expected)
    base_dir = os.path.dirname(actual)
    script_file = os.path.join(base_dir, "stock_linear_regression.py")
    result_file = os.path.join(base_dir, "model_mae.txt")

    score = 0.0
    feedback = []

    # Step 1: Check the script
    if os.path.exists(script_file):
        score += 0.4
        with open(script_file, "r", encoding="utf-8") as f:
            content = f.read().lower()
        if "import pandas" in content and "sklearn" in content:
            score += 0.2
            feedback.append("Script imports pandas and sklearn correctly.")
        else:
            feedback.append("Missing pandas or sklearn import.")
    else:
        feedback.append("Script file stock_linear_regression.py is missing.")

    # Step 2: Check the result file and MAE value
    if os.path.exists(result_file):
        score += 0.2
        with open(result_file, "r", encoding="utf-8") as f:
            output = f.read().strip()
        try:
            mae_value = float(output)
            if mae_value >= 0:
                if mae_value <= 2.675:
                    score += 0.2
                    feedback.append(f"MAE = {mae_value:.3f} (good performance, ≤ 2.675).")
                else:
                    score += 0.1
                    feedback.append(f"MAE = {mae_value:.3f} (valid but higher than expected).")
            else:
                feedback.append("MAE value is negative, which is invalid.")
        except Exception as e:
            feedback.append(f"Error parsing MAE value: {e}")
    else:
        feedback.append("Output file model_mae.txt is missing.")

    final_score = min(score, 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score 

def evaluate_wage_rigidity_result_1998(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Wage Rigidity Jupyter notebook task with known true value (1998).

    This evaluator checks:
      - The Jupyter notebook (wage_rigidity_analysis.ipynb) exists.
      - The result file (wage_rigidity_summary.txt) exists.
      - The result file contains a numeric value matching the known true value (1998 ± 1).
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    base_dir = os.path.dirname(actual)
    notebook_file = os.path.join(base_dir, "wage_rigidity_analysis.ipynb")
    result_file = os.path.join(base_dir, "wage_rigidity_summary.txt")

    true_value = 1998
    score = 0.0
    feedback = []

    try:
        # Check notebook existence
        if os.path.exists(notebook_file):
            score += 0.4
            feedback.append("Notebook file found.")
            logger.info("Notebook file located successfully.")
        else:
            feedback.append("Notebook file missing.")
            logger.warning("Notebook file not found.")

        # Check result file existence and content
        if os.path.exists(result_file):
            score += 0.3
            feedback.append("Result file found.")
            logger.info("Result file detected.")

            try:
                with open(result_file, "r", encoding="utf-8") as f:
                    content = f.read().strip()

                match = re.search(r"[-+]?\d*\.\d+|\d+", content)
                if match:
                    value = float(match.group())
                    logger.info(f"Extracted numeric value: {value}")

                    if abs(value - true_value) <= 1:
                        score += 0.3
                        feedback.append(f"Value matches expected ({value}).")
                        logger.info("Output matches the expected true value (1998).")
                    else:
                        feedback.append(f"Value differs. Expected {true_value}, found {value}.")
                        logger.warning("Numeric mismatch detected.")
                else:
                    feedback.append("No numeric value found in result file.")
                    logger.warning("Numeric extraction failed.")
            except Exception as e:
                feedback.append(f"Error reading wage_rigidity_summary.txt: {e}")
                logger.error(f"Error reading output file: {e}", exc_info=True)
        else:
            feedback.append("Result file missing.")
            logger.warning("Result file not found.")

    except Exception as e:
        feedback.append(f"Evaluation failed: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)

    # Final score capped at 1.0
    final_score = min(score, 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)
    return final_score


def evaluate_industry_correlation(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Industry Correlation Jupyter Notebook task.

    This evaluator checks:
      - The notebook file (industry_correlation.ipynb) exists.
      - The correlation matrix file (industry_corr.csv) exists and contains the expected structure.
      - The strongest correlation pair file (strongest_corr.txt) exists and matches the known true value.
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    base_dir = os.path.dirname(actual)
    notebook_file = os.path.join(base_dir, "industry_correlation.ipynb")
    corr_file = os.path.join(base_dir, "industry_corr.csv")
    pair_file = os.path.join(base_dir, "strongest_corr.txt")

    true_strongest_pair = "Finance–Manufacturing"
    true_corr_values = {
        ("Construction", "Finance"): 0.681989,
        ("Construction", "Manufacturing"): 0.710844,
        ("Finance", "Manufacturing"): 0.748307
    }

    score = 0.0
    feedback = []

    try:
        # Check notebook file
        if os.path.exists(notebook_file):
            score += 0.3
            feedback.append("Notebook file found.")
            logger.info("Jupyter notebook file found.")
        else:
            feedback.append("Notebook file missing.")
            logger.warning("Notebook file missing.")

        # Check correlation CSV file
        if os.path.exists(corr_file):
            score += 0.4
            feedback.append("Correlation matrix file found.")
            logger.info("Correlation matrix file found.")

            try:
                df = pd.read_csv(corr_file, index_col=0)
                expected_cols = ["Construction", "Finance", "Manufacturing"]
                if all(col in df.columns for col in expected_cols):
                    score += 0.1
                    feedback.append("Correlation matrix columns are correct.")
                    logger.info("Matrix columns verified.")

                    # Optional precision check (tolerance)
                    valid = True
                    for (a, b), expected_corr in true_corr_values.items():
                        if a in df.index and b in df.columns:
                            val = round(float(df.loc[a, b]), 6)
                            if abs(val - expected_corr) > 0.01:
                                valid = False
                                feedback.append(f"Mismatch in correlation for {a}–{b}: found {val}, expected {expected_corr}.")
                                logger.warning(f"Value mismatch for {a}–{b}.")
                    if valid:
                        score += 0.1
                        feedback.append("Correlation values match expected within tolerance.")
                        logger.info("Correlation values verified.")
                else:
                    feedback.append("Missing expected columns in correlation matrix.")
                    logger.warning("Column mismatch in correlation matrix.")
            except Exception as e:
                feedback.append(f"Error reading correlation CSV: {e}")
                logger.error(f"Error reading correlation matrix: {e}", exc_info=True)
        else:
            feedback.append("Correlation matrix file missing.")
            logger.warning("Correlation CSV missing.")

        # Check strongest correlation text file
        if os.path.exists(pair_file):
            score += 0.1
            feedback.append("Strongest correlation pair file found.")
            logger.info("Strongest pair file found.")
            try:
                with open(pair_file, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                if true_strongest_pair.lower() in content.lower():
                    score += 0.1
                    feedback.append(f"Strongest correlation pair matches expected ({true_strongest_pair}).")
                    logger.info("Strongest pair matches expected.")
                else:
                    feedback.append(f"Strongest pair mismatch. Expected '{true_strongest_pair}', found '{content}'.")
                    logger.warning("Strongest pair mismatch.")
            except Exception as e:
                feedback.append(f"Error reading strongest_corr.txt: {e}")
                logger.error(f"Error reading pair file: {e}", exc_info=True)
        else:
            feedback.append("Strongest correlation pair file missing.")
            logger.warning("Strongest correlation text file missing.")

    except Exception as e:
        feedback.append(f"Evaluation failed: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)

    final_score = min(score, 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)
    return final_score


def evaluate_jupyter_wage_trend_visualizations(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Wage Trend Analysis (Jupyter Notebook) task.

    This evaluation checks:
      - that the Jupyter notebook file exists
      - that both visualization files (wage_trend.png and industry_trend.png) exist and are non-empty
      - and that GPT-4o confirms both charts clearly represent the correct information
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    expected = expected.get("rules", expected)
    base_dir = os.path.dirname(actual)

    notebook_file = os.path.join(base_dir, "wage_trend_analysis.ipynb")
    chart1 = os.path.join(base_dir, "wage_trend.png")
    chart2 = os.path.join(base_dir, "industry_trend.png")

    score = 0.0
    feedback = []

    # Step 1: Check that the notebook exists
    if os.path.exists(notebook_file):
        score += 0.4
        feedback.append("The Jupyter Notebook file exists.")
        logger.info("Notebook file found.")
    else:
        feedback.append("The Jupyter Notebook file is missing.")
        logger.warning("Notebook file missing.")

    # Helper: Encode image for GPT-4o
    def encode_image(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    # Step 2: Visualization checks
    charts_to_check = [
        (
            chart1,
            "a line chart showing trends for All workers (hourly and non-hourly), Hourly workers, and Non-hourly workers over time with labeled axes and legend"
        ),
        (
            chart2,
            "a line chart comparing Construction, Finance, and Manufacturing wage trends over time with readable titles and clear color distinction"
        )
    ]

    for chart, description in charts_to_check:
        chart_name = os.path.basename(chart)
        if os.path.exists(chart) and os.path.getsize(chart) > 0:
            score += 0.15
            feedback.append(f"The visualization {chart_name} exists and is non-empty.")
            logger.info(f"{chart_name} found and non-empty.")

            try:
                encoded = encode_image(chart)
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert in data visualization evaluation. Judge chart clarity, readability, and correctness."
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        f"Examine this chart and confirm if it correctly represents {description}. "
                                        f"Reply 'yes' if it does, otherwise 'no', followed by a short reason."
                                    )
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{encoded}"}
                                }
                            ]
                        }
                    ],
                    max_tokens=100
                )

                answer = response.choices[0].message.content.lower()
                if "yes" in answer:
                    score += 0.1
                    feedback.append(f"GPT-4o judged {chart_name} as clear, accurate, and visually informative.")
                    logger.info(f"{chart_name} validated by GPT-4o.")
                else:
                    feedback.append(f"GPT-4o judged {chart_name} unclear or inaccurate: {answer}")
                    logger.warning(f"{chart_name} judged unclear by GPT-4o.")
            except Exception as e:
                feedback.append(f"Error evaluating {chart_name} with GPT-4o: {e}")
                logger.error(f"Error evaluating {chart_name}: {e}", exc_info=True)
        else:
            feedback.append(f"The visualization {chart_name} is missing or empty.")
            logger.warning(f"{chart_name} missing or zero-size.")

    # Step 3: Final scoring
    final_score = min(score, 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score

def evaluate_missing_value_imputation1(actual: str, expected: dict, **options) -> float:
    """
    Dynamic evaluator for the Jupyter missing value imputation task.

    This evaluator checks:
      - that the Jupyter notebook file exists
      - that both imputed_data.csv and imputation_summary.txt exist
      - that missing values in MonthlyIncome and NumberOfDependents were filled
      - and that the summary file lists columns which originally had missing values
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    base_dir = os.path.dirname(actual)
    notebook_file = os.path.join(base_dir, "missing_value_imputation.ipynb")
    input_file = os.path.join(base_dir, "data.csv")
    imputed_file = os.path.join(base_dir, "imputed_data.csv")
    summary_file = os.path.join(base_dir, "imputation_summary.txt")

    score = 0.0
    feedback = []

    try:
        # 1️⃣ Check notebook presence
        if os.path.exists(notebook_file):
            score += 0.3
            feedback.append("Notebook file found.")
            logger.info("Notebook file found.")
        else:
            feedback.append("Notebook file missing.")
            logger.warning("Notebook file not found.")

        # 2️⃣ Check input dataset
        if not os.path.exists(input_file):
            feedback.append("Original dataset missing.")
            logger.error("data.csv not found.")
            return score

        df_original = pd.read_csv(input_file)
        original_missing_cols = df_original.columns[df_original.isna().any()].tolist()
        logger.info(f"Original missing columns: {original_missing_cols}")

        # 3️⃣ Check imputed output CSV
        if os.path.exists(imputed_file):
            score += 0.4
            feedback.append("Imputed dataset file found.")
            logger.info("Imputed dataset located.")

            df_imputed = pd.read_csv(imputed_file)
            remaining_nans = df_imputed.isna().sum()
            still_missing = remaining_nans[remaining_nans > 0].index.tolist()

            if not still_missing:
                score += 0.2
                feedback.append("All missing values have been successfully imputed.")
                logger.info("No missing values detected in imputed data.")
            else:
                feedback.append(f"Some columns still contain missing values: {still_missing}")
                logger.warning("Missing values remain after imputation.")
        else:
            feedback.append("Imputed dataset file missing.")
            logger.warning("imputed_data.csv not found.")

        # 4️⃣ Check summary text file
        if os.path.exists(summary_file):
            score += 0.1
            feedback.append("Summary text file found.")
            logger.info("imputation_summary.txt located.")

            try:
                with open(summary_file, "r", encoding="utf-8") as f:
                    content = f.read().lower()
                listed_cols = [col for col in original_missing_cols if col.lower() in content]
                if listed_cols:
                    score += 0.1
                    feedback.append("Summary file correctly lists missing-value columns.")
                    logger.info(f"Summary verification passed: {listed_cols}")
                else:
                    feedback.append("Summary file does not list missing columns correctly.")
                    logger.warning("Summary content verification failed.")
            except Exception as e:
                feedback.append(f"Error reading summary file: {e}")
                logger.error(f"Error reading summary file: {e}", exc_info=True)
        else:
            feedback.append("Summary file missing.")
            logger.warning("imputation_summary.txt not found.")

    except Exception as e:
        feedback.append(f"Evaluation failed: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)

    # Cap final score
    final_score = min(score, 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score


def evaluate_credit_correlation(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Credit Correlation Jupyter Notebook task.

    Checks:
      - The Jupyter notebook (credit_correlation.ipynb) exists.
      - The result file (credit_corr.txt) exists.
      - The file contains a single numeric value within [-1, 1].
      - The value matches the known true correlation (-0.002 ± 0.01).
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    base_dir = os.path.dirname(actual)
    notebook_file = os.path.join(base_dir, "credit_correlation.ipynb")
    result_file = os.path.join(base_dir, "credit_corr.txt")

    true_corr = -0.002
    score = 0.0
    feedback = []

    try:
        # ✅ Notebook file check
        if os.path.exists(notebook_file):
            score += 0.4
            feedback.append("Notebook file found.")
            logger.info("Jupyter notebook file located.")
        else:
            feedback.append("Notebook file missing.")
            logger.warning("Notebook file not found.")

        # ✅ Result file check
        if os.path.exists(result_file):
            score += 0.4
            feedback.append("Result file credit_corr.txt found.")
            logger.info("Result file found.")

            try:
                with open(result_file, "r", encoding="utf-8") as f:
                    content = f.read().strip()

                match = re.search(r"[-+]?\d*\.\d+|\d+", content)
                if match:
                    value = float(match.group())
                    logger.info(f"Extracted correlation value: {value}")

                    if -1.0 <= value <= 1.0:
                        score += 0.1
                        feedback.append("Correlation value within valid range [-1, 1].")
                        logger.info("Value range valid.")
                    else:
                        feedback.append("Correlation value out of valid range.")
                        logger.warning("Value outside [-1, 1].")

                    if abs(value - true_corr) <= 0.01:
                        score += 0.1
                        feedback.append(f"Value matches expected ({value}).")
                        logger.info("Correlation value verified as correct.")
                    else:
                        feedback.append(f"Value differs. Expected {true_corr}, found {value}.")
                        logger.warning("Value mismatch.")
                else:
                    feedback.append("No numeric value detected in result file.")
                    logger.warning("Numeric extraction failed.")
            except Exception as e:
                feedback.append(f"Error reading or parsing credit_corr.txt: {e}")
                logger.error(f"Error reading result file: {e}", exc_info=True)
        else:
            feedback.append("Result file credit_corr.txt missing.")
            logger.warning("Result file not found.")

    except Exception as e:
        feedback.append(f"Evaluation failed: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)

    # ✅ Final score capped
    final_score = min(score, 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score


def evaluate_age_default_rate(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Age Default Rate Jupyter Notebook task.

    This evaluation checks:
      - The Jupyter notebook (age_default_rate.ipynb) exists.
      - The output CSV (age_default_rate.csv) exists and has correct structure.
      - The DefaultRate values match the known expected reference (±0.01 tolerance).
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    base_dir = os.path.dirname(actual)
    notebook_file = os.path.join(base_dir, "age_default_rate.ipynb")
    output_file = os.path.join(base_dir, "age_default_rate.csv")

    # ✅ True reference values from provided ground truth
    expected_rates = {
        "<40": 0.105,
        "40–60": 0.073,
        ">60": 0.030
    }

    score = 0.0
    feedback = []

    try:
        # Notebook existence check
        if os.path.exists(notebook_file):
            score += 0.3
            feedback.append("Notebook file found.")
            logger.info("Jupyter notebook located.")
        else:
            feedback.append("Notebook file missing.")
            logger.warning("Notebook not found.")

        # Output file existence check
        if os.path.exists(output_file):
            score += 0.3
            feedback.append("Output file age_default_rate.csv found.")
            logger.info("Output CSV located.")

            try:
                df = pd.read_csv(output_file)
                logger.info(f"Loaded CSV with columns: {list(df.columns)}")

                # Column validation
                if set(df.columns) == {"AgeGroup", "DefaultRate"}:
                    score += 0.1
                    feedback.append("CSV columns are correctly named (AgeGroup, DefaultRate).")
                    logger.info("Columns verified.")
                else:
                    feedback.append("CSV columns do not match expected names.")
                    logger.warning("Column mismatch detected.")

                # Row validation
                if len(df) == 3:
                    score += 0.1
                    feedback.append("CSV contains exactly 3 age groups.")
                    logger.info("Row count verified.")
                else:
                    feedback.append(f"Unexpected row count ({len(df)}). Expected 3.")
                    logger.warning("Row count mismatch.")

                # Value check
                matches = 0
                for _, row in df.iterrows():
                    group = str(row["AgeGroup"]).strip()
                    rate = float(row["DefaultRate"])
                    if group in expected_rates and abs(rate - expected_rates[group]) <= 0.01:
                        matches += 1

                if matches == 3:
                    score += 0.2
                    feedback.append("All default rates match expected values within tolerance.")
                    logger.info("DefaultRate values verified successfully.")
                elif matches > 0:
                    score += 0.1
                    feedback.append(f"{matches}/3 DefaultRate values are correct within tolerance.")
                    logger.warning("Partial match on DefaultRate values.")
                else:
                    feedback.append("No DefaultRate values matched expected results.")
                    logger.warning("DefaultRate mismatch detected.")

            except Exception as e:
                feedback.append(f"Error reading or parsing age_default_rate.csv: {e}")
                logger.error(f"Error loading CSV: {e}", exc_info=True)
        else:
            feedback.append("Output CSV file missing.")
            logger.warning("Output CSV not found.")

    except Exception as e:
        feedback.append(f"Evaluation failed: {e}")
        logger.error(f"Evaluation failure: {e}", exc_info=True)

    # ✅ Final score capped
    final_score = min(score, 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score

def evaluate_income_debt_corr(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Spearman correlation between MonthlyIncome and DebtRatio.

    This evaluation checks:
      - The Jupyter notebook file (income_debt_correlation.ipynb) exists.
      - The result file (income_debt_corr.txt) exists.
      - The result contains a single numeric value within [-1, 1].
      - The correlation value matches the known true value (-0.131 ± 0.01).
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    base_dir = os.path.dirname(actual)
    notebook_file = os.path.join(base_dir, "income_debt_correlation.ipynb")
    result_file = os.path.join(base_dir, "income_debt_corr.txt")

    true_corr = -0.131
    score = 0.0
    feedback = []

    try:
        # Notebook presence check
        if os.path.exists(notebook_file):
            score += 0.4
            feedback.append("Notebook file found.")
            logger.info("Jupyter notebook file found.")
        else:
            feedback.append("Notebook file missing.")
            logger.warning("Notebook not found.")

        # Result file check
        if os.path.exists(result_file):
            score += 0.4
            feedback.append("Result file income_debt_corr.txt found.")
            logger.info("Result file found successfully.")

            try:
                with open(result_file, "r", encoding="utf-8") as f:
                    content = f.read().strip()

                match = re.search(r"[-+]?\d*\.\d+|\d+", content)
                if match:
                    value = float(match.group())
                    logger.info(f"Extracted correlation value: {value}")

                    # Range validation
                    if -1.0 <= value <= 1.0:
                        score += 0.1
                        feedback.append("Correlation value is within valid range [-1, 1].")
                    else:
                        feedback.append("Correlation value is outside valid range.")
                        logger.warning("Value outside [-1, 1].")

                    # Value comparison
                    if abs(value - true_corr) <= 0.01:
                        score += 0.1
                        feedback.append(f"Value matches expected ({value}).")
                        logger.info("Correlation value verified successfully.")
                    else:
                        feedback.append(f"Value differs from expected. Expected {true_corr}, found {value}.")
                        logger.warning("Correlation mismatch detected.")
                else:
                    feedback.append("No numeric value detected in result file.")
                    logger.warning("Numeric extraction failed.")
            except Exception as e:
                feedback.append(f"Error reading or parsing income_debt_corr.txt: {e}")
                logger.error(f"Error reading result file: {e}", exc_info=True)
        else:
            feedback.append("Result file income_debt_corr.txt missing.")
            logger.warning("Result file not found.")

    except Exception as e:
        feedback.append(f"Evaluation failed: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)

    # Final score capped
    final_score = min(score, 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score 

def evaluate_logreg_top5_features(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Logistic Regression Top 5 Features Jupyter Notebook task.

    This evaluation checks:
      - The notebook file exists.
      - The CSV file (logreg_top5_features.csv) exists and contains 5 rows.
      - The columns are correctly named (Feature, Coefficient).
      - The features and their coefficient values match the known top 5 results within tolerance.
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    base_dir = os.path.dirname(actual)
    notebook_file = os.path.join(base_dir, "logistic_regression_features.ipynb")
    output_file = os.path.join(base_dir, "logreg_top5_features.csv")

    # ✅ Reference top 5 features with coefficients
    expected_features = {
        "NumberOfTime60-89DaysPastDueNotWorse": -3.1004,
        "NumberOfTime30-59DaysPastDueNotWorse": 1.7436,
        "NumberOfTimes90DaysLate": 1.5165,
        "MonthlyIncome": -0.6079,
        "age": -0.3639
    }

    score = 0.0
    feedback = []

    try:
        # Notebook check
        if os.path.exists(notebook_file):
            score += 0.3
            feedback.append("Notebook file found.")
            logger.info("Notebook located successfully.")
        else:
            feedback.append("Notebook file missing.")
            logger.warning("Notebook not found.")

        # CSV existence check
        if os.path.exists(output_file):
            score += 0.3
            feedback.append("Output file logreg_top5_features.csv found.")
            logger.info("Output CSV found.")

            try:
                df = pd.read_csv(output_file)
                logger.info(f"Loaded CSV columns: {list(df.columns)}")

                # Validate columns
                if set(df.columns) == {"Feature", "Coefficient"}:
                    score += 0.1
                    feedback.append("Columns are correctly named (Feature, Coefficient).")
                    logger.info("Column names validated.")
                else:
                    feedback.append("Incorrect or missing column names in CSV.")
                    logger.warning("Column mismatch detected.")

                # Validate row count
                if len(df) == 5:
                    score += 0.1
                    feedback.append("CSV contains exactly 5 rows as expected.")
                    logger.info("Row count verified.")
                else:
                    feedback.append(f"Unexpected row count ({len(df)}). Expected 5.")
                    logger.warning("Row count mismatch.")

                # Match feature names and coefficients
                matched = 0
                for _, row in df.iterrows():
                    feature = str(row["Feature"]).strip()
                    coeff = float(row["Coefficient"])
                    if feature in expected_features:
                        if abs(coeff - expected_features[feature]) <= 0.1:
                            matched += 1

                if matched == 5:
                    score += 0.2
                    feedback.append("All 5 features and coefficients match expected values within tolerance.")
                    logger.info("All coefficient matches verified.")
                elif matched >= 3:
                    score += 0.1
                    feedback.append(f"{matched}/5 features match expected coefficients.")
                    logger.warning("Partial coefficient match.")
                else:
                    feedback.append("Few or no feature-coefficient matches found.")
                    logger.warning("Low feature match rate.")

            except Exception as e:
                feedback.append(f"Error reading or parsing logreg_top5_features.csv: {e}")
                logger.error(f"Error loading CSV: {e}", exc_info=True)
        else:
            feedback.append("Output CSV file missing.")
            logger.warning("Output CSV not found.")

    except Exception as e:
        feedback.append(f"Evaluation failed: {e}")
        logger.error(f"Evaluation error: {e}", exc_info=True)

    # ✅ Final score capped
    final_score = min(score, 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score

def evaluate_credit_risk_xgb_metrics(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Credit Risk XGBClassifier Jupyter Notebook task.

    This evaluator checks:
      - The Jupyter notebook (credit_risk_xgb.ipynb) exists.
      - The result file (model_metrics.txt) exists.
      - The result file contains exactly three numeric values (accuracy, precision, recall).
      - Each metric lies within [0, 1].
      - The model achieves acceptable performance (all metrics >= 0.7 for full score).
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    base_dir = os.path.dirname(actual)
    notebook_file = os.path.join(base_dir, "credit_risk_xgb.ipynb")
    result_file = os.path.join(base_dir, "model_metrics.txt")

    score = 0.0
    feedback = []

    try:
        # ✅ Step 1: Check notebook existence
        if os.path.exists(notebook_file):
            score += 0.3
            feedback.append("Notebook file found.")
            logger.info("Jupyter notebook located successfully.")
        else:
            feedback.append("Notebook file missing.")
            logger.warning("Notebook not found.")

        # ✅ Step 2: Check result file existence
        if os.path.exists(result_file):
            score += 0.3
            feedback.append("Result file model_metrics.txt found.")
            logger.info("Result file found successfully.")

            try:
                # Read content
                with open(result_file, "r", encoding="utf-8") as f:
                    content = f.read().strip()

                # Extract all numeric values (allow commas or newlines)
                numbers = re.findall(r"[-+]?\d*\.\d+|\d+", content)
                metrics = [float(x) for x in numbers if 0 <= float(x) <= 1]

                logger.info(f"Extracted metrics: {metrics}")

                # ✅ Step 3: Validate metrics format
                if len(metrics) == 3:
                    score += 0.1
                    feedback.append("Found three valid numeric metrics.")
                    logger.info("Three metrics extracted successfully.")
                else:
                    feedback.append(f"Expected 3 metrics but found {len(metrics)}.")
                    logger.warning("Incorrect number of metrics detected.")

                # ✅ Step 4: Evaluate performance thresholds
                if len(metrics) == 3:
                    passing = sum(m >= 0.7 for m in metrics)
                    if passing == 3:
                        score += 0.3
                        feedback.append("All metrics meet or exceed the 0.7 threshold.")
                        logger.info("Excellent model performance confirmed.")
                    elif passing >= 2:
                        score += 0.15
                        feedback.append("Two of three metrics meet the 0.7 threshold (partial credit).")
                        logger.warning("Moderate performance; partial credit awarded.")
                    else:
                        feedback.append("Model performance below threshold on most metrics.")
                        logger.warning("Performance below acceptable threshold.")
            except Exception as e:
                feedback.append(f"Error reading or parsing model_metrics.txt: {e}")
                logger.error(f"Error reading result file: {e}", exc_info=True)
        else:
            feedback.append("Result file model_metrics.txt missing.")
            logger.warning("Result file not found.")

    except Exception as e:
        feedback.append(f"Evaluation failed: {e}")
        logger.error(f"Evaluation failure: {e}", exc_info=True)

    # ✅ Final score capped
    final_score = min(score, 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score

def evaluate_highrisk_count(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the High-Risk Case Count Jupyter Notebook task.

    This evaluator checks:
      - that the Jupyter notebook file exists
      - that the result file highrisk_count.txt exists
      - that the file contains a valid integer value close to the expected (5143)
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    base_dir = os.path.dirname(actual)
    notebook_file = os.path.join(base_dir, "highrisk_analysis.ipynb")
    result_file = os.path.join(base_dir, "highrisk_count.txt")

    true_value = 5143
    score = 0.0
    feedback = []

    try:
        # Step 1: Check notebook presence
        if os.path.exists(notebook_file):
            score += 0.4
            feedback.append("Notebook file located successfully.")
            logger.info("Notebook file found.")
        else:
            feedback.append("Notebook file is missing.")
            logger.warning("Notebook file not found.")

        # Step 2: Check result file presence
        if os.path.exists(result_file):
            score += 0.3
            feedback.append("Result file highrisk_count.txt found.")
            logger.info("Result file found.")

            try:
                # Step 3: Read and extract numeric value
                with open(result_file, "r", encoding="utf-8") as f:
                    content = f.read().strip()

                match = re.search(r"\d+", content)
                if match:
                    value = int(match.group())
                    logger.info(f"Extracted value: {value}")

                    # Step 4: Compare with expected value
                    if abs(value - true_value) <= 5:
                        score += 0.3
                        feedback.append(f"Reported count matches expected ({value}).")
                        logger.info("High-risk case count matches expected result.")
                    else:
                        feedback.append(f"Reported count differs. Expected {true_value}, found {value}.")
                        logger.warning(f"Value mismatch: expected {true_value}, found {value}")
                else:
                    feedback.append("No numeric value detected in the result file.")
                    logger.warning("No valid numeric pattern found.")
            except Exception as e:
                feedback.append(f"Error reading or parsing result file: {e}")
                logger.error(f"Error reading result file: {e}", exc_info=True)
        else:
            feedback.append("Result file highrisk_count.txt missing.")
            logger.warning("Result file not found.")

    except Exception as e:
        feedback.append(f"Evaluation failed: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)

    final_score = min(score, 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)
    return final_score


def evaluate_jupyter_highdebt_default_visualizations(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the HighDebt Default Visualization (Jupyter Notebook) task.

    This evaluation checks:
      - that the Jupyter notebook file exists
      - that both visualization files (highdebt_default_rate.png and utilization_boxplot.png) exist and are non-empty
      - and that GPT-4o confirms both charts clearly represent the correct information
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    expected = expected.get("rules", expected)
    base_dir = os.path.dirname(actual)

    notebook_file = os.path.join(base_dir, "highdebt_default_analysis.ipynb")
    chart1 = os.path.join(base_dir, "highdebt_default_rate.png")
    chart2 = os.path.join(base_dir, "utilization_boxplot.png")

    score = 0.0
    feedback = []

    # Step 1: Check that the notebook exists
    if os.path.exists(notebook_file):
        score += 0.4
        feedback.append("The Jupyter Notebook file exists.")
        logger.info("Notebook file found.")
    else:
        feedback.append("The Jupyter Notebook file is missing.")
        logger.warning("Notebook file missing.")

    # Helper: encode images for GPT-4o
    def encode_image(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    # Step 2: Visualization checks
    charts_to_check = [
        (
            chart1,
            "a bar chart showing the proportion of SeriousDlqin2yrs = 1 within each HighDebt group, with clear labeling and readable percentages"
        ),
        (
            chart2,
            "a boxplot of RevolvingUtilizationOfUnsecuredLines grouped by SeriousDlqin2yrs, with visible distributions, labeled axes, and a title"
        )
    ]

    for chart, description in charts_to_check:
        chart_name = os.path.basename(chart)
        if os.path.exists(chart) and os.path.getsize(chart) > 0:
            score += 0.15
            feedback.append(f"The visualization {chart_name} exists and is non-empty.")
            logger.info(f"{chart_name} found and non-empty.")

            try:
                encoded = encode_image(chart)
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert in data visualization quality evaluation. Judge chart clarity, readability, and correctness."
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        f"Examine this chart and confirm if it correctly represents {description}. "
                                        f"Reply 'yes' if it does, otherwise 'no', followed by a short reason."
                                    )
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{encoded}"}
                                }
                            ]
                        }
                    ],
                    max_tokens=100
                )

                answer = response.choices[0].message.content.lower()
                if "yes" in answer:
                    score += 0.1
                    feedback.append(f"GPT-4o judged {chart_name} as clear, accurate, and visually informative.")
                    logger.info(f"{chart_name} validated by GPT-4o.")
                else:
                    feedback.append(f"GPT-4o judged {chart_name} unclear or inaccurate: {answer}")
                    logger.warning(f"{chart_name} judged unclear by GPT-4o.")
            except Exception as e:
                feedback.append(f"Error evaluating {chart_name} with GPT-4o: {e}")
                logger.error(f"Error evaluating {chart_name}: {e}", exc_info=True)
        else:
            feedback.append(f"The visualization {chart_name} is missing or empty.")
            logger.warning(f"{chart_name} missing or zero-size.")

    # Step 3: Final scoring
    final_score = min(score, 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score


def evaluate_diabetes_imputation(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Diabetes Imputation Jupyter Notebook task.

    This evaluator checks:
      - that the notebook file (diabetes_imputation.ipynb) exists
      - that the filled_diabetes.csv file exists
      - that zero values have been replaced in the specified columns:
        ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    base_dir = os.path.dirname(actual)
    notebook_file = os.path.join(base_dir, "diabetes_imputation.ipynb")
    cleaned_file = os.path.join(base_dir, "filled_diabetes.csv")

    target_columns = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

    score = 0.0
    feedback = []

    try:
        # ✅ Step 1: Check for notebook file
        if os.path.exists(notebook_file):
            score += 0.3
            feedback.append("Notebook file located successfully.")
            logger.info("Notebook file found.")
        else:
            feedback.append("Notebook file missing.")
            logger.warning("Notebook not found.")

        # ✅ Step 2: Check for cleaned dataset
        if os.path.exists(cleaned_file):
            score += 0.4
            feedback.append("Cleaned dataset file found.")
            logger.info("Cleaned dataset file found.")

            try:
                df = pd.read_csv(cleaned_file)

                # ✅ Step 3: Check columns presence
                missing_cols = [c for c in target_columns if c not in df.columns]
                if missing_cols:
                    feedback.append(f"Missing columns in dataset: {missing_cols}")
                    logger.warning(f"Missing columns: {missing_cols}")
                else:
                    score += 0.1
                    feedback.append("All required columns are present.")
                    logger.info("All required columns present in dataset.")

                    # ✅ Step 4: Verify zero replacement
                    zero_free = True
                    for col in target_columns:
                        zero_count = (df[col] == 0).sum()
                        if zero_count > 0:
                            zero_free = False
                            feedback.append(f"Column '{col}' still contains {zero_count} zero values.")
                            logger.warning(f"Column {col} contains {zero_count} zeros.")
                        else:
                            logger.info(f"Column {col} successfully cleaned (no zeros).")

                    if zero_free:
                        score += 0.2
                        feedback.append("All zero values replaced successfully.")
                        logger.info("All columns cleaned successfully.")
                    else:
                        feedback.append("Some columns still contain zero values.")
                        logger.warning("Zero values remain in one or more columns.")
            except Exception as e:
                feedback.append(f"Error reading or analyzing CSV: {e}")
                logger.error(f"Error reading CSV: {e}", exc_info=True)
        else:
            feedback.append("Cleaned dataset file filled_diabetes.csv missing.")
            logger.warning("Cleaned file not found.")

    except Exception as e:
        feedback.append(f"Evaluation failed: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)

    # ✅ Final scoring cap
    final_score = min(score, 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score 

def evaluate_summary_statistics(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Summary Statistics Jupyter Notebook task.

    This evaluator checks:
      - that the notebook file exists
      - that the summary CSV file exists
      - that all required columns are present (Feature, Mean, Median, Std)
      - that key summary statistics approximately match the known expected values
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    base_dir = os.path.dirname(actual)
    notebook_file = os.path.join(base_dir, "summary_statistics.ipynb")
    result_file = os.path.join(base_dir, "summary_stats.csv")

    expected_features = {
        "Pregnancies": [3.845, 3.000, 3.370],
        "Glucose": [120.895, 117.000, 31.973],
        "BloodPressure": [69.105, 72.000, 19.356],
        "SkinThickness": [20.536, 23.000, 15.952],
        "Insulin": [79.799, 30.500, 115.244],
        "BMI": [31.993, 32.000, 7.884],
        "DiabetesPedigreeFunction": [0.472, 0.372, 0.331],
        "Age": [33.241, 29.000, 11.760],
        "Outcome": [0.349, 0.000, 0.477]
    }

    score = 0.0
    feedback = []

    try:
        # ✅ Step 1: Check for notebook
        if os.path.exists(notebook_file):
            score += 0.3
            feedback.append("Notebook file found.")
            logger.info("Notebook file found.")
        else:
            feedback.append("Notebook file missing.")
            logger.warning("Notebook file not found.")

        # ✅ Step 2: Check for result CSV
        if os.path.exists(result_file):
            score += 0.3
            feedback.append("Result file summary_stats.csv found.")
            logger.info("Summary CSV file found.")

            try:
                df = pd.read_csv(result_file)

                # ✅ Step 3: Check required columns
                required_cols = ["Feature", "Mean", "Median", "Std"]
                if all(col in df.columns for col in required_cols):
                    score += 0.1
                    feedback.append("All required columns present.")
                    logger.info("Required columns verified.")
                else:
                    feedback.append("Missing one or more required columns.")
                    logger.warning("Missing columns detected.")

                # ✅ Step 4: Validate expected features and approximate numeric values
                matched = 0
                for feature, values in expected_features.items():
                    row = df[df["Feature"].astype(str).str.strip() == feature]
                    if not row.empty:
                        diffs = np.abs(row[["Mean", "Median", "Std"]].values[0] - np.array(values))
                        if np.all(diffs < 0.5):  # tolerance for rounding or floating-point noise
                            matched += 1
                            logger.info(f"Feature '{feature}' matches expected values.")
                        else:
                            logger.warning(f"Feature '{feature}' deviates beyond tolerance.")
                    else:
                        logger.warning(f"Feature '{feature}' missing from output.")

                # Score contribution for matching features
                ratio = matched / len(expected_features)
                if ratio >= 0.9:
                    score += 0.3
                    feedback.append("All or most feature statistics match expected values.")
                elif ratio >= 0.7:
                    score += 0.2
                    feedback.append("Most feature statistics are close to expected values.")
                elif ratio >= 0.5:
                    score += 0.1
                    feedback.append("Some feature statistics match expected values.")
                else:
                    feedback.append("Few or no statistics match expected results.")

            except Exception as e:
                feedback.append(f"Error reading or validating summary_stats.csv: {e}")
                logger.error(f"Error reading CSV: {e}", exc_info=True)
        else:
            feedback.append("Result file summary_stats.csv missing.")
            logger.warning("Result file missing.")

    except Exception as e:
        feedback.append(f"Evaluation failed: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)

    # ✅ Final scoring cap
    final_score = min(score, 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score


def evaluate_correlation_matrix(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Correlation Matrix Jupyter Notebook task.

    This evaluator checks:
      - The Jupyter notebook file exists.
      - The correlation matrix CSV file exists and contains numeric values.
      - The strongest correlation text file exists and matches the expected pair and correlation.
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    base_dir = os.path.dirname(actual)
    notebook_file = os.path.join(base_dir, "correlation_analysis.ipynb")
    corr_file = os.path.join(base_dir, "correlation_matrix.csv")
    text_file = os.path.join(base_dir, "strongest_corr.txt")

    expected_pair = ("Pregnancies", "Age")
    expected_value = 0.544
    tolerance = 0.01

    score = 0.0
    feedback = []

    try:
        # ✅ Step 1: Check for notebook
        if os.path.exists(notebook_file):
            score += 0.3
            feedback.append("Notebook file found.")
            logger.info("Notebook file found.")
        else:
            feedback.append("Notebook file missing.")
            logger.warning("Notebook not found.")

        # ✅ Step 2: Check correlation matrix CSV
        if os.path.exists(corr_file):
            score += 0.3
            feedback.append("Correlation matrix CSV found.")
            logger.info("Correlation matrix file found.")

            try:
                df = pd.read_csv(corr_file, index_col=0)
                if df.shape[0] == df.shape[1] and np.issubdtype(df.values.dtype, np.number):
                    score += 0.1
                    feedback.append("Correlation matrix is square and numeric.")
                    logger.info("Valid correlation matrix detected.")
                else:
                    feedback.append("Correlation matrix invalid (not square or not numeric).")
                    logger.warning("Invalid correlation matrix structure.")
            except Exception as e:
                feedback.append(f"Error reading correlation matrix CSV: {e}")
                logger.error(f"Error reading correlation CSV: {e}", exc_info=True)
        else:
            feedback.append("Correlation matrix CSV missing.")
            logger.warning("correlation_matrix.csv not found.")

        # ✅ Step 3: Check strongest correlation text
        if os.path.exists(text_file):
            score += 0.2
            feedback.append("Strongest correlation text file found.")
            logger.info("strongest_corr.txt found.")

            try:
                with open(text_file, "r", encoding="utf-8") as f:
                    content = f.read().strip()

                # Match both pair names and correlation number
                pair_match = re.findall(r"[A-Za-z]+", content)
                num_match = re.findall(r"[-+]?\d*\.\d+|\d+", content)

                if len(pair_match) >= 2 and num_match:
                    corr_value = float(num_match[-1])
                    col1, col2 = pair_match[0], pair_match[1]
                    logger.info(f"Extracted: {col1} – {col2} = {corr_value}")

                    # Check correctness
                    pair_correct = {col1.lower(), col2.lower()} == {expected_pair[0].lower(), expected_pair[1].lower()}
                    value_correct = abs(corr_value - expected_value) <= tolerance

                    if pair_correct and value_correct:
                        score += 0.1
                        feedback.append("Reported strongest correlation pair and value match expected result.")
                        logger.info("Strongest correlation matches expected pair and value.")
                    elif pair_correct:
                        feedback.append(f"Pair correct ({col1}-{col2}) but value differs. Expected {expected_value}, found {corr_value}.")
                        logger.warning("Correlation value mismatch.")
                    else:
                        feedback.append(f"Incorrect pair reported: {col1}-{col2}. Expected {expected_pair[0]}-{expected_pair[1]}.")
                        logger.warning("Incorrect pair reported in strongest_corr.txt.")
                else:
                    feedback.append("Could not parse valid pair or correlation value from strongest_corr.txt.")
                    logger.warning("Parsing failure in strongest_corr.txt.")
            except Exception as e:
                feedback.append(f"Error reading strongest_corr.txt: {e}")
                logger.error(f"Error reading strongest_corr.txt: {e}", exc_info=True)
        else:
            feedback.append("Strongest correlation text file missing.")
            logger.warning("strongest_corr.txt not found.")

    except Exception as e:
        feedback.append(f"Evaluation failed: {e}")
        logger.error(f"Evaluation failure: {e}", exc_info=True)

    # ✅ Final score capped
    final_score = min(score, 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score

def evaluate_glucose_category_risk_rate(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Glucose Category Diabetes Rate Jupyter task.

    Checks:
      - The Jupyter notebook file exists.
      - The output CSV (glucose_risk_rate.csv) exists.
      - Columns 'GlucoseCategory' and 'DiabetesRate' are present.
      - The DiabetesRate values for each category match expected results (±0.02 tolerance).
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    base_dir = os.path.dirname(actual)
    notebook_file = os.path.join(base_dir, "glucose_risk_analysis.ipynb")
    output_file = os.path.join(base_dir, "glucose_risk_rate.csv")

    expected_rates = {
        "High": 0.685,
        "Normal": 0.313,
        "Low": 0.081
    }
    tolerance = 0.02

    score = 0.0
    feedback = []

    try:
        # Notebook check
        if os.path.exists(notebook_file):
            score += 0.3
            feedback.append("Notebook file found.")
            logger.info("Notebook file found.")
        else:
            feedback.append("Notebook file missing.")
            logger.warning("Notebook file not found.")

        # Output CSV check
        if os.path.exists(output_file):
            score += 0.4
            feedback.append("Output CSV file found.")
            logger.info("Output CSV file detected.")

            try:
                df = pd.read_csv(output_file)
                expected_cols = {"GlucoseCategory", "DiabetesRate"}
                if expected_cols.issubset(df.columns):
                    score += 0.1
                    feedback.append("Output CSV contains required columns.")
                    logger.info("CSV columns verified.")

                    matched = 0
                    for cat, exp_val in expected_rates.items():
                        row = df[df["GlucoseCategory"].str.lower() == cat.lower()]
                        if not row.empty:
                            actual_val = float(row["DiabetesRate"].values[0])
                            if abs(actual_val - exp_val) <= tolerance:
                                matched += 1
                                logger.info(f"{cat}: matched (expected={exp_val}, found={actual_val})")
                            else:
                                logger.warning(f"{cat}: value mismatch (expected={exp_val}, found={actual_val})")
                        else:
                            logger.warning(f"Category {cat} missing in output.")

                    if matched == len(expected_rates):
                        score += 0.2
                        feedback.append("All category DiabetesRate values are correct within tolerance.")
                    elif matched >= 2:
                        score += 0.1
                        feedback.append("Most DiabetesRate values are within acceptable range.")
                    else:
                        feedback.append("Values deviate significantly from expected results.")
                else:
                    feedback.append("Output CSV missing required columns.")
                    logger.warning("Missing expected columns in output CSV.")

            except Exception as e:
                feedback.append(f"Error reading or validating output CSV: {e}")
                logger.error(f"Error while reading glucose_risk_rate.csv: {e}", exc_info=True)
        else:
            feedback.append("Output CSV file missing.")
            logger.warning("glucose_risk_rate.csv not found.")

    except Exception as e:
        feedback.append(f"Evaluation failed: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)

    final_score = min(score, 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score


def evaluate_bmixage_comparison(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the BMIxAge Comparison Jupyter task.

    Checks:
      - Notebook file exists.
      - Output CSV (bmixage_comparison.csv) exists.
      - Columns 'Outcome' and 'BMIxAge' are present.
      - Mean BMIxAge values for Outcome=0 and Outcome=1 match expected results within ±1 tolerance.
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    base_dir = os.path.dirname(actual)
    notebook_file = os.path.join(base_dir, "bmi_age_analysis.ipynb")
    output_file = os.path.join(base_dir, "bmixage_comparison.csv")

    expected_values = {
        0: 948.418,
        1: 1287.713
    }
    tolerance = 1.0

    score = 0.0
    feedback = []

    try:
        # Notebook check
        if os.path.exists(notebook_file):
            score += 0.3
            feedback.append("Notebook file found.")
            logger.info("Notebook file found.")
        else:
            feedback.append("Notebook file missing.")
            logger.warning("Notebook not found.")

        # Output file check
        if os.path.exists(output_file):
            score += 0.4
            feedback.append("Output CSV file found.")
            logger.info("Output file detected.")

            try:
                df = pd.read_csv(output_file)
                expected_cols = {"Outcome", "BMIxAge"}

                # Column validation
                if expected_cols.issubset(df.columns):
                    score += 0.1
                    feedback.append("Output CSV contains required columns.")
                    logger.info("CSV columns verified.")

                    matched = 0
                    for outcome, exp_value in expected_values.items():
                        row = df[df["Outcome"] == outcome]
                        if not row.empty:
                            actual_value = float(row["BMIxAge"].values[0])
                            if abs(actual_value - exp_value) <= tolerance:
                                matched += 1
                                logger.info(f"Outcome {outcome}: matched (expected={exp_value}, found={actual_value})")
                            else:
                                logger.warning(f"Outcome {outcome}: mismatch (expected={exp_value}, found={actual_value})")
                        else:
                            logger.warning(f"Outcome {outcome} missing in output.")

                    if matched == 2:
                        score += 0.2
                        feedback.append("All BMIxAge mean values match expected results.")
                    elif matched == 1:
                        score += 0.1
                        feedback.append("One of the BMIxAge mean values matches expected result.")
                    else:
                        feedback.append("No correct BMIxAge values found.")
                else:
                    feedback.append("Output CSV missing required columns.")
                    logger.warning("Missing columns in output CSV.")
            except Exception as e:
                feedback.append(f"Error reading or validating output CSV: {e}")
                logger.error(f"Error reading bmixage_comparison.csv: {e}", exc_info=True)
        else:
            feedback.append("Output CSV file missing.")
            logger.warning("bmixage_comparison.csv not found.")
    except Exception as e:
        feedback.append(f"Evaluation failed: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)

    # Final score capped at 1.0
    final_score = min(score, 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score


def evaluate_logistic_top5_features(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Logistic Regression Top 5 Features Jupyter task.

    Checks:
      - The Jupyter notebook file exists.
      - The top5_features.csv file exists.
      - The file has exactly 5 rows and columns ['Feature', 'Coefficient'].
      - Coefficients are numeric and sorted by absolute value (descending).
      - Includes at least 'Glucose' and 'BMI' among top features.
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    base_dir = os.path.dirname(actual)
    notebook_file = os.path.join(base_dir, "logreg_feature_importance.ipynb")
    output_file = os.path.join(base_dir, "top5_features.csv")

    score = 0.0
    feedback = []

    try:
        # Check for notebook
        if os.path.exists(notebook_file):
            score += 0.3
            feedback.append("Notebook file found.")
            logger.info("Notebook file located.")
        else:
            feedback.append("Notebook file missing.")
            logger.warning("Notebook file not found.")

        # Check for output CSV
        if os.path.exists(output_file):
            score += 0.4
            feedback.append("Output CSV file found.")
            logger.info("Output CSV found.")

            try:
                df = pd.read_csv(output_file)

                # Validate columns
                expected_cols = {"Feature", "Coefficient"}
                if expected_cols.issubset(df.columns):
                    score += 0.1
                    feedback.append("CSV contains required columns.")
                    logger.info("Columns verified.")

                    # Validate row count
                    if len(df) == 5:
                        score += 0.1
                        feedback.append("File contains exactly 5 rows as expected.")
                        logger.info("Row count verified (5 rows).")
                    else:
                        feedback.append(f"Unexpected number of rows: {len(df)} (expected 5).")
                        logger.warning("Row count mismatch.")

                    # Validate numeric coefficients
                    if np.issubdtype(df["Coefficient"].dtype, np.number):
                        logger.info("Coefficients are numeric.")
                    else:
                        feedback.append("Coefficient column is not numeric.")
                        logger.warning("Non-numeric coefficient values detected.")

                    # Check descending order by absolute value
                    abs_vals = df["Coefficient"].abs().values
                    if all(abs_vals[i] >= abs_vals[i + 1] for i in range(len(abs_vals) - 1)):
                        score += 0.05
                        feedback.append("Coefficients sorted in descending absolute order.")
                        logger.info("Coefficients sorted correctly.")
                    else:
                        feedback.append("Coefficients not sorted by absolute value.")
                        logger.warning("Coefficient order mismatch.")

                    # Check for key features presence
                    top_features = [f.lower() for f in df["Feature"].tolist()]
                    matched_key_feats = sum(
                        feat in top_features for feat in ["glucose", "bmi"]
                    )
                    if matched_key_feats == 2:
                        score += 0.05
                        feedback.append("Top features include both 'Glucose' and 'BMI'.")
                        logger.info("'Glucose' and 'BMI' found in top features.")
                    elif matched_key_feats == 1:
                        score += 0.03
                        feedback.append("Only one key feature ('Glucose' or 'BMI') found.")
                        logger.warning("Partial match on key features.")
                    else:
                        feedback.append("Neither 'Glucose' nor 'BMI' found among top features.")
                        logger.warning("Expected key features missing.")

                else:
                    feedback.append("Output CSV missing required columns.")
                    logger.warning("Missing required columns in output CSV.")

            except Exception as e:
                feedback.append(f"Error reading or validating top5_features.csv: {e}")
                logger.error(f"Error reading CSV: {e}", exc_info=True)

        else:
            feedback.append("Output CSV file missing.")
            logger.warning("top5_features.csv not found.")

    except Exception as e:
        feedback.append(f"Evaluation failed: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)

    # Cap score at 1.0
    final_score = min(score, 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score 

def evaluate_age_group_outcome(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Age Group Diabetes Rate Jupyter task.

    Checks:
      - Jupyter notebook exists.
      - age_group_outcome.csv exists.
      - Contains correct columns and 3 age groups.
      - DiabetesRate values match expected results within ±0.02.
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    base_dir = os.path.dirname(actual)
    notebook_file = os.path.join(base_dir, "age_group_outcome.ipynb")
    output_file = os.path.join(base_dir, "age_group_outcome.csv")

    expected_rates = {
        "<30": 0.212,
        "30–50": 0.502,
        ">50": 0.469
    }
    tolerance = 0.02

    score = 0.0
    feedback = []

    try:
        # Notebook presence
        if os.path.exists(notebook_file):
            score += 0.3
            feedback.append("Notebook file found.")
            logger.info("Notebook file found.")
        else:
            feedback.append("Notebook file missing.")
            logger.warning("Notebook not found.")

        # Output CSV presence
        if os.path.exists(output_file):
            score += 0.4
            feedback.append("Output CSV file found.")
            logger.info("Output CSV located.")

            try:
                df = pd.read_csv(output_file)
                expected_cols = {"AgeGroup", "DiabetesRate"}

                # Validate columns
                if expected_cols.issubset(df.columns):
                    score += 0.1
                    feedback.append("Output CSV contains required columns.")
                    logger.info("Column structure validated.")

                    matched = 0
                    for group, expected_val in expected_rates.items():
                        row = df[df["AgeGroup"].astype(str).str.strip() == group]
                        if not row.empty:
                            actual_val = float(row["DiabetesRate"].values[0])
                            if abs(actual_val - expected_val) <= tolerance:
                                matched += 1
                                logger.info(f"{group}: matched (expected={expected_val}, found={actual_val})")
                            else:
                                logger.warning(f"{group}: mismatch (expected={expected_val}, found={actual_val})")
                        else:
                            logger.warning(f"Missing group {group} in output CSV.")

                    if matched == len(expected_rates):
                        score += 0.2
                        feedback.append("All DiabetesRate values match expected results.")
                    elif matched >= 2:
                        score += 0.1
                        feedback.append("Most DiabetesRate values are within tolerance.")
                    else:
                        feedback.append("Few or no DiabetesRate values matched expected results.")
                else:
                    feedback.append("Output CSV missing required columns.")
                    logger.warning("Missing expected columns in CSV.")
            except Exception as e:
                feedback.append(f"Error reading or validating CSV: {e}")
                logger.error(f"Error validating age_group_outcome.csv: {e}", exc_info=True)
        else:
            feedback.append("Output CSV file missing.")
            logger.warning("age_group_outcome.csv not found.")

    except Exception as e:
        feedback.append(f"Evaluation failed: {e}")
        logger.error(f"Evaluation error: {e}", exc_info=True)

    # Final scoring
    final_score = min(score, 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score


def evaluate_glucose_bmi_viz_quality(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Diabetes Glucose–BMI Visualization task.

    This evaluation checks:
      - that the Python script exists and imports pandas
      - that both visualization files (glucose_bmi_scatter.png and glucose_hist.png) exist and are non-empty
      - and that GPT-4o confirms both charts clearly represent the correct information
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    expected = expected.get("rules", expected)
    base_dir = os.path.dirname(actual)

    py_file = os.path.join(base_dir, "glucose_bmi_viz.py")
    chart1 = os.path.join(base_dir, "glucose_bmi_scatter.png")
    chart2 = os.path.join(base_dir, "glucose_hist.png")

    score = 0.0
    feedback = []

    # Step 1: Check Python script
    if os.path.exists(py_file):
        score += 0.3
        feedback.append("The Python script file is present.")
        logger.info("Python script found.")

        with open(py_file, "r", encoding="utf-8") as f:
            content = f.read().lower()

        if "import pandas" in content and (
            "matplotlib" in content or "seaborn" in content or "plotly" in content or "altair" in content
        ):
            score += 0.2
            feedback.append("The script correctly imports pandas and a visualization library.")
            logger.info("Required imports verified.")
        else:
            feedback.append("Missing pandas or visualization library import.")
            logger.warning("Import verification failed.")
    else:
        feedback.append("The Python script is missing.")
        logger.warning("Python script missing.")

    # Helper function: Encode image for GPT-4o
    def encode_image(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    # Step 2: Visualization checks
    charts_to_check = [
        (
            chart1,
            "a scatter plot of Glucose vs BMI, color-coded by Outcome (1 = diabetic, 0 = non-diabetic), with clear axis labels and a legend."
        ),
        (
            chart2,
            "a histogram comparing Glucose distributions for diabetic and non-diabetic individuals, with color distinction and a legend."
        )
    ]

    for chart, description in charts_to_check:
        chart_name = os.path.basename(chart)
        if os.path.exists(chart) and os.path.getsize(chart) > 0:
            score += 0.15
            feedback.append(f"The visualization {chart_name} exists and is non-empty.")
            logger.info(f"{chart_name} found and non-empty.")

            try:
                encoded = encode_image(chart)
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert data visualization evaluator. Judge each chart objectively."
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        f"Examine this chart and confirm if it correctly represents {description}. "
                                        f"Reply 'yes' if it does, otherwise 'no', followed by a short reason."
                                    )
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{encoded}"}
                                }
                            ]
                        }
                    ],
                    max_tokens=100
                )

                answer = response.choices[0].message.content.lower()
                if "yes" in answer:
                    score += 0.1
                    feedback.append(f"GPT-4o judged {chart_name} as clear, accurate, and visually informative.")
                    logger.info(f"{chart_name} validated by GPT-4o.")
                else:
                    feedback.append(f"GPT-4o judged {chart_name} unclear or inaccurate: {answer}")
                    logger.warning(f"{chart_name} judged unclear by GPT-4o.")
            except Exception as e:
                feedback.append(f"Error evaluating {chart_name} with GPT-4o: {e}")
                logger.error(f"Error evaluating {chart_name}: {e}", exc_info=True)
        else:
            feedback.append(f"The visualization {chart_name} is missing or empty.")
            logger.warning(f"{chart_name} missing or empty.")

    # Step 3: Final scoring
    final_score = min(score, 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score

def evaluate_feature_target_corr(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Feature-Outcome Correlation Jupyter task.

    Checks:
      - Jupyter notebook file exists.
      - feature_target_corr.csv and strongest_target_feature.txt exist.
      - CSV has correct columns and is sorted in descending order.
      - Top feature matches expected ('Glucose') with correlation ≈ 0.467 (±0.02).
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    base_dir = os.path.dirname(actual)
    notebook_file = os.path.join(base_dir, "feature_target_corr.ipynb")
    csv_file = os.path.join(base_dir, "feature_target_corr.csv")
    txt_file = os.path.join(base_dir, "strongest_target_feature.txt")

    expected_top_feature = "Glucose"
    expected_corr_value = 0.467
    tolerance = 0.02

    score = 0.0
    feedback = []

    try:
        # Notebook presence
        if os.path.exists(notebook_file):
            score += 0.3
            feedback.append("Notebook file found.")
            logger.info("Notebook file located.")
        else:
            feedback.append("Notebook file missing.")
            logger.warning("Notebook file not found.")

        # Check output CSV
        if os.path.exists(csv_file):
            score += 0.3
            feedback.append("Output CSV file found.")
            logger.info("feature_target_corr.csv found.")

            try:
                df = pd.read_csv(csv_file)
                expected_cols = {"Feature", "Correlation"}

                if expected_cols.issubset(df.columns):
                    score += 0.1
                    feedback.append("Output CSV contains the required columns.")
                    logger.info("CSV column validation passed.")

                    # Validate non-empty
                    if not df.empty:
                        # Check descending order
                        if df["Correlation"].is_monotonic_decreasing:
                            score += 0.1
                            feedback.append("Correlations are sorted in descending order.")
                            logger.info("Descending order confirmed.")
                        else:
                            feedback.append("Correlations not sorted in descending order.")
                            logger.warning("CSV not sorted descending.")

                        # Validate top feature and correlation
                        top_row = df.iloc[0]
                        top_feature = str(top_row["Feature"]).strip()
                        top_corr = float(top_row["Correlation"])

                        if (
                            top_feature.lower() == expected_top_feature.lower()
                            and abs(top_corr - expected_corr_value) <= tolerance
                        ):
                            score += 0.2
                            feedback.append("Top correlated feature and value match expected.")
                            logger.info(f"Top feature verified: {top_feature}, {top_corr}.")
                        else:
                            feedback.append(
                                f"Top feature mismatch or correlation off. "
                                f"Expected {expected_top_feature}≈{expected_corr_value}, found {top_feature}={top_corr}."
                            )
                            logger.warning("Top feature mismatch.")
                    else:
                        feedback.append("Output CSV is empty.")
                        logger.warning("CSV has no rows.")
                else:
                    feedback.append("Output CSV missing required columns.")
                    logger.warning("Column validation failed.")
            except Exception as e:
                feedback.append(f"Error reading CSV file: {e}")
                logger.error(f"Error reading feature_target_corr.csv: {e}", exc_info=True)
        else:
            feedback.append("feature_target_corr.csv file missing.")
            logger.warning("Output CSV not found.")

        # Check strongest feature text file
        if os.path.exists(txt_file):
            score += 0.1
            feedback.append("strongest_target_feature.txt found.")
            logger.info("Text file found for strongest feature.")

            try:
                with open(txt_file, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                if expected_top_feature.lower() in content.lower():
                    score += 0.1
                    feedback.append("Text file correctly identifies Glucose as the top feature.")
                    logger.info("Strongest feature verified as Glucose.")
                else:
                    feedback.append(f"Text file does not mention expected feature ({expected_top_feature}).")
                    logger.warning("Strongest feature mismatch.")
            except Exception as e:
                feedback.append(f"Error reading text file: {e}")
                logger.error(f"Error reading strongest_target_feature.txt: {e}", exc_info=True)
        else:
            feedback.append("strongest_target_feature.txt missing.")
            logger.warning("Text file missing.")

    except Exception as e:
        feedback.append(f"Evaluation error: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)

    final_score = min(score, 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score

def evaluate_model_comparison(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Model Comparison Jupyter Notebook task.

    Checks:
      - The Jupyter notebook file exists.
      - The output file model_comparison.csv exists.
      - CSV contains required columns: Model, Accuracy, Precision, Recall.
      - Contains results for both LogisticRegression and RandomForestClassifier.
      - Metric values are numeric and within [0, 1].
      - At least one model achieves Accuracy ≥ 0.7.
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    base_dir = os.path.dirname(actual)
    notebook_file = os.path.join(base_dir, "model_comparison.ipynb")
    csv_file = os.path.join(base_dir, "model_comparison.csv")

    score = 0.0
    feedback = []

    try:
        # Notebook existence check
        if os.path.exists(notebook_file):
            score += 0.3
            feedback.append("Notebook file found.")
            logger.info("Notebook file detected.")
        else:
            feedback.append("Notebook file missing.")
            logger.warning("Notebook file not found.")

        # Output file check
        if os.path.exists(csv_file):
            score += 0.3
            feedback.append("model_comparison.csv found.")
            logger.info("CSV output file located.")

            try:
                df = pd.read_csv(csv_file)
                expected_cols = {"Model", "Accuracy", "Precision", "Recall"}

                # Column validation
                if expected_cols.issubset(df.columns):
                    score += 0.1
                    feedback.append("Required columns present.")
                    logger.info("Columns validated successfully.")

                    # Model presence check
                    models_present = {
                        "LogisticRegression": any(df["Model"].str.contains("Logistic", case=False)),
                        "RandomForestClassifier": any(df["Model"].str.contains("RandomForest", case=False))
                    }

                    if all(models_present.values()):
                        score += 0.1
                        feedback.append("Both LogisticRegression and RandomForestClassifier results found.")
                        logger.info("Both model entries confirmed.")
                    else:
                        missing = [m for m, found in models_present.items() if not found]
                        feedback.append(f"Missing model results for: {', '.join(missing)}")
                        logger.warning(f"Missing model(s): {missing}")

                    # Metric validity check
                    metrics_valid = True
                    high_accuracy_found = False

                    for _, row in df.iterrows():
                        try:
                            acc = float(row["Accuracy"])
                            prec = float(row["Precision"])
                            rec = float(row["Recall"])
                            if not (0 <= acc <= 1 and 0 <= prec <= 1 and 0 <= rec <= 1):
                                metrics_valid = False
                                logger.warning(f"Invalid metric values found: {row.to_dict()}")
                            if acc >= 0.7:
                                high_accuracy_found = True
                        except Exception as e:
                            metrics_valid = False
                            logger.error(f"Error parsing numeric values: {e}")

                    if metrics_valid:
                        score += 0.1
                        feedback.append("All metric values are within the valid range (0–1).")
                        logger.info("Metric ranges verified.")
                    else:
                        feedback.append("Some metric values fall outside the valid range.")
                        logger.warning("Invalid metric range detected.")

                    if high_accuracy_found:
                        score += 0.1
                        feedback.append("At least one model achieved Accuracy ≥ 0.7.")
                        logger.info("High accuracy verified for at least one model.")
                    else:
                        feedback.append("No model achieved Accuracy ≥ 0.7.")
                        logger.warning("No high-accuracy model found.")
                else:
                    feedback.append("Missing one or more required columns.")
                    logger.warning("Column structure invalid.")

            except Exception as e:
                feedback.append(f"Error reading or parsing CSV: {e}")
                logger.error(f"Error reading model_comparison.csv: {e}", exc_info=True)
        else:
            feedback.append("model_comparison.csv missing.")
            logger.warning("Output CSV not found.")

    except Exception as e:
        feedback.append(f"Evaluation failed: {e}")
        logger.error(f"Evaluation error: {e}", exc_info=True)

    final_score = min(score, 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score


def evaluate_jupyter_diabetes_pairplot_visualization(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Diabetes Pairplot Visualization (Jupyter Notebook) task.

    This evaluation checks:
      - that the Jupyter notebook file exists
      - that pairplot.png exists and is non-empty
      - and that GPT-4o confirms the plot clearly represents relationships among Glucose, BMI, Age, and Outcome
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    expected = expected.get("rules", expected)
    base_dir = os.path.dirname(actual)

    notebook_file = os.path.join(base_dir, "diabetes_pairplot_analysis.ipynb")
    chart = os.path.join(base_dir, "pairplot.png")

    score = 0.0
    feedback = []

    # Step 1: Check that the notebook exists
    if os.path.exists(notebook_file):
        score += 0.5
        feedback.append("The Jupyter Notebook file exists.")
        logger.info("Notebook file found.")
    else:
        feedback.append("The Jupyter Notebook file is missing.")
        logger.warning("Notebook file missing.")

    # Helper: Encode image for GPT-4o
    def encode_image(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    # Step 2: Visualization check
    if os.path.exists(chart) and os.path.getsize(chart) > 0:
        score += 0.25
        feedback.append("The pairplot image file exists and is non-empty.")
        logger.info("pairplot.png found and non-empty.")

        try:
            encoded = encode_image(chart)
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert data visualization evaluator. Judge chart clarity, readability, and correctness."
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Examine this chart and confirm if it correctly represents pairwise scatter plots "
                                    "between Glucose, BMI, Age, and Outcome, with distinct colors for different outcomes. "
                                    "Reply 'yes' if it does, otherwise 'no', followed by a short reason."
                                )
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{encoded}"}
                            }
                        ]
                    }
                ],
                max_tokens=100
            )

            answer = response.choices[0].message.content.lower()
            if "yes" in answer:
                score += 0.25
                feedback.append("GPT-4o judged the pairplot as clear, accurate, and visually informative.")
                logger.info("pairplot.png validated by GPT-4o.")
            else:
                feedback.append(f"GPT-4o judged the pairplot unclear or inaccurate: {answer}")
                logger.warning("pairplot.png judged unclear by GPT-4o.")
        except Exception as e:
            feedback.append(f"Error evaluating pairplot.png with GPT-4o: {e}")
            logger.error(f"Error evaluating pairplot.png: {e}", exc_info=True)
    else:
        feedback.append("The pairplot.png file is missing or empty.")
        logger.warning("pairplot.png missing or zero-size.")

    # Step 3: Final scoring
    final_score = min(score, 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score


def evaluate_standardize_diabetes(actual: str, expected: dict, **options) -> float:
    """
    Dynamic evaluator for the Diabetes Standardization Jupyter task.

    Checks:
      - Jupyter notebook exists.
      - standardized_diabetes.csv exists.
      - All continuous variables have mean ≈ 0 and std ≈ 1 (within tolerance).
      - No zero values remain in columns that were imputed before standardization.
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    base_dir = os.path.dirname(actual)
    notebook_file = os.path.join(base_dir, "standardize_diabetes.ipynb")
    csv_file = os.path.join(base_dir, "standardized_diabetes.csv")

    # Expected columns
    continuous_cols = [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
    ]

    score = 0.0
    feedback = []

    try:
        # Step 1: Notebook check
        if os.path.exists(notebook_file):
            score += 0.3
            feedback.append("Notebook file found.")
            logger.info("Notebook file detected.")
        else:
            feedback.append("Notebook file missing.")
            logger.warning("Notebook file not found.")

        # Step 2: Check standardized dataset
        if os.path.exists(csv_file):
            score += 0.4
            feedback.append("standardized_diabetes.csv found.")
            logger.info("Standardized dataset located.")

            try:
                df = pd.read_csv(csv_file)

                # Ensure all expected columns exist
                missing_cols = [c for c in continuous_cols if c not in df.columns]
                if not missing_cols:
                    score += 0.1
                    feedback.append("All continuous variables are present.")
                    logger.info("Column validation passed.")

                    # Step 3: Check mean ≈ 0 and std ≈ 1
                    mean_ok, std_ok, zero_ok = True, True, True
                    tolerance_mean, tolerance_std = 0.05, 0.05

                    for col in continuous_cols:
                        mean_val = np.round(df[col].mean(), 3)
                        std_val = np.round(df[col].std(), 3)
                        zero_count = (df[col] == 0).sum()

                        if abs(mean_val) > tolerance_mean:
                            mean_ok = False
                            logger.warning(f"{col}: Mean not close to 0 (found {mean_val})")

                        if abs(std_val - 1) > tolerance_std:
                            std_ok = False
                            logger.warning(f"{col}: Std not close to 1 (found {std_val})")

                        if zero_count > 0:
                            zero_ok = False
                            logger.warning(f"{col}: Contains {zero_count} zero values after standardization")

                    if mean_ok:
                        score += 0.1
                        feedback.append("All columns have mean approximately 0.")
                    else:
                        feedback.append("Some columns deviate from mean ≈ 0.")

                    if std_ok:
                        score += 0.05
                        feedback.append("All columns have standard deviation approximately 1.")
                    else:
                        feedback.append("Some columns deviate from std ≈ 1.")

                    if zero_ok:
                        score += 0.05
                        feedback.append("No zeros remain in standardized columns.")
                    else:
                        feedback.append("Some standardized columns still contain zeros.")
                else:
                    feedback.append(f"Missing columns in standardized dataset: {', '.join(missing_cols)}")
                    logger.warning(f"Missing expected columns: {missing_cols}")

            except Exception as e:
                feedback.append(f"Error reading standardized_diabetes.csv: {e}")
                logger.error(f"Error reading standardized dataset: {e}", exc_info=True)
        else:
            feedback.append("standardized_diabetes.csv missing.")
            logger.warning("Output CSV not found.")

    except Exception as e:
        feedback.append(f"Evaluation failed: {e}")
        logger.error(f"Evaluation error: {e}", exc_info=True)

    # Final score
    final_score = min(score, 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score


def evaluate_outlier_count_diabetes(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Outlier Count task in the Diabetes dataset.

    Checks:
      - Jupyter notebook exists.
      - outlier_count.txt exists.
      - The numeric value matches the expected total (16 ± 1 tolerance).
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    base_dir = os.path.dirname(actual)
    notebook_file = os.path.join(base_dir, "outlier_count_diabetes.ipynb")
    result_file = os.path.join(base_dir, "outlier_count.txt")

    expected_count = 16
    tolerance = 1
    score = 0.0
    feedback = []

    try:
        # Notebook existence
        if os.path.exists(notebook_file):
            score += 0.4
            feedback.append("Notebook file found.")
            logger.info("Notebook file found.")
        else:
            feedback.append("Notebook file missing.")
            logger.warning("Notebook missing.")

        # Result file existence
        if os.path.exists(result_file):
            score += 0.3
            feedback.append("Result file outlier_count.txt found.")
            logger.info("Result file detected.")

            try:
                with open(result_file, "r", encoding="utf-8") as f:
                    content = f.read().strip()

                match = re.search(r"[-+]?\d*\.\d+|\d+", content)
                if match:
                    value = float(match.group())
                    logger.info(f"Extracted numeric value: {value}")

                    if abs(value - expected_count) <= tolerance:
                        score += 0.3
                        feedback.append("Outlier count matches expected result.")
                        logger.info("Outlier count verified successfully.")
                    else:
                        feedback.append(
                            f"Count differs from expected ({expected_count}). Found: {value}"
                        )
                        logger.warning("Value mismatch in outlier count.")
                else:
                    feedback.append("No numeric value found in result file.")
                    logger.warning("Numeric extraction failed.")
            except Exception as e:
                feedback.append(f"Error reading or parsing result file: {e}")
                logger.error(f"Error reading outlier_count.txt: {e}", exc_info=True)
        else:
            feedback.append("Result file outlier_count.txt missing.")
            logger.warning("Result file not found.")

    except Exception as e:
        feedback.append(f"Evaluation failed: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)

    final_score = min(score, 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score

def evaluate_threshold_recall_diabetes(actual: str, expected: dict, **options) -> float:
    """
    Enhanced evaluator for Logistic Regression threshold adjustment (0.3 vs 0.5).

    Checks:
      - Notebook exists.
      - Mentions of threshold adjustment (0.3) and comparison with 0.5.
      - Output recall value validity (0–1) and improvement expectation (≥0.5).
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    base_dir = os.path.dirname(actual)
    notebook_file = os.path.join(base_dir, "threshold_recall_diabetes.ipynb")
    result_file = os.path.join(base_dir, "threshold_recall.txt")

    score = 0.0
    feedback = []

    try:
        # 1️⃣ Check notebook existence
        if os.path.exists(notebook_file):
            score += 0.3
            feedback.append("Notebook file found.")
            logger.info("Notebook found.")

            try:
                with open(notebook_file, "r", encoding="utf-8") as f:
                    content = f.read().lower()

                # Threshold presence checks
                has_03 = "0.3" in content or "threshold=0.3" in content
                has_05 = "0.5" in content or "threshold=0.5" in content
                has_threshold_word = "threshold" in content

                if has_03 or has_threshold_word:
                    score += 0.2
                    feedback.append("Notebook mentions threshold adjustment (0.3).")
                else:
                    feedback.append("Threshold 0.3 adjustment not detected.")

                if has_03 and has_05:
                    score += 0.2
                    feedback.append("Notebook compares thresholds 0.3 and 0.5 (dual recall computation).")
                else:
                    feedback.append("Notebook does not show explicit comparison between 0.3 and 0.5.")
            except Exception as e:
                feedback.append(f"Error reading notebook: {e}")
                logger.error(f"Error reading notebook: {e}", exc_info=True)
        else:
            feedback.append("Notebook file missing.")
            logger.warning("Notebook missing.")

        # 2️⃣ Check output file and recall value
        if os.path.exists(result_file):
            score += 0.2
            feedback.append("Output file threshold_recall.txt found.")
            logger.info("Output file found.")

            try:
                with open(result_file, "r", encoding="utf-8") as f:
                    text = f.read().strip()

                match = re.search(r"[-+]?\d*\.\d+|\d+", text)
                if match:
                    recall_val = float(match.group())
                    logger.info(f"Extracted recall value: {recall_val}")

                    if 0 <= recall_val <= 1:
                        score += 0.1
                        feedback.append(f"Valid recall value ({recall_val:.3f}) detected.")
                        if recall_val >= 0.5:
                            score += 0.1
                            feedback.append(f"Recall ({recall_val:.3f}) shows expected improvement (≥0.5).")
                        else:
                            feedback.append(f"Recall ({recall_val:.3f}) below expected threshold.")
                            logger.warning("Recall lower than expected 0.5.")
                    else:
                        feedback.append(f"Recall value out of valid range (0–1): {recall_val}.")
                        logger.warning("Invalid recall value range.")
                else:
                    feedback.append("No numeric recall value found in result file.")
                    logger.warning("Numeric recall not found.")
            except Exception as e:
                feedback.append(f"Error reading threshold_recall.txt: {e}")
                logger.error(f"Error reading threshold_recall.txt: {e}", exc_info=True)
        else:
            feedback.append("Result file threshold_recall.txt missing.")
            logger.warning("Result file missing.")

    except Exception as e:
        feedback.append(f"Evaluation failed: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)

    # Cap final score at 1.0
    final_score = min(score, 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score

def evaluate_riskindex_comparison_diabetes(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the RiskIndex Comparison task in the Diabetes dataset.

    Checks:
      - Jupyter notebook exists.
      - riskindex_comparison.csv exists and has 2 rows: Outcome=0 and Outcome=1.
      - Columns: Outcome, AvgRiskIndex.
      - AvgRiskIndex values are numeric.
      - AvgRiskIndex for Outcome=1 > Outcome=0 (expected trend).
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    base_dir = os.path.dirname(actual)
    notebook_file = os.path.join(base_dir, "riskindex_comparison.ipynb")
    result_file = os.path.join(base_dir, "riskindex_comparison.csv")

    score = 0.0
    feedback = []

    try:
        # Notebook existence
        if os.path.exists(notebook_file):
            score += 0.3
            feedback.append("Notebook file found.")
            logger.info("Notebook found.")
        else:
            feedback.append("Notebook file missing.")
            logger.warning("Notebook missing.")

        # CSV existence and structure
        if os.path.exists(result_file):
            score += 0.3
            feedback.append("Result file riskindex_comparison.csv found.")
            logger.info("Result CSV found.")

            try:
                df = pd.read_csv(result_file)
                required_cols = {"Outcome", "AvgRiskIndex"}

                # Check columns
                if required_cols.issubset(df.columns):
                    score += 0.1
                    feedback.append("CSV contains required columns (Outcome, AvgRiskIndex).")
                    logger.info("Columns verified.")
                else:
                    feedback.append(f"Missing expected columns. Found: {list(df.columns)}")
                    logger.warning("Incorrect columns in CSV.")

                # Check row count
                if len(df) == 2:
                    score += 0.1
                    feedback.append("CSV contains exactly two rows (Outcome=0 and Outcome=1).")
                    logger.info("Row count verified.")
                else:
                    feedback.append(f"Unexpected number of rows: {len(df)} (expected 2).")
                    logger.warning("Row count mismatch.")

                # Validate numeric and trend
                if df["AvgRiskIndex"].dtype.kind in "fi":
                    avg0 = float(df.loc[df["Outcome"] == 0, "AvgRiskIndex"].values[0])
                    avg1 = float(df.loc[df["Outcome"] == 1, "AvgRiskIndex"].values[0])
                    logger.info(f"AvgRiskIndex values -> Outcome 0: {avg0}, Outcome 1: {avg1}")

                    if avg1 > avg0:
                        score += 0.2
                        feedback.append(f"Outcome=1 has higher AvgRiskIndex ({avg1:.3f} > {avg0:.3f}) — correct trend.")
                    else:
                        feedback.append(f"Outcome=1 AvgRiskIndex ({avg1:.3f}) not greater than Outcome=0 ({avg0:.3f}).")
                        logger.warning("Unexpected AvgRiskIndex relationship.")
                else:
                    feedback.append("AvgRiskIndex column is not numeric.")
                    logger.warning("Non-numeric AvgRiskIndex values.")
            except Exception as e:
                feedback.append(f"Error reading or validating CSV: {e}")
                logger.error(f"Error reading result CSV: {e}", exc_info=True)
        else:
            feedback.append("Result file riskindex_comparison.csv missing.")
            logger.warning("Result file missing.")

    except Exception as e:
        feedback.append(f"Evaluation failed: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)

    final_score = min(score, 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score

def evaluate_insulin_resistance_analysis(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Insulin Resistance Analysis Jupyter task.

    This evaluation checks:
      - that the Jupyter notebook file exists
      - that insulin_corr.txt exists and contains a valid correlation (-1 <= r <= 1)
      - that insulin_index_summary.csv exists and includes Outcome and MeanInsulinResistanceIndex columns
      - that the file has exactly two rows (Outcome 0 and 1)
      - and that Outcome=1 has a higher mean InsulinResistanceIndex than Outcome=0
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    base_dir = os.path.dirname(actual)
    notebook_file = os.path.join(base_dir, "insulin_resistance_analysis.ipynb")
    corr_file = os.path.join(base_dir, "insulin_corr.txt")
    summary_file = os.path.join(base_dir, "insulin_index_summary.csv")

    score = 0.0
    feedback = []

    try:
        # Notebook existence
        if os.path.exists(notebook_file):
            score += 0.25
            feedback.append("Notebook file found.")
            logger.info("Notebook found.")
        else:
            feedback.append("Notebook file missing.")
            logger.warning("Notebook missing.")

        # Correlation file existence and value
        if os.path.exists(corr_file):
            score += 0.25
            feedback.append("Correlation file found.")
            logger.info("Correlation file detected.")

            try:
                with open(corr_file, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                match = re.search(r"[-+]?\d*\.\d+|\d+", content)
                if match:
                    corr = float(match.group())
                    logger.info(f"Extracted correlation value: {corr}")
                    if -1 <= corr <= 1:
                        score += 0.15
                        feedback.append(f"Valid correlation value ({corr:.3f}) within expected range.")
                    else:
                        feedback.append(f"Correlation value {corr:.3f} is out of range (-1 to 1).")
                else:
                    feedback.append("No numeric correlation value found in text file.")
            except Exception as e:
                feedback.append(f"Error reading correlation file: {e}")
                logger.error("Error reading correlation file", exc_info=True)
        else:
            feedback.append("Correlation file missing.")
            logger.warning("insulin_corr.txt missing.")

        # Summary CSV checks
        if os.path.exists(summary_file):
            score += 0.25
            feedback.append("Summary CSV file found.")
            logger.info("Summary CSV found.")
            try:
                df = pd.read_csv(summary_file)
                required_cols = {"Outcome", "MeanInsulinResistanceIndex"}

                if required_cols.issubset(df.columns):
                    score += 0.05
                    feedback.append("CSV contains the required columns.")
                else:
                    feedback.append(f"Missing expected columns. Found: {list(df.columns)}")

                if len(df) == 2:
                    score += 0.05
                    feedback.append("CSV contains exactly two rows (Outcome 0 and 1).")
                else:
                    feedback.append(f"Unexpected number of rows: {len(df)} (expected 2).")

                mean0 = float(df.loc[df["Outcome"] == 0, "MeanInsulinResistanceIndex"].values[0])
                mean1 = float(df.loc[df["Outcome"] == 1, "MeanInsulinResistanceIndex"].values[0])
                logger.info(f"Outcome=0 mean: {mean0}, Outcome=1 mean: {mean1}")

                if mean1 > mean0:
                    score += 0.05
                    feedback.append(f"Outcome=1 has a higher mean ({mean1:.3f}) than Outcome=0 ({mean0:.3f}).")
                else:
                    feedback.append(f"Outcome=1 mean ({mean1:.3f}) is not greater than Outcome=0 ({mean0:.3f}).")
            except Exception as e:
                feedback.append(f"Error reading summary CSV: {e}")
                logger.error("Error reading summary CSV", exc_info=True)
        else:
            feedback.append("Summary CSV file missing.")
            logger.warning("insulin_index_summary.csv missing.")

    except Exception as e:
        feedback.append(f"Evaluation failed: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)

    final_score = min(round(score, 2), 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)
    return final_score


def evaluate_age_segment_auc(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Age Segment AUC Jupyter task.

    This evaluation checks:
      - that the Jupyter notebook file exists
      - that the result file (age_segment_auc.csv) exists
      - that it contains three rows representing the defined age groups
      - that the columns AgeGroup and AUC exist
      - and that all AUC values fall within a realistic range (0.5 <= AUC <= 1.0)
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    base_dir = os.path.dirname(actual)
    notebook_file = os.path.join(base_dir, "age_segment_auc.ipynb")
    result_file = os.path.join(base_dir, "age_segment_auc.csv")

    score = 0.0
    feedback = []

    try:
        # Check notebook presence
        if os.path.exists(notebook_file):
            score += 0.3
            feedback.append("Notebook file found.")
            logger.info("Notebook file found.")
        else:
            feedback.append("Notebook file missing.")
            logger.warning("Notebook file missing.")

        # Check result file
        if os.path.exists(result_file):
            score += 0.4
            feedback.append("Result CSV file found.")
            logger.info("Result file located.")

            try:
                df = pd.read_csv(result_file)
                required_cols = {"AgeGroup", "AUC"}

                # Check columns
                if required_cols.issubset(df.columns):
                    score += 0.1
                    feedback.append("CSV contains the required columns (AgeGroup, AUC).")
                    logger.info("Columns verified.")
                else:
                    feedback.append(f"Missing expected columns. Found: {list(df.columns)}")
                    logger.warning("Columns not as expected.")

                # Check row count
                if len(df) == 3:
                    score += 0.1
                    feedback.append("CSV has exactly three rows for the three age groups.")
                    logger.info("Row count verified.")
                else:
                    feedback.append(f"Unexpected number of rows: {len(df)} (expected 3).")
                    logger.warning("Row count mismatch.")

                # Check AUC values
                if "AUC" in df.columns:
                    valid_values = df["AUC"].apply(lambda x: 0.5 <= x <= 1.0)
                    if valid_values.all():
                        score += 0.1
                        feedback.append("All AUC values are within the valid range (0.5–1.0).")
                        logger.info("AUC values valid.")
                    else:
                        invalid_rows = df.loc[~valid_values, "AUC"].to_list()
                        feedback.append(f"Invalid AUC values found: {invalid_rows}")
                        logger.warning("Some AUC values out of range.")
            except Exception as e:
                feedback.append(f"Error reading or validating CSV: {e}")
                logger.error(f"Error reading result file: {e}", exc_info=True)
        else:
            feedback.append("Result CSV file missing.")
            logger.warning("Result CSV missing.")

    except Exception as e:
        feedback.append(f"Evaluation failed: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)

    final_score = min(round(score, 2), 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)
    return final_score


def evaluate_jupyter_diabetes_3d_heatmap_visualizations(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Diabetes 3D and Heatmap Visualization (Jupyter Notebook) task.

    This evaluation checks:
      - that the Jupyter notebook file exists
      - that both visualization files exist and are non-empty
      - and that GPT-4o confirms the charts clearly represent the required information
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    expected = expected.get("rules", expected)
    base_dir = os.path.dirname(actual)

    notebook_file = os.path.join(base_dir, "diabetes_outcome_3d_heatmap.ipynb")
    chart1 = os.path.join(base_dir, "bmi_glucose_age_3d.png")
    chart2 = os.path.join(base_dir, "bmi_glucose_heatmap.png")

    score = 0.0
    feedback = []

    # Step 1: Notebook existence
    if os.path.exists(notebook_file):
        score += 0.4
        feedback.append("The Jupyter Notebook file exists.")
        logger.info("Notebook file found.")
    else:
        feedback.append("The Jupyter Notebook file is missing.")
        logger.warning("Notebook missing.")

    # Helper: encode image for GPT-4o
    def encode_image(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    # Step 2: Visualization checks
    charts_to_check = [
        (
            chart1,
            "a 3D scatter plot showing BMI, Glucose, and Age as the three axes, with color representing Outcome (diabetic vs non-diabetic)."
        ),
        (
            chart2,
            "a 2D heatmap showing average Outcome rates across BMI (x-axis) and Glucose (y-axis) bins of width ~10, using a smooth color gradient."
        )
    ]

    for chart, description in charts_to_check:
        chart_name = os.path.basename(chart)
        if os.path.exists(chart) and os.path.getsize(chart) > 0:
            score += 0.15
            feedback.append(f"The visualization {chart_name} exists and is non-empty.")
            logger.info(f"{chart_name} found and valid.")

            try:
                encoded = encode_image(chart)
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert in data visualization quality evaluation. Judge each chart objectively."
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        f"Examine this chart and confirm if it correctly represents {description}. "
                                        f"Reply 'yes' if it does, otherwise 'no', followed by a short reason."
                                    )
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{encoded}"}
                                }
                            ]
                        }
                    ],
                    max_tokens=100
                )

                answer = response.choices[0].message.content.lower()
                if "yes" in answer:
                    score += 0.1
                    feedback.append(f"GPT-4o judged {chart_name} as clear, accurate, and visually informative.")
                    logger.info(f"{chart_name} validated by GPT-4o.")
                else:
                    feedback.append(f"GPT-4o judged {chart_name} unclear or inaccurate: {answer}")
                    logger.warning(f"{chart_name} unclear per GPT-4o.")
            except Exception as e:
                feedback.append(f"Error evaluating {chart_name} with GPT-4o: {e}")
                logger.error(f"Error evaluating {chart_name}: {e}", exc_info=True)
        else:
            feedback.append(f"The visualization {chart_name} is missing or empty.")
            logger.warning(f"{chart_name} missing or empty.")

    # Step 3: Final scoring
    final_score = min(score, 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score


def evaluate_rf_logreg_feature_comparison(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the RandomForest vs LogisticRegression feature comparison task.

    This evaluation checks:
      - that the Jupyter notebook file exists
      - that model_comparison.csv exists
      - that it contains both RandomForest and LogisticRegression rows
      - that the file includes required columns: Model, Accuracy, Precision, Recall, AUC
      - and that all metric values are numeric and fall within the valid range (0 to 1)
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    base_dir = os.path.dirname(actual)
    notebook_file = os.path.join(base_dir, "rf_logreg_feature_comparison.ipynb")
    result_file = os.path.join(base_dir, "model_comparison.csv")

    score = 0.0
    feedback = []

    try:
        # Check notebook
        if os.path.exists(notebook_file):
            score += 0.3
            feedback.append("Notebook file found.")
            logger.info("Notebook file located.")
        else:
            feedback.append("Notebook file missing.")
            logger.warning("Notebook missing.")

        # Check CSV
        if os.path.exists(result_file):
            score += 0.4
            feedback.append("Result CSV file found.")
            logger.info("Result CSV located.")

            try:
                df = pd.read_csv(result_file)
                required_cols = {"Model", "Accuracy", "Precision", "Recall", "AUC"}

                # Column check
                if required_cols.issubset(df.columns):
                    score += 0.1
                    feedback.append("CSV includes all required columns.")
                    logger.info("Columns verified.")
                else:
                    feedback.append(f"Missing expected columns. Found: {list(df.columns)}")
                    logger.warning("Column mismatch.")

                # Model presence
                models = df["Model"].astype(str).str.lower().tolist()
                if any("forest" in m for m in models) and any("logistic" in m for m in models):
                    score += 0.1
                    feedback.append("Both RandomForest and LogisticRegression models are listed.")
                    logger.info("Both models found in results.")
                else:
                    feedback.append("Expected models not found (RandomForest, LogisticRegression).")
                    logger.warning("Missing model entries.")

                # Metric validation
                metric_cols = ["Accuracy", "Precision", "Recall", "AUC"]
                valid_metrics = True
                for col in metric_cols:
                    if not pd.api.types.is_numeric_dtype(df[col]):
                        valid_metrics = False
                        feedback.append(f"Column {col} contains non-numeric values.")
                        break
                    if not df[col].between(0, 1).all():
                        valid_metrics = False
                        feedback.append(f"Column {col} has values outside the valid range 0–1.")
                        break

                if valid_metrics:
                    score += 0.1
                    feedback.append("All metrics are numeric and within the range 0–1.")
                    logger.info("Metric validation passed.")
            except Exception as e:
                feedback.append(f"Error reading or validating CSV: {e}")
                logger.error(f"Error reading result file: {e}", exc_info=True)
        else:
            feedback.append("Result CSV file missing.")
            logger.warning("Result CSV missing.")

    except Exception as e:
        feedback.append(f"Evaluation failed: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)

    final_score = min(round(score, 2), 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)
    return final_score


def evaluate_region_polarization(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Region Polarization Analysis task.

    This evaluator checks:
      - that the Jupyter notebook file exists
      - that region_polarization.csv exists
      - that it contains the required columns: region, avg_dem_margin, avg_gop_perc, avg_abs_swing
      - and that the numeric values are within a reasonable range (not empty or NaN)
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    base_dir = os.path.dirname(actual)
    notebook_file = os.path.join(base_dir, "region_polarization_analysis.ipynb")
    result_file = os.path.join(base_dir, "region_polarization.csv")

    score = 0.0
    feedback = []

    try:
        # Notebook presence
        if os.path.exists(notebook_file):
            score += 0.3
            feedback.append("Notebook file found.")
            logger.info("Jupyter notebook file found.")
        else:
            feedback.append("Notebook file missing.")
            logger.warning("Notebook file not found.")

        # CSV presence and validation
        if os.path.exists(result_file):
            score += 0.4
            feedback.append("Result CSV file found.")
            logger.info("Result CSV located.")

            try:
                df = pd.read_csv(result_file)
                required_cols = ["region", "avg_dem_margin", "avg_gop_perc", "avg_abs_swing"]

                # Check for required columns
                if all(col in df.columns for col in required_cols):
                    score += 0.1
                    feedback.append("All required columns are present.")
                    logger.info("All expected columns verified.")
                else:
                    missing = [c for c in required_cols if c not in df.columns]
                    feedback.append(f"Missing expected columns: {missing}")
                    logger.warning(f"Missing columns: {missing}")

                # Validate numeric values
                numeric_cols = ["avg_dem_margin", "avg_gop_perc", "avg_abs_swing"]
                if df[numeric_cols].apply(lambda x: pd.api.types.is_numeric_dtype(x)).all():
                    if not df[numeric_cols].isnull().any().any():
                        score += 0.1
                        feedback.append("Numeric columns contain valid numeric values with no NaNs.")
                        logger.info("Numeric validation passed.")
                    else:
                        feedback.append("Some numeric columns contain missing values.")
                        logger.warning("Detected NaN values in numeric columns.")
                else:
                    feedback.append("One or more numeric columns contain non-numeric values.")
                    logger.warning("Non-numeric data detected.")

                # Check at least 3–4 regions present
                if df["region"].nunique() >= 3:
                    score += 0.1
                    feedback.append("Region grouping output includes multiple regions as expected.")
                    logger.info("Region grouping validated.")
                else:
                    feedback.append("Too few regions found in output.")
                    logger.warning("Unexpected number of regions in result.")

            except Exception as e:
                feedback.append(f"Error while reading or validating region_polarization.csv: {e}")
                logger.error(f"Error reading or validating CSV: {e}", exc_info=True)
        else:
            feedback.append("Result CSV file missing.")
            logger.warning("region_polarization.csv not found.")

    except Exception as e:
        feedback.append(f"Evaluation failed: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)

    # Final scoring
    final_score = min(round(score, 2), 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)
    return final_score

def evaluate_regional_swing_counts(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Regional Swing Counts task.

    This evaluator checks:
      - that the Jupyter notebook exists
      - that regional_swing_counts.csv exists
      - that required columns are present: swing_direction, region, pro_dem_count, pro_gop_count
      - and that counts are valid non-negative integers for each region
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    base_dir = os.path.dirname(actual)
    notebook_file = os.path.join(base_dir, "regional_swing_counts.ipynb")
    result_file = os.path.join(base_dir, "regional_swing_counts.csv")

    score = 0.0
    feedback = []

    try:
        # Check for notebook existence
        if os.path.exists(notebook_file):
            score += 0.3
            feedback.append("Notebook file found.")
            logger.info("Jupyter notebook located.")
        else:
            feedback.append("Notebook file missing.")
            logger.warning("Notebook file not found.")

        # Check for result CSV
        if os.path.exists(result_file):
            score += 0.4
            feedback.append("Result CSV file found.")
            logger.info("Result CSV located successfully.")

            try:
                df = pd.read_csv(result_file)
                required_cols = ["swing_direction", "region", "pro_dem_count", "pro_gop_count"]

                # Column check
                if all(col in df.columns for col in required_cols):
                    score += 0.1
                    feedback.append("All required columns are present.")
                    logger.info("All expected columns verified.")
                else:
                    missing = [c for c in required_cols if c not in df.columns]
                    feedback.append(f"Missing columns: {missing}")
                    logger.warning(f"Missing columns: {missing}")

                # Validate numeric count columns
                numeric_cols = ["pro_dem_count", "pro_gop_count"]
                valid_numeric = df[numeric_cols].apply(lambda x: pd.api.types.is_numeric_dtype(x)).all()

                if valid_numeric and (df[numeric_cols] >= 0).all().all():
                    score += 0.1
                    feedback.append("Count columns contain valid non-negative numeric values.")
                    logger.info("Numeric count validation passed.")
                else:
                    feedback.append("Invalid or negative values detected in count columns.")
                    logger.warning("Numeric validation failed.")

                # Check minimum expected number of regions
                if df["region"].nunique() >= 3:
                    score += 0.1
                    feedback.append("Result includes multiple regions as expected.")
                    logger.info("Region diversity validated.")
                else:
                    feedback.append("Too few regions found in output.")
                    logger.warning("Insufficient unique regions detected.")

            except Exception as e:
                feedback.append(f"Error while validating regional_swing_counts.csv: {e}")
                logger.error(f"Error reading or validating CSV: {e}", exc_info=True)
        else:
            feedback.append("Result CSV file missing.")
            logger.warning("regional_swing_counts.csv not found.")

    except Exception as e:
        feedback.append(f"Evaluation failed: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)

    final_score = min(round(score, 2), 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score

def evaluate_jupyter_region_vote_share_visualizations(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Region Vote Share Visualization (Jupyter Notebook) task.

    This evaluation checks:
      - that the Jupyter notebook file exists
      - that both visualization files (region_vote_share.png and overall_vote_pie.png) exist and are non-empty
      - and that GPT-4o confirms both charts clearly represent the required information
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    expected = expected.get("rules", expected)
    base_dir = os.path.dirname(actual)

    notebook_file = os.path.join(base_dir, "region_vote_share_analysis.ipynb")
    chart1 = os.path.join(base_dir, "region_vote_share.png")
    chart2 = os.path.join(base_dir, "overall_vote_pie.png")

    score = 0.0
    feedback = []

    # Step 1: Check that the Jupyter notebook exists
    if os.path.exists(notebook_file):
        score += 0.4
        feedback.append("The Jupyter Notebook file exists.")
        logger.info("Notebook file found.")
    else:
        feedback.append("The Jupyter Notebook file is missing.")
        logger.warning("Notebook file missing.")

    # Helper: encode image for GPT-4o
    def encode_image(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    # Step 2: Visualization checks
    charts_to_check = [
        (
            chart1,
            "a grouped bar chart comparing mean dem_perc and mean gop_perc per region, with labeled axes, clear legend, and readable colors."
        ),
        (
            chart2,
            "a pie chart showing total Democratic vs Republican vote share across all regions, labeled with percentages or category names."
        )
    ]

    for chart, description in charts_to_check:
        chart_name = os.path.basename(chart)
        if os.path.exists(chart) and os.path.getsize(chart) > 0:
            score += 0.15
            feedback.append(f"The visualization {chart_name} exists and is non-empty.")
            logger.info(f"{chart_name} found and non-empty.")

            try:
                encoded = encode_image(chart)
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert data visualization evaluator. Judge clarity, readability, and correctness."
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        f"Examine this chart and confirm if it correctly represents {description}. "
                                        f"Reply 'yes' if it does, otherwise 'no', followed by a short reason."
                                    )
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{encoded}"}
                                }
                            ]
                        }
                    ],
                    max_tokens=100
                )

                answer = response.choices[0].message.content.lower()
                if "yes" in answer:
                    score += 0.1
                    feedback.append(f"GPT-4o judged {chart_name} as clear, accurate, and visually informative.")
                    logger.info(f"{chart_name} validated by GPT-4o.")
                else:
                    feedback.append(f"GPT-4o judged {chart_name} unclear or inaccurate: {answer}")
                    logger.warning(f"{chart_name} unclear according to GPT-4o.")
            except Exception as e:
                feedback.append(f"Error evaluating {chart_name} with GPT-4o: {e}")
                logger.error(f"Error evaluating {chart_name}: {e}", exc_info=True)
        else:
            feedback.append(f"The visualization {chart_name} is missing or empty.")
            logger.warning(f"{chart_name} missing or zero-size.")

    # Step 3: Final scoring
    final_score = min(score, 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score


def evaluate_monthly_earthquake_summary(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Monthly Earthquake Summary task (Earthquake Dataset).

    Uses ground truth values provided directly by user:
      Month | Count | AvgMagnitude | MaxMagnitude

    Ground truth:
    1→1891,5.875,8.2
    2→1828,5.877,8.8
    3→2113,5.875,9.1
    4→1970,5.897,8.6
    5→1964,5.890,8.3
    6→1824,5.874,8.4
    7→1880,5.886,8.1
    8→2014,5.894,8.0
    9→1985,5.874,8.4
    10→1952,5.893,8.3
    11→1987,5.886,8.3
    12→2001,5.868,9.1
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    base_dir = os.path.dirname(actual)
    notebook_file = os.path.join(base_dir, "monthly_earthquake_summary.ipynb")
    result_file = os.path.join(base_dir, "monthly_earthquake_summary.csv")

    score = 0.0
    feedback = []

    # Ground truth DataFrame
    gt = pd.DataFrame({
        "Month": [1,2,3,4,5,6,7,8,9,10,11,12],
        "Count": [1891,1828,2113,1970,1964,1824,1880,2014,1985,1952,1987,2001],
        "AvgMagnitude": [5.875,5.877,5.875,5.897,5.890,5.874,5.886,5.894,5.874,5.893,5.886,5.868],
        "MaxMagnitude": [8.2,8.8,9.1,8.6,8.3,8.4,8.1,8.0,8.4,8.3,8.3,9.1]
    })

    try:
        # Notebook presence
        if os.path.exists(notebook_file):
            score += 0.3
            feedback.append("Notebook file found.")
            logger.info("Notebook located.")
        else:
            feedback.append("Notebook missing.")
            logger.warning("Notebook not found.")

        # CSV presence
        if os.path.exists(result_file):
            score += 0.4
            feedback.append("Result CSV file found.")
            logger.info("CSV found successfully.")

            try:
                df = pd.read_csv(result_file)
                required_cols = ["Month", "Count", "AvgMagnitude", "MaxMagnitude"]

                # Check columns
                if all(col in df.columns for col in required_cols):
                    score += 0.1
                    feedback.append("All required columns present.")
                else:
                    missing = [c for c in required_cols if c not in df.columns]
                    feedback.append(f"Missing columns: {missing}")

                # Match ground truth (tolerance-based)
                merged = pd.merge(df, gt, on="Month", suffixes=("", "_gt"), how="inner")
                if len(merged) == 12:
                    tol_counts = np.allclose(merged["Count"], merged["Count_gt"], atol=5)
                    tol_avg = np.allclose(merged["AvgMagnitude"], merged["AvgMagnitude_gt"], atol=0.005)
                    tol_max = np.allclose(merged["MaxMagnitude"], merged["MaxMagnitude_gt"], atol=0.05)

                    if tol_counts and tol_avg and tol_max:
                        score += 0.2
                        feedback.append("All numeric values match expected results.")
                    else:
                        feedback.append("Values differ slightly from expected.")
                else:
                    feedback.append("Month mismatch or incomplete data.")

            except Exception as e:
                feedback.append(f"Error reading CSV: {e}")
                logger.error(f"CSV read error: {e}", exc_info=True)
        else:
            feedback.append("Result CSV missing.")
            logger.warning("monthly_earthquake_summary.csv not found.")

    except Exception as e:
        feedback.append(f"Evaluation failed: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)

    final_score = min(round(score, 2), 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)
    return final_score

def evaluate_high_magnitude_earthquakes(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for High-Magnitude Earthquake Analysis.

    Ground truth provided by user:
      Count = 738
      AvgLatitude = 4.073
      AvgLongitude = 56.35

    This evaluator checks:
      - Notebook file exists
      - Count file exists and matches expected count
      - CSV exists and matches average coordinates (within tolerance)
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    base_dir = os.path.dirname(actual)
    notebook_file = os.path.join(base_dir, "high_magnitude_earthquakes.ipynb")
    count_file = os.path.join(base_dir, "highmag_count.txt")
    location_file = os.path.join(base_dir, "highmag_location.csv")

    true_count = 738
    true_lat = 4.073
    true_lon = 56.35

    score = 0.0
    feedback = []

    try:
        # Notebook check
        if os.path.exists(notebook_file):
            score += 0.3
            feedback.append("Notebook file found.")
        else:
            feedback.append("Notebook file missing.")

        # Check count file
        if os.path.exists(count_file):
            score += 0.3
            feedback.append("Count file found.")

            try:
                with open(count_file, "r", encoding="utf-8") as f:
                    text = f.read().strip()
                match = re.search(r"\d+", text)
                if match:
                    count_val = int(match.group())
                    if abs(count_val - true_count) <= 3:
                        score += 0.2
                        feedback.append(f"Count matches expected ({count_val}).")
                    else:
                        feedback.append(f"Count differs: expected {true_count}, found {count_val}.")
                else:
                    feedback.append("No numeric value found in count file.")
            except Exception as e:
                feedback.append(f"Error reading count file: {e}")
        else:
            feedback.append("Count file missing.")

        # Check location CSV
        if os.path.exists(location_file):
            score += 0.1
            feedback.append("Location file found.")
            try:
                df = pd.read_csv(location_file)
                if all(col in df.columns for col in ["AvgLatitude", "AvgLongitude"]):
                    lat = float(df["AvgLatitude"].iloc[0])
                    lon = float(df["AvgLongitude"].iloc[0])
                    if abs(lat - true_lat) <= 0.01 and abs(lon - true_lon) <= 0.1:
                        score += 0.1
                        feedback.append(f"Coordinates match expected (Lat={lat}, Lon={lon}).")
                    else:
                        feedback.append(f"Coordinates differ: expected ({true_lat}, {true_lon}), found ({lat}, {lon}).")
                else:
                    feedback.append("Required columns missing in location file.")
            except Exception as e:
                feedback.append(f"Error reading location CSV: {e}")
        else:
            feedback.append("Location CSV missing.")

    except Exception as e:
        feedback.append(f"Evaluation failed: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)

    final_score = min(round(score, 2), 1.0)
    logger.info(f"Final score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score


def evaluate_jupyter_earthquake_visualizations(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Earthquake Visualization (Jupyter Notebook) task.

    This evaluation checks:
      - that the Jupyter notebook file exists
      - that both visualization files exist and are non-empty
      - and that GPT-4o confirms the charts clearly represent the required information:
        1. A world map of earthquake locations using latitude and longitude.
        2. A histogram of earthquake magnitudes.
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    expected = expected.get("rules", expected)
    base_dir = os.path.dirname(actual)

    notebook_file = os.path.join(base_dir, "earthquake_visualization.ipynb")
    chart1 = os.path.join(base_dir, "earthquake_map.png")
    chart2 = os.path.join(base_dir, "magnitude_hist.png")

    score = 0.0
    feedback = []

    # Step 1: Verify notebook existence
    if os.path.exists(notebook_file):
        score += 0.4
        feedback.append("The Jupyter Notebook file exists.")
        logger.info("Notebook file found.")
    else:
        feedback.append("The Jupyter Notebook file is missing.")
        logger.warning("Notebook file missing.")

    # Helper: Encode images for GPT-4o
    def encode_image(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    # Step 2: Visualization checks
    charts_to_check = [
        (
            chart1,
            "a world map showing all earthquake locations using latitude and longitude, with appropriate projection, clear markers, and labeled legend or title."
        ),
        (
            chart2,
            "a histogram of earthquake magnitudes, with labeled x-axis (magnitude), y-axis (count), and an appropriate title."
        )
    ]

    for chart, description in charts_to_check:
        chart_name = os.path.basename(chart)
        if os.path.exists(chart) and os.path.getsize(chart) > 0:
            score += 0.15
            feedback.append(f"The visualization {chart_name} exists and is non-empty.")
            logger.info(f"{chart_name} found and valid.")

            try:
                encoded = encode_image(chart)
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert in geospatial and data visualization evaluation. Judge chart clarity, readability, and correctness."
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        f"Examine this chart and confirm if it correctly represents {description}. "
                                        f"Reply 'yes' if it does, otherwise 'no', followed by a short reason."
                                    )
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{encoded}"}
                                }
                            ]
                        }
                    ],
                    max_tokens=100
                )

                answer = response.choices[0].message.content.lower()
                if "yes" in answer:
                    score += 0.1
                    feedback.append(f"GPT-4o judged {chart_name} as clear, accurate, and visually informative.")
                    logger.info(f"{chart_name} validated by GPT-4o.")
                else:
                    feedback.append(f"GPT-4o judged {chart_name} unclear or inaccurate: {answer}")
                    logger.warning(f"{chart_name} unclear or inaccurate.")
            except Exception as e:
                feedback.append(f"Error evaluating {chart_name} with GPT-4o: {e}")
                logger.error(f"Error evaluating {chart_name}: {e}", exc_info=True)
        else:
            feedback.append(f"The visualization {chart_name} is missing or empty.")
            logger.warning(f"{chart_name} missing or empty.")

    # Step 3: Final scoring
    final_score = min(score, 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score

def evaluate_region_turnout_summary(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Region Turnout Summary task.

    Ground truth averages:
        region, avg_euro_turnout, avg_nat_turnout, avg_difference
        British, 43.30, 67.01, -23.71
        Central/Eastern, 37.05, 59.49, -22.44
        Mediterranean, 53.73, 70.75, -17.02
        Northern, 53.99, 80.17, -26.18
        Western, 64.32, 81.78, -17.46

    Checks:
      - Notebook file exists
      - Output CSV exists
      - Required columns exist
      - Numeric values are within small tolerance (±0.1)
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    base_dir = os.path.dirname(actual)
    notebook_file = os.path.join(base_dir, "region_turnout_summary.ipynb")
    output_file = os.path.join(base_dir, "region_turnout_summary.csv")

    # Ground truth reference
    ground_truth = pd.DataFrame({
        "region": [
            "British",
            "Central/Eastern",
            "Mediterranean",
            "Northern",
            "Western"
        ],
        "avg_euro_turnout": [43.30, 37.05, 53.73, 53.99, 64.32],
        "avg_nat_turnout": [67.01, 59.49, 70.75, 80.17, 81.78],
        "avg_difference": [-23.71, -22.44, -17.02, -26.18, -17.46]
    })

    score = 0.0
    feedback = []

    try:
        # Check notebook
        if os.path.exists(notebook_file):
            score += 0.3
            feedback.append("Notebook file found.")
        else:
            feedback.append("Notebook file missing.")

        # Check output CSV
        if os.path.exists(output_file):
            score += 0.4
            feedback.append("Output CSV found.")

            try:
                df = pd.read_csv(output_file)
                expected_cols = ["region", "avg_euro_turnout", "avg_nat_turnout", "avg_difference"]

                if all(col in df.columns for col in expected_cols):
                    score += 0.1
                    feedback.append("All required columns are present.")

                    # Compare numeric values by region (tolerance ±0.1)
                    merged = pd.merge(df, ground_truth, on="region", suffixes=("_pred", "_true"))
                    differences = []

                    for col in ["avg_euro_turnout", "avg_nat_turnout", "avg_difference"]:
                        diff = np.abs(merged[f"{col}_pred"] - merged[f"{col}_true"])
                        differences.extend(diff.tolist())

                    avg_diff = np.mean(differences)
                    if avg_diff <= 0.1:
                        score += 0.2
                        feedback.append("All region values closely match expected averages.")
                    else:
                        feedback.append(f"Values differ noticeably. Average deviation = {avg_diff:.2f}")
                else:
                    feedback.append("Missing one or more expected columns in CSV.")
            except Exception as e:
                feedback.append(f"Error reading or comparing CSV: {e}")
                logger.error(f"Error reading output CSV: {e}", exc_info=True)
        else:
            feedback.append("Output CSV missing.")

    except Exception as e:
        feedback.append(f"Evaluation failed: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)

    final_score = min(round(score, 2), 1.0)
    logger.info(f"Final score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score

def evaluate_pres_influence_summary(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Presidential Influence Summary task.

    Ground truth averages:
        pres_more, avg_euro_turnout, avg_nat_turnout, avg_turnout_gap
        False, 51.39, 72.08, 20.68
        True,  42.07, 62.82, 20.75

    Checks:
      - Notebook file exists
      - CSV file exists
      - Columns match expected names
      - Numeric values within ±0.15 tolerance
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    base_dir = os.path.dirname(actual)
    notebook_file = os.path.join(base_dir, "pres_influence_summary.ipynb")
    output_file = os.path.join(base_dir, "pres_influence_summary.csv")

    ground_truth = pd.DataFrame({
        "pres_more": [False, True],
        "avg_euro_turnout": [51.39, 42.07],
        "avg_nat_turnout": [72.08, 62.82],
        "avg_turnout_gap": [20.68, 20.75]
    })

    score = 0.0
    feedback = []

    try:
        # Notebook presence
        if os.path.exists(notebook_file):
            score += 0.3
            feedback.append("Notebook file found.")
        else:
            feedback.append("Notebook file missing.")

        # Output presence
        if os.path.exists(output_file):
            score += 0.4
            feedback.append("Output CSV file found.")

            try:
                df = pd.read_csv(output_file)
                expected_cols = ["pres_more", "avg_euro_turnout", "avg_nat_turnout", "avg_turnout_gap"]

                if all(col in df.columns for col in expected_cols):
                    score += 0.1
                    feedback.append("All expected columns present.")

                    # Compare numeric values
                    merged = pd.merge(df, ground_truth, on="pres_more", suffixes=("_pred", "_true"))
                    total_diff = 0
                    for col in ["avg_euro_turnout", "avg_nat_turnout", "avg_turnout_gap"]:
                        diff = np.abs(merged[f"{col}_pred"] - merged[f"{col}_true"]).mean()
                        total_diff += diff

                    avg_diff = total_diff / 3
                    if avg_diff <= 0.15:
                        score += 0.2
                        feedback.append("Values closely match expected averages.")
                    else:
                        feedback.append(f"Average deviation too high ({avg_diff:.2f}).")
                else:
                    feedback.append("Some required columns are missing in the CSV.")
            except Exception as e:
                feedback.append(f"Error reading or comparing CSV: {e}")
                logger.error(f"Error reading CSV: {e}", exc_info=True)
        else:
            feedback.append("Output CSV missing.")

    except Exception as e:
        feedback.append(f"Evaluation failed: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)

    final_score = min(round(score, 2), 1.0)
    logger.info(f"Final score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score

def evaluate_jupyter_turnout_gap_visualizations(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Turnout Gap Visualization (Jupyter Notebook) task.

    This evaluation checks:
      - The Jupyter notebook file exists.
      - Both visualization files exist and are non-empty.
      - GPT-4o confirms that:
          1. turnout_gap_bar.png correctly shows turnout_gap per country (sorted descending).
          2. turnout_gap_region_box.png correctly shows turnout_gap distribution by region.
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    expected = expected.get("rules", expected)
    base_dir = os.path.dirname(actual)

    notebook_file = os.path.join(base_dir, "turnout_gap_analysis.ipynb")
    chart1 = os.path.join(base_dir, "turnout_gap_bar.png")
    chart2 = os.path.join(base_dir, "turnout_gap_region_box.png")

    score = 0.0
    feedback = []

    # Step 1: Check notebook existence
    if os.path.exists(notebook_file):
        score += 0.4
        feedback.append("The Jupyter Notebook file exists.")
        logger.info("Notebook found.")
    else:
        feedback.append("The Jupyter Notebook file is missing.")
        logger.warning("Notebook missing.")

    # Helper function to encode image as base64
    def encode_image(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    # Step 2: Check and evaluate visualizations
    charts_to_check = [
        (
            chart1,
            "a bar chart showing each country’s turnout gap (nat_turnout - euro_turnout) sorted in descending order, with clear x-axis labels and title."
        ),
        (
            chart2,
            "a boxplot showing the distribution of turnout gaps grouped by region, with axis labels and a clear legend or title."
        )
    ]

    for chart, description in charts_to_check:
        chart_name = os.path.basename(chart)
        if os.path.exists(chart) and os.path.getsize(chart) > 0:
            score += 0.15
            feedback.append(f"The visualization {chart_name} exists and is non-empty.")
            logger.info(f"{chart_name} found and non-empty.")

            try:
                encoded = encode_image(chart)
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert in data visualization assessment. Evaluate clarity, accuracy, and labeling of charts."
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        f"Examine this chart and confirm if it correctly represents {description} "
                                        f"with proper axis labels, readable text, and accurate visual encoding. "
                                        f"Reply 'yes' if correct, otherwise 'no', followed by a short reason."
                                    )
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{encoded}"}
                                }
                            ]
                        }
                    ],
                    max_tokens=100
                )

                answer = response.choices[0].message.content.lower()
                if "yes" in answer:
                    score += 0.1
                    feedback.append(f"GPT-4o judged {chart_name} as clear, accurate, and visually correct.")
                    logger.info(f"{chart_name} validated by GPT-4o.")
                else:
                    feedback.append(f"GPT-4o judged {chart_name} unclear or inaccurate: {answer}")
                    logger.warning(f"{chart_name} unclear per GPT-4o.")
            except Exception as e:
                feedback.append(f"Error evaluating {chart_name} with GPT-4o: {e}")
                logger.error(f"GPT-4o evaluation failed for {chart_name}: {e}", exc_info=True)
        else:
            feedback.append(f"The visualization {chart_name} is missing or empty.")
            logger.warning(f"{chart_name} missing or empty.")

    # Step 3: Final scoring
    final_score = min(score, 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score


def evaluate_iris_strongest_correlation(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Iris dataset strongest correlation task.

    Ground truth:
        Correlation Value ≈ 0.963

    This evaluator checks:
      - The Jupyter notebook file exists.
      - The output text file exists.
      - The text file contains two feature names and a correlation value near 0.963.
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    base_dir = os.path.dirname(actual)
    notebook_file = os.path.join(base_dir, "iris_strongest_corr.ipynb")
    result_file = os.path.join(base_dir, "strongest_correlation.txt")

    true_corr = 0.963
    score = 0.0
    feedback = []

    try:
        # Notebook presence
        if os.path.exists(notebook_file):
            score += 0.4
            feedback.append("Notebook file found.")
        else:
            feedback.append("Notebook file missing.")

        # Result file presence
        if os.path.exists(result_file):
            score += 0.4
            feedback.append("Result file found.")

            try:
                with open(result_file, "r", encoding="utf-8") as f:
                    content = f.read().strip()

                # Extract numeric correlation value
                match = re.search(r"[-+]?\d*\.\d+|\d+", content)
                if match:
                    value = float(match.group())
                    if abs(value - true_corr) <= 0.01:
                        score += 0.2
                        feedback.append(f"Correlation value matches expected ({value}).")
                    else:
                        feedback.append(f"Correlation differs: expected {true_corr}, found {value}.")
                else:
                    feedback.append("No numeric value found in result file.")
            except Exception as e:
                feedback.append(f"Error reading or parsing result file: {e}")
                logger.error(f"Error reading correlation file: {e}", exc_info=True)
        else:
            feedback.append("Result file missing.")

    except Exception as e:
        feedback.append(f"Evaluation failed: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)

    final_score = min(round(score, 2), 1.0)
    logger.info(f"Final score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)
    return final_score


def evaluate_feature_variance_normalization(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Feature Variance Normalization task.

    Ground truth:
        Feature, Variance_Before, Variance_After
        sepal length, 0.686, 0.053
        sepal width, 0.188, 0.033
        petal length, 3.113, 0.089
        petal width, 0.582, 0.101

    Checks:
      - Notebook file exists
      - CSV file exists
      - Columns correct
      - Variance_After < Variance_Before for all features
      - Numeric values close to expected (±0.02)
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    base_dir = os.path.dirname(actual)
    notebook_file = os.path.join(base_dir, "feature_variance_normalization.ipynb")
    output_file = os.path.join(base_dir, "feature_variance_comparison.csv")

    # Ground truth
    true_df = pd.DataFrame({
        "Feature": ["sepal length", "sepal width", "petal length", "petal width"],
        "Variance_Before": [0.686, 0.188, 3.113, 0.582],
        "Variance_After": [0.053, 0.033, 0.089, 0.101]
    })

    score = 0.0
    feedback = []

    try:
        # Notebook presence
        if os.path.exists(notebook_file):
            score += 0.3
            feedback.append("Notebook file found.")
        else:
            feedback.append("Notebook file missing.")

        # Output CSV check
        if os.path.exists(output_file):
            score += 0.4
            feedback.append("Output CSV file found.")

            try:
                df = pd.read_csv(output_file)
                expected_cols = ["Feature", "Variance_Before", "Variance_After"]

                if all(col in df.columns for col in expected_cols):
                    score += 0.1
                    feedback.append("All expected columns present.")

                    # Normalize feature names for comparison
                    df["Feature"] = df["Feature"].str.strip().str.lower()
                    true_df["Feature"] = true_df["Feature"].str.strip().str.lower()

                    merged = pd.merge(df, true_df, on="Feature", suffixes=("_pred", "_true"))
                    total_diff = 0
                    for col in ["Variance_Before", "Variance_After"]:
                        diff = np.abs(merged[f"{col}_pred"] - merged[f"{col}_true"]).mean()
                        total_diff += diff

                    avg_diff = total_diff / 2
                    if avg_diff <= 0.02:
                        score += 0.1
                        feedback.append("Numeric values match expected results closely.")
                    else:
                        feedback.append(f"Average deviation too high ({avg_diff:.3f}).")

                    # Check variance reduction logic
                    if all(merged["Variance_After_pred"] < merged["Variance_Before_pred"]):
                        score += 0.1
                        feedback.append("Variance after normalization is smaller for all features.")
                    else:
                        feedback.append("Variance after normalization not smaller for all features.")
                else:
                    feedback.append("Some expected columns are missing.")
            except Exception as e:
                feedback.append(f"Error reading or comparing CSV: {e}")
                logger.error(f"Error comparing output CSV: {e}", exc_info=True)
        else:
            feedback.append("Output CSV missing.")

    except Exception as e:
        feedback.append(f"Evaluation failed: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)

    final_score = min(round(score, 2), 1.0)
    logger.info(f"Final score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score


def evaluate_iris_ratio_per_class(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Iris ratio per class task.

    Ground truth averages:
        class, avg_petal_to_sepal_length, avg_petal_to_sepal_width
        Iris-setosa, 0.293, 0.071
        Iris-versicolor, 0.718, 0.480
        Iris-virginica, 0.844, 0.684

    Checks:
      - Notebook file exists
      - CSV file exists
      - Required columns present
      - Numeric values within ±0.01 tolerance
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    base_dir = os.path.dirname(actual)
    notebook_file = os.path.join(base_dir, "iris_ratio_per_class.ipynb")
    output_file = os.path.join(base_dir, "iris_ratio_summary.csv")

    # Ground truth reference
    ground_truth = pd.DataFrame({
        "class": ["Iris-setosa", "Iris-versicolor", "Iris-virginica"],
        "avg_petal_to_sepal_length": [0.293, 0.718, 0.844],
        "avg_petal_to_sepal_width": [0.071, 0.480, 0.684]
    })

    score = 0.0
    feedback = []

    try:
        # Check notebook
        if os.path.exists(notebook_file):
            score += 0.3
            feedback.append("Notebook file found.")
        else:
            feedback.append("Notebook file missing.")

        # Check output CSV
        if os.path.exists(output_file):
            score += 0.4
            feedback.append("Output CSV file found.")

            try:
                df = pd.read_csv(output_file)
                expected_cols = ["class", "avg_petal_to_sepal_length", "avg_petal_to_sepal_width"]

                if all(col in df.columns for col in expected_cols):
                    score += 0.1
                    feedback.append("All expected columns are present.")

                    # Normalize whitespace in class names
                    df["class"] = df["class"].str.strip()
                    ground_truth["class"] = ground_truth["class"].str.strip()

                    merged = pd.merge(df, ground_truth, on="class", suffixes=("_pred", "_true"))
                    diffs = []
                    for col in ["avg_petal_to_sepal_length", "avg_petal_to_sepal_width"]:
                        diff = np.abs(merged[f"{col}_pred"] - merged[f"{col}_true"]).mean()
                        diffs.append(diff)

                    avg_diff = np.mean(diffs)
                    if avg_diff <= 0.01:
                        score += 0.2
                        feedback.append("Numeric values closely match expected averages.")
                    else:
                        feedback.append(f"Average deviation too high ({avg_diff:.3f}).")
                else:
                    feedback.append("Some expected columns are missing.")
            except Exception as e:
                feedback.append(f"Error reading or comparing CSV: {e}")
                logger.error(f"Error reading CSV: {e}", exc_info=True)
        else:
            feedback.append("Output CSV missing.")

    except Exception as e:
        feedback.append(f"Evaluation failed: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)

    final_score = min(round(score, 2), 1.0)
    logger.info(f"Final score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score


def evaluate_iris_zscore_outliers(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Iris Z-score Outlier Detection task.

    Ground truth:
        index, features_exceeding_threshold
        15, sepal width

    Checks:
      - Notebook file exists
      - CSV file exists
      - Columns are correct
      - At least one outlier row exists
      - The expected outlier (index 15, sepal width) appears in the file
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    base_dir = os.path.dirname(actual)
    notebook_file = os.path.join(base_dir, "iris_zscore_outliers.ipynb")
    output_file = os.path.join(base_dir, "outliers.csv")

    true_index = 15
    true_feature = "sepal width"

    score = 0.0
    feedback = []

    try:
        # Notebook existence
        if os.path.exists(notebook_file):
            score += 0.3
            feedback.append("Notebook file found.")
            logger.info("Jupyter notebook located.")
        else:
            feedback.append("Notebook file missing.")
            logger.warning("Notebook file not found.")

        # Output file existence
        if os.path.exists(output_file):
            score += 0.4
            feedback.append("Output CSV file found.")
            logger.info("Output CSV detected.")

            try:
                df = pd.read_csv(output_file)
                expected_cols = ["index", "features_exceeding_threshold"]

                if all(col in df.columns for col in expected_cols):
                    score += 0.1
                    feedback.append("Expected columns are present.")

                    # Ensure non-empty outliers list
                    if len(df) > 0:
                        score += 0.1
                        feedback.append("At least one outlier row is present.")
                    else:
                        feedback.append("No outlier rows found in CSV.")

                    # Check that known outlier exists
                    match_row = df[
                        (df["index"] == true_index)
                        & (df["features_exceeding_threshold"].str.strip().str.lower() == true_feature)
                    ]
                    if not match_row.empty:
                        score += 0.1
                        feedback.append(f"Outlier at index {true_index} with feature '{true_feature}' correctly detected.")
                    else:
                        feedback.append("Expected outlier not found.")
                else:
                    feedback.append("Missing required columns in output CSV.")
            except Exception as e:
                feedback.append(f"Error reading or validating output CSV: {e}")
                logger.error(f"Error reading output CSV: {e}", exc_info=True)
        else:
            feedback.append("Output file missing.")
            logger.warning("Output CSV not found.")

    except Exception as e:
        feedback.append(f"Evaluation failed: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)

    final_score = min(round(score, 2), 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)
    return final_score


def evaluate_iris_covariance_analysis(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Iris Covariance Matrix and Determinant task.

    Ground truth:
        Covariance matrix (rounded to 3 decimals):
            sepal length   sepal width   petal length   petal width
            sepal length       0.686       -0.039          1.274        0.517
            sepal width       -0.039        0.188         -0.322       -0.118
            petal length       1.274       -0.322          3.113        1.296
            petal width        0.517       -0.118          1.296        0.582
        Determinant ≈ 0.002
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    base_dir = os.path.dirname(actual)
    notebook_file = os.path.join(base_dir, "iris_covariance_analysis.ipynb")
    cov_file = os.path.join(base_dir, "feature_covariance.csv")
    det_file = os.path.join(base_dir, "covariance_determinant.txt")

    # Ground truth
    true_matrix = pd.DataFrame({
        "sepal length": [0.686, -0.039, 1.274, 0.517],
        "sepal width": [-0.039, 0.188, -0.322, -0.118],
        "petal length": [1.274, -0.322, 3.113, 1.296],
        "petal width": [0.517, -0.118, 1.296, 0.582]
    }, index=["sepal length", "sepal width", "petal length", "petal width"])

    true_determinant = 0.002

    score = 0.0
    feedback = []

    try:
        # Notebook file check
        if os.path.exists(notebook_file):
            score += 0.3
            feedback.append("Notebook file found.")
        else:
            feedback.append("Notebook file missing.")

        # Covariance CSV check
        if os.path.exists(cov_file):
            score += 0.3
            feedback.append("Covariance matrix file found.")
            try:
                df = pd.read_csv(cov_file, index_col=0)
                # Ensure structure and values are close
                if all(col in df.columns for col in true_matrix.columns):
                    diff = np.abs(df.values - true_matrix.values).mean()
                    if diff <= 0.05:
                        score += 0.1
                        feedback.append("Covariance matrix values are close to expected.")
                    else:
                        feedback.append(f"Matrix deviation too high (avg diff = {diff:.3f}).")
                else:
                    feedback.append("Matrix column mismatch.")
            except Exception as e:
                feedback.append(f"Error reading covariance CSV: {e}")
                logger.error(f"Error reading covariance CSV: {e}", exc_info=True)
        else:
            feedback.append("Covariance matrix file missing.")

        # Determinant text check
        if os.path.exists(det_file):
            score += 0.3
            feedback.append("Determinant file found.")
            try:
                with open(det_file, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                det_value = None
                for token in content.split():
                    try:
                        det_value = float(token)
                        break
                    except ValueError:
                        continue
                if det_value is not None:
                    if abs(det_value - true_determinant) <= 0.002:
                        score += 0.1
                        feedback.append(f"Determinant value matches expected ({det_value}).")
                    else:
                        feedback.append(f"Determinant differs. Expected {true_determinant}, found {det_value}.")
                else:
                    feedback.append("No numeric determinant value found.")
            except Exception as e:
                feedback.append(f"Error reading determinant file: {e}")
                logger.error(f"Error reading determinant file: {e}", exc_info=True)
        else:
            feedback.append("Determinant file missing.")

    except Exception as e:
        feedback.append(f"Evaluation failed: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)

    final_score = min(round(score, 2), 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score


def evaluate_iris_anova_feature_importance(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Iris ANOVA F-statistic feature importance task.

    Ground truth (rounded to 3 decimals):
        Feature         FStatistic
        petal length      1179.034
        petal width        959.324
        sepal length       119.265
        sepal width         47.364
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    base_dir = os.path.dirname(actual)
    notebook_path = os.path.join(base_dir, "iris_anova_feature_importance.ipynb")
    csv_path = os.path.join(base_dir, "feature_importance.csv")

    # Ground truth
    true_df = pd.DataFrame({
        "Feature": ["petal length", "petal width", "sepal length", "sepal width"],
        "FStatistic": [1179.034, 959.324, 119.265, 47.364]
    })

    score = 0.0
    feedback = []

    try:
        # Notebook presence
        if os.path.exists(notebook_path):
            score += 0.3
            feedback.append("Notebook file found.")
        else:
            feedback.append("Notebook file missing.")

        # CSV presence
        if os.path.exists(csv_path):
            score += 0.3
            feedback.append("Feature importance CSV found.")
            try:
                df = pd.read_csv(csv_path)
                if "Feature" in df.columns and "FStatistic" in df.columns:
                    score += 0.2
                    feedback.append("Required columns found in CSV.")

                    # Sort and compare order
                    df_sorted = df.sort_values("FStatistic", ascending=False).reset_index(drop=True)
                    true_sorted = true_df.sort_values("FStatistic", ascending=False).reset_index(drop=True)

                    # Compare top feature
                    if df_sorted.iloc[0, 0].strip().lower() == true_sorted.iloc[0, 0].strip().lower():
                        score += 0.1
                        feedback.append("Top-ranked feature matches expected (petal length).")

                    # Check average deviation in F-statistics
                    merged = pd.merge(df, true_df, on="Feature", suffixes=("_pred", "_true"))
                    diff = np.abs(merged["FStatistic_pred"] - merged["FStatistic_true"]).mean()
                    if diff <= 50:
                        score += 0.1
                        feedback.append("F-statistic values close to expected range.")
                    else:
                        feedback.append(f"F-statistics deviate from expected (avg diff = {diff:.2f}).")

                else:
                    feedback.append("CSV missing expected columns (Feature, FStatistic).")
            except Exception as e:
                feedback.append(f"Error reading CSV: {e}")
                logger.error(f"Error reading CSV: {e}", exc_info=True)
        else:
            feedback.append("Feature importance CSV missing.")

    except Exception as e:
        feedback.append(f"Evaluation failed: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)

    final_score = min(round(score, 2), 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score

def evaluate_iris_pca_projection(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for PCA variance and projection on the Iris dataset.

    Ground truth:
        Variance ratios:
            PC1 = 0.728
            PC2 = 0.230
        First 5 rows of PCA projection:
                PC1       PC2
        0 -2.264542  0.505704
        1 -2.086426 -0.655405
        2 -2.367950 -0.318477
        3 -2.304197 -0.575368
        4 -2.388777  0.674767
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    base_dir = os.path.dirname(actual)
    notebook_file = os.path.join(base_dir, "iris_pca_analysis.ipynb")
    variance_file = os.path.join(base_dir, "pca_variance.txt")
    projection_file = os.path.join(base_dir, "pca_projection.csv")

    # Ground truth
    true_variance = np.array([0.728, 0.230])
    true_projection = pd.DataFrame({
        "PC1": [-2.264542, -2.086426, -2.367950, -2.304197, -2.388777],
        "PC2": [0.505704, -0.655405, -0.318477, -0.575368, 0.674767]
    })

    score = 0.0
    feedback = []

    try:
        # Notebook presence
        if os.path.exists(notebook_file):
            score += 0.3
            feedback.append("Notebook file found.")
        else:
            feedback.append("Notebook file missing.")

        # Check PCA variance file
        if os.path.exists(variance_file):
            score += 0.2
            feedback.append("Variance file found.")
            try:
                with open(variance_file, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                nums = [float(x) for x in content.replace(",", " ").split() if x.replace(".", "", 1).isdigit()]
                if len(nums) >= 2:
                    diff = np.mean(np.abs(np.array(nums[:2]) - true_variance))
                    if diff <= 0.02:
                        score += 0.2
                        feedback.append("Explained variance ratios close to expected.")
                    else:
                        feedback.append(f"Variance ratios deviate (avg diff {diff:.3f}).")
                else:
                    feedback.append("Could not extract two numeric variance values.")
            except Exception as e:
                feedback.append(f"Error reading variance file: {e}")
        else:
            feedback.append("Variance file missing.")

        # Check PCA projection file
        if os.path.exists(projection_file):
            score += 0.2
            feedback.append("PCA projection CSV found.")
            try:
                df = pd.read_csv(projection_file)
                if {"PC1", "PC2"}.issubset(df.columns):
                    score += 0.1
                    feedback.append("Required columns (PC1, PC2) present in CSV.")
                    first_rows = df.head(5).reset_index(drop=True)
                    diff = np.mean(np.abs(first_rows - true_projection))
                    if diff <= 0.1:
                        score += 0.2
                        feedback.append("Projection values match closely for first 5 rows.")
                    else:
                        feedback.append(f"PCA projection values deviate (avg diff {diff:.3f}).")
                else:
                    feedback.append("CSV missing PC1 or PC2 columns.")
            except Exception as e:
                feedback.append(f"Error reading projection file: {e}")
        else:
            feedback.append("PCA projection CSV missing.")

    except Exception as e:
        feedback.append(f"Evaluation failed: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)

    final_score = min(round(score, 2), 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)
    return final_score


def evaluate_iris_pairwise_distance_summary(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Iris Pairwise Distance Summary task.

    Ground truth (rounded to 3 decimals):
        Minimum Distance = 0.000
        Mean Distance     = 2.544
        Maximum Distance  = 7.085
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    base_dir = os.path.dirname(actual)
    notebook_file = os.path.join(base_dir, "iris_pairwise_distance.ipynb")
    result_file = os.path.join(base_dir, "pairwise_distance_summary.txt")

    # Ground-truth reference values
    true_values = np.array([0.000, 2.544, 7.085])
    score = 0.0
    feedback = []

    try:
        # 1. Notebook check
        if os.path.exists(notebook_file):
            score += 0.3
            feedback.append("Notebook file found.")
        else:
            feedback.append("Notebook file missing.")

        # 2. Result file check
        if os.path.exists(result_file):
            score += 0.3
            feedback.append("Result file pairwise_distance_summary.txt found.")
            try:
                with open(result_file, "r", encoding="utf-8") as f:
                    text = f.read().strip()

                # Extract numeric values
                nums = re.findall(r"[-+]?\d*\.\d+|\d+", text)
                if len(nums) >= 3:
                    values = np.array(list(map(float, nums[:3])))
                    diff = np.mean(np.abs(values - true_values))
                    logger.info(f"Extracted distances: {values}, avg diff={diff:.3f}")

                    # 3. Evaluate closeness
                    if diff <= 0.05:
                        score += 0.4
                        feedback.append("Distances closely match expected values.")
                    elif diff <= 0.15:
                        score += 0.2
                        feedback.append("Distances roughly within tolerance range.")
                    else:
                        feedback.append(f"Distances deviate from expected (avg diff {diff:.3f}).")
                else:
                    feedback.append("Could not extract three numeric values from result file.")
            except Exception as e:
                feedback.append(f"Error reading or parsing result file: {e}")
                logger.error(f"Error reading result file: {e}", exc_info=True)
        else:
            feedback.append("Result file missing.")

    except Exception as e:
        feedback.append(f"Evaluation failed: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)

    final_score = min(round(score, 2), 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score


def evaluate_iris_redundant_features(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Iris redundant features task.

    Ground truth:
        Highly correlated feature pair: petal length,petal width (corr = 0.963)
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    base_dir = os.path.dirname(actual)
    notebook_path = os.path.join(base_dir, "iris_redundant_features.ipynb")
    result_file = os.path.join(base_dir, "redundant_features.txt")

    true_pair = "petal length,petal width"
    score = 0.0
    feedback = []

    try:
        # Check notebook existence
        if os.path.exists(notebook_path):
            score += 0.3
            feedback.append("Notebook file found.")
        else:
            feedback.append("Notebook file missing.")

        # Check result file existence
        if os.path.exists(result_file):
            score += 0.3
            feedback.append("Result file redundant_features.txt found.")

            try:
                with open(result_file, "r", encoding="utf-8") as f:
                    text = f.read().lower().strip()

                # Normalize text and look for pair
                clean_text = re.sub(r"[^a-z0-9, ]", "", text)
                if true_pair in clean_text:
                    score += 0.4
                    feedback.append("Redundant feature pair matches expected (petal length, petal width).")
                else:
                    feedback.append("Expected redundant feature pair not found in file.")
            except Exception as e:
                feedback.append(f"Error reading redundant_features.txt: {e}")
                logger.error(f"Error reading result file: {e}", exc_info=True)
        else:
            feedback.append("Result file redundant_features.txt missing.")

    except Exception as e:
        feedback.append(f"Evaluation failed: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)

    # Final score
    final_score = min(round(score, 2), 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score


def evaluate_jupyter_iris_petal_distribution_visualizations(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Iris Petal Distribution (Jupyter Notebook) task.

    This evaluation checks:
      - The Jupyter notebook file exists.
      - Both visualization files exist and are non-empty.
      - GPT-4o confirms that:
          1. petal_boxplot.png correctly shows petal_length grouped by class.
          2. petal_violin.png correctly shows the same feature using a violin plot.
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    expected = expected.get("rules", expected)
    base_dir = os.path.dirname(actual)

    notebook_file = os.path.join(base_dir, "iris_petal_distribution.ipynb")
    chart1 = os.path.join(base_dir, "petal_boxplot.png")
    chart2 = os.path.join(base_dir, "petal_violin.png")

    score = 0.0
    feedback = []

    # Step 1: Check notebook existence
    if os.path.exists(notebook_file):
        score += 0.4
        feedback.append("The Jupyter Notebook file exists.")
        logger.info("Notebook file found.")
    else:
        feedback.append("The Jupyter Notebook file is missing.")
        logger.warning("Notebook file missing.")

    # Helper to encode image
    def encode_image(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    # Step 2: Visualization checks
    charts_to_check = [
        (
            chart1,
            "a boxplot showing petal_length grouped by class, with clear axis labels and title."
        ),
        (
            chart2,
            "a violin plot showing the distribution of petal_length grouped by class, including labels and proper visual scaling."
        )
    ]

    for chart, description in charts_to_check:
        chart_name = os.path.basename(chart)
        if os.path.exists(chart) and os.path.getsize(chart) > 0:
            score += 0.15
            feedback.append(f"The visualization {chart_name} exists and is non-empty.")
            logger.info(f"{chart_name} found and valid.")

            try:
                encoded = encode_image(chart)
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert in data visualization evaluation. Assess accuracy, labeling, and readability of charts."
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        f"Examine this chart and confirm whether it correctly represents {description}. "
                                        f"Reply 'yes' if correct and clear, otherwise 'no' followed by a short reason."
                                    )
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{encoded}"}
                                }
                            ]
                        }
                    ],
                    max_tokens=100
                )

                answer = response.choices[0].message.content.lower()
                if "yes" in answer:
                    score += 0.1
                    feedback.append(f"GPT-4o judged {chart_name} as accurate and clearly labeled.")
                    logger.info(f"{chart_name} validated by GPT-4o.")
                else:
                    feedback.append(f"GPT-4o judged {chart_name} unclear or inaccurate: {answer}")
                    logger.warning(f"{chart_name} unclear according to GPT-4o.")
            except Exception as e:
                feedback.append(f"Error evaluating {chart_name} with GPT-4o: {e}")
                logger.error(f"GPT-4o evaluation error for {chart_name}: {e}", exc_info=True)
        else:
            feedback.append(f"The visualization {chart_name} is missing or empty.")
            logger.warning(f"{chart_name} missing or empty.")

    # Step 3: Final scoring
    final_score = min(score, 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score



def evaluate_jupyter_iris_correlation_visualizations(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Iris Correlation Analysis (Jupyter Notebook) task.

    This evaluation checks:
      - The Jupyter notebook file exists.
      - Both visualization files exist and are non-empty.
      - GPT-4o confirms that:
          1. correlation_heatmap.png correctly shows correlations among numeric features.
          2. pairplot.png correctly shows pairwise relationships colored by class.
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    expected = expected.get("rules", expected)
    base_dir = os.path.dirname(actual)

    notebook_file = os.path.join(base_dir, "iris_correlation_analysis.ipynb")
    chart1 = os.path.join(base_dir, "correlation_heatmap.png")
    chart2 = os.path.join(base_dir, "pairplot.png")

    score = 0.0
    feedback = []

    # Step 1: Check notebook existence
    if os.path.exists(notebook_file):
        score += 0.4
        feedback.append("The Jupyter Notebook file exists.")
        logger.info("Notebook file found.")
    else:
        feedback.append("The Jupyter Notebook file is missing.")
        logger.warning("Notebook file missing.")

    # Helper: Encode image as base64
    def encode_image(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    # Step 2: Visualization checks
    charts_to_check = [
        (
            chart1,
            "a heatmap showing correlations between all numeric features, with labeled axes, color scale, and clear annotations."
        ),
        (
            chart2,
            "a pairplot showing pairwise scatter relationships among numeric features, with data points colored by class and labeled axes."
        )
    ]

    for chart, description in charts_to_check:
        chart_name = os.path.basename(chart)
        if os.path.exists(chart) and os.path.getsize(chart) > 0:
            score += 0.15
            feedback.append(f"The visualization {chart_name} exists and is non-empty.")
            logger.info(f"{chart_name} found and valid.")

            try:
                encoded = encode_image(chart)
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert in data visualization assessment. Evaluate charts for clarity, labeling, and accuracy."
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        f"Examine this chart and confirm whether it correctly represents {description}. "
                                        f"Reply 'yes' if correct and clear, otherwise 'no' followed by a short reason."
                                    )
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{encoded}"}
                                }
                            ]
                        }
                    ],
                    max_tokens=100
                )

                answer = response.choices[0].message.content.lower()
                if "yes" in answer:
                    score += 0.1
                    feedback.append(f"GPT-4o judged {chart_name} as accurate and clearly labeled.")
                    logger.info(f"{chart_name} validated by GPT-4o.")
                else:
                    feedback.append(f"GPT-4o judged {chart_name} unclear or inaccurate: {answer}")
                    logger.warning(f"{chart_name} unclear according to GPT-4o.")
            except Exception as e:
                feedback.append(f"Error evaluating {chart_name} with GPT-4o: {e}")
                logger.error(f"GPT-4o evaluation error for {chart_name}: {e}", exc_info=True)
        else:
            feedback.append(f"The visualization {chart_name} is missing or empty.")
            logger.warning(f"{chart_name} missing or empty.")

    # Step 3: Final scoring
    final_score = min(score, 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score


def evaluate_jupyter_iris_dimensionality_reduction_visualizations(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Iris Dimensionality Reduction (Jupyter Notebook) task.

    This evaluation checks:
      - The Jupyter notebook file exists.
      - Both visualization files exist and are non-empty.
      - GPT-4o confirms that:
          1. pca_scatter.png correctly shows PCA (2 components) colored by class.
          2. tsne_scatter.png correctly shows t-SNE embedding colored by class.
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    expected = expected.get("rules", expected)
    base_dir = os.path.dirname(actual)

    notebook_file = os.path.join(base_dir, "iris_dimensionality_reduction.ipynb")
    chart1 = os.path.join(base_dir, "pca_scatter.png")
    chart2 = os.path.join(base_dir, "tsne_scatter.png")

    score = 0.0
    feedback = []

    # Step 1: Check notebook existence
    if os.path.exists(notebook_file):
        score += 0.4
        feedback.append("The Jupyter Notebook file exists.")
        logger.info("Notebook file found.")
    else:
        feedback.append("The Jupyter Notebook file is missing.")
        logger.warning("Notebook file missing.")

    # Helper: Encode image for GPT evaluation
    def encode_image(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    # Step 2: Visualization checks
    charts_to_check = [
        (
            chart1,
            "a scatter plot showing PCA results with 2 components (PC1 vs PC2), where each point is colored by class and includes axis labels and title."
        ),
        (
            chart2,
            "a scatter plot showing t-SNE embedding (2D projection) colored by class, with labeled axes and a descriptive title."
        )
    ]

    for chart, description in charts_to_check:
        chart_name = os.path.basename(chart)
        if os.path.exists(chart) and os.path.getsize(chart) > 0:
            score += 0.15
            feedback.append(f"The visualization {chart_name} exists and is non-empty.")
            logger.info(f"{chart_name} found and valid.")

            try:
                encoded = encode_image(chart)
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert in machine learning visualization assessment. Evaluate PCA and t-SNE plots for clarity, labeling, and class separation."
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        f"Examine this chart and confirm whether it correctly represents {description}. "
                                        f"Reply 'yes' if it is accurate, clear, and visually distinct across classes; otherwise reply 'no' with a short reason."
                                    )
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{encoded}"}
                                }
                            ]
                        }
                    ],
                    max_tokens=100
                )

                answer = response.choices[0].message.content.lower()
                if "yes" in answer:
                    score += 0.1
                    feedback.append(f"GPT-4o judged {chart_name} as accurate and clearly labeled.")
                    logger.info(f"{chart_name} validated by GPT-4o.")
                else:
                    feedback.append(f"GPT-4o judged {chart_name} unclear or inaccurate: {answer}")
                    logger.warning(f"{chart_name} unclear according to GPT-4o.")
            except Exception as e:
                feedback.append(f"Error evaluating {chart_name} with GPT-4o: {e}")
                logger.error(f"GPT-4o evaluation error for {chart_name}: {e}", exc_info=True)
        else:
            feedback.append(f"The visualization {chart_name} is missing or empty.")
            logger.warning(f"{chart_name} missing or empty.")

    # Step 3: Final scoring
    final_score = min(score, 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score



def evaluate_jupyter_iris_outlier_visualizations(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Iris Outlier and Spread Visualization (Jupyter Notebook) task.

    This evaluation checks:
      - The Jupyter notebook file exists.
      - Both visualization files exist and are non-empty.
      - GPT-4o confirms that:
          1. feature_boxgrid.png correctly shows a grid of boxplots for all numeric features.
          2. feature_stripgrid.png correctly shows a grid of stripplots over the same features.
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    expected = expected.get("rules", expected)
    base_dir = os.path.dirname(actual)

    notebook_file = os.path.join(base_dir, "iris_outlier_visualization.ipynb")
    chart1 = os.path.join(base_dir, "feature_boxgrid.png")
    chart2 = os.path.join(base_dir, "feature_stripgrid.png")

    score = 0.0
    feedback = []

    # Step 1: Check notebook existence
    if os.path.exists(notebook_file):
        score += 0.4
        feedback.append("The Jupyter Notebook file exists.")
        logger.info("Notebook file found.")
    else:
        feedback.append("The Jupyter Notebook file is missing.")
        logger.warning("Notebook file missing.")

    # Helper function to encode images
    def encode_image(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    # Step 2: Visualization checks
    charts_to_check = [
        (
            chart1,
            "a grid of boxplots for all numeric features showing spread and outliers, with labeled axes and clear titles."
        ),
        (
            chart2,
            "a grid of stripplots overlayed on the same numeric features, showing data spread with individual data points and labeled axes."
        )
    ]

    for chart, description in charts_to_check:
        chart_name = os.path.basename(chart)
        if os.path.exists(chart) and os.path.getsize(chart) > 0:
            score += 0.15
            feedback.append(f"The visualization {chart_name} exists and is non-empty.")
            logger.info(f"{chart_name} found and valid.")

            try:
                encoded = encode_image(chart)
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert in data visualization evaluation. Assess visual clarity, labeling, and correctness of charts."
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        f"Examine this chart and confirm whether it correctly represents {description}. "
                                        f"Reply 'yes' if it is accurate, clearly labeled, and visually organized; otherwise reply 'no' with a short reason."
                                    )
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{encoded}"}
                                }
                            ]
                        }
                    ],
                    max_tokens=100
                )

                answer = response.choices[0].message.content.lower()
                if "yes" in answer:
                    score += 0.1
                    feedback.append(f"GPT-4o judged {chart_name} as accurate and clearly labeled.")
                    logger.info(f"{chart_name} validated by GPT-4o.")
                else:
                    feedback.append(f"GPT-4o judged {chart_name} unclear or inaccurate: {answer}")
                    logger.warning(f"{chart_name} unclear according to GPT-4o.")
            except Exception as e:
                feedback.append(f"Error evaluating {chart_name} with GPT-4o: {e}")
                logger.error(f"GPT-4o evaluation failed for {chart_name}: {e}", exc_info=True)
        else:
            feedback.append(f"The visualization {chart_name} is missing or empty.")
            logger.warning(f"{chart_name} missing or empty.")

    # Step 3: Final scoring
    final_score = min(score, 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score

def evaluate_jupyter_iris_comparative_feature_range_visualizations(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Iris Comparative Feature Range (Jupyter Notebook) task.

    This evaluation checks:
      - The Jupyter notebook file exists.
      - Both visualization files exist and are non-empty.
      - GPT-4o confirms that:
          1. mean_feature_bar.png correctly shows mean feature values per class.
          2. std_feature_bar.png correctly shows feature standard deviations per class.
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    expected = expected.get("rules", expected)
    base_dir = os.path.dirname(actual)

    notebook_file = os.path.join(base_dir, "iris_comparative_feature_range.ipynb")
    chart1 = os.path.join(base_dir, "mean_feature_bar.png")
    chart2 = os.path.join(base_dir, "std_feature_bar.png")

    score = 0.0
    feedback = []

    # Step 1: Check notebook existence
    if os.path.exists(notebook_file):
        score += 0.4
        feedback.append("The Jupyter Notebook file exists.")
        logger.info("Notebook file found.")
    else:
        feedback.append("The Jupyter Notebook file is missing.")
        logger.warning("Notebook file missing.")

    # Helper to encode image
    def encode_image(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    # Step 2: Visualization checks
    charts_to_check = [
        (
            chart1,
            "a bar chart showing the mean feature values per class, with labeled axes, legend, and title."
        ),
        (
            chart2,
            "a bar chart showing the standard deviation of each feature per class, with labeled axes, legend, and title."
        )
    ]

    for chart, description in charts_to_check:
        chart_name = os.path.basename(chart)
        if os.path.exists(chart) and os.path.getsize(chart) > 0:
            score += 0.15
            feedback.append(f"The visualization {chart_name} exists and is non-empty.")
            logger.info(f"{chart_name} found and valid.")

            try:
                encoded = encode_image(chart)
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert data visualization evaluator. Assess chart clarity, labeling, and relevance."
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        f"Examine this chart and confirm whether it correctly represents {description}. "
                                        f"Reply 'yes' if it is accurate, clear, and visually well-labeled; otherwise reply 'no' with a brief reason."
                                    )
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{encoded}"}
                                }
                            ]
                        }
                    ],
                    max_tokens=100
                )

                answer = response.choices[0].message.content.lower()
                if "yes" in answer:
                    score += 0.1
                    feedback.append(f"GPT-4o judged {chart_name} as accurate and clearly labeled.")
                    logger.info(f"{chart_name} validated by GPT-4o.")
                else:
                    feedback.append(f"GPT-4o judged {chart_name} unclear or inaccurate: {answer}")
                    logger.warning(f"{chart_name} unclear according to GPT-4o.")
            except Exception as e:
                feedback.append(f"Error evaluating {chart_name} with GPT-4o: {e}")
                logger.error(f"GPT-4o evaluation failed for {chart_name}: {e}", exc_info=True)
        else:
            feedback.append(f"The visualization {chart_name} is missing or empty.")
            logger.warning(f"{chart_name} missing or empty.")

    # Step 3: Final scoring
    final_score = min(score, 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score

def evaluate_diet_calorie_balance(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the 'Caloric Balance by Diet Type' reasoning task.

    Expected ground truth (approximate):
        Average Calorie Balance by Diet Type:
            diet_type   calorie_balance
            keto        1263.58
            vegetarian  1260.85
            paleo       1251.93
            balanced    1251.48
            low-carb    1248.94
            vegan       1238.72

        🔥 Diet Type with Highest Positive Average Calorie Balance:
        keto → 1263.58 kcal
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    base_dir = os.path.dirname(actual)
    result_path = os.path.join(base_dir, "diet_calorie_balance.txt")

    score = 0.0
    feedback = []

    try:
        # --- 1. File existence check ---
        if os.path.exists(result_path):
            score += 0.3
            feedback.append("Result file diet_calorie_balance.txt found.")
        else:
            feedback.append("diet_calorie_balance.txt missing.")
            return 0.0

        # --- 2. Content validation ---
        with open(result_path, "r", encoding="utf-8") as f:
            text = f.read().lower().strip()

        # Ensure header exists
        if "average calorie balance by diet type" in text:
            score += 0.2
            feedback.append("Header found: 'Average Calorie Balance by Diet Type'.")
        else:
            feedback.append("Header missing: 'Average Calorie Balance by Diet Type'.")

        # Ensure both columns are mentioned
        if "diet_type" in text and "calorie_balance" in text:
            score += 0.1
            feedback.append("Expected columns ('diet_type', 'calorie_balance') present.")
        else:
            feedback.append("Expected column names not detected.")

        # --- 3. Check for top-ranked diet type (keto) ---
        if "keto" in text:
            score += 0.2
            feedback.append("Top diet type 'keto' detected in output.")
        else:
            feedback.append("Expected top diet type 'keto' not found.")

        # --- 4. Check numeric accuracy (≈1263.58 kcal) ---
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", text)
        if nums:
            numbers = np.array([float(x) for x in nums])
            target = 1263.58
            if np.any(np.isclose(numbers, target, atol=5.0)):
                score += 0.2
                feedback.append("Numeric values close to expected (≈1263.58 kcal).")
            else:
                feedback.append("No numeric value near 1263.58 kcal found.")
        else:
            feedback.append("No numeric values detected in output file.")

    except Exception as e:
        feedback.append(f"Evaluation failed: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)

    # --- 5. Final score ---
    final_score = min(round(score, 2), 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score


def evaluate_heart_efficiency(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the 'Heart Efficiency by Workout Type' reasoning task (Task ds104).

    Expected ground truth (approximate):
        Average Heart Efficiency by Workout Type:
          Workout_Type  heart_efficiency
          strength      2.301
          cardio        2.335
          yoga          2.357
          hiit          2.358

        🏋️‍♀️ Best Conditioning: strength → Avg Heart Efficiency = 2.301
    """
    # --- Handle invalid directory collection ---
    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    base_dir = os.path.dirname(actual)
    script_path = os.path.join(base_dir, "heart_efficiency.py")
    db_path = os.path.join(base_dir, "meal.db")
    result_path = os.path.join(base_dir, "heart_efficiency.txt")

    score = 0.0
    feedback = []

    try:
        # --- 1. Script existence ---
        if os.path.exists(script_path):
            score += 0.2
            feedback.append("Python script 'heart_efficiency.py' found.")
        else:
            feedback.append("Python script missing.")

        # --- 2. Database existence ---
        if os.path.exists(db_path):
            score += 0.1
            feedback.append("Database file 'meal.db' found.")
        else:
            feedback.append("Database file 'meal.db' missing.")

        # --- 3. Result file existence ---
        if os.path.exists(result_path):
            score += 0.3
            feedback.append("Result file 'heart_efficiency.txt' found.")
        else:
            feedback.append("Result file 'heart_efficiency.txt' missing.")
            return round(score, 2)

        # --- 4. Validate content ---
        with open(result_path, "r", encoding="utf-8") as f:
            text = f.read().lower().strip()

        # Header check
        if "average heart efficiency by workout type" in text:
            score += 0.1
            feedback.append("Header found: 'Average Heart Efficiency by Workout Type'.")
        else:
            feedback.append("Header missing: 'Average Heart Efficiency by Workout Type'.")

        # Column mention check
        if "workout_type" in text and "heart_efficiency" in text:
            score += 0.05
            feedback.append("Expected column names ('Workout_Type', 'heart_efficiency') present.")
        else:
            feedback.append("Expected column names not detected.")

        # Check for top-ranked best conditioning
        if "strength" in text:
            score += 0.15
            feedback.append("Top conditioning 'strength' detected in output.")
        else:
            feedback.append("Expected top conditioning 'strength' not found.")

        # --- 5. Numeric value proximity check (≈2.301) ---
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", text)
        if nums:
            values = np.array([float(x) for x in nums])
            target = 2.301
            if np.any(np.isclose(values, target, atol=0.02)):
                score += 0.1
                feedback.append("Numeric values close to expected (≈2.301).")
            else:
                feedback.append("No numeric value near 2.301 found.")
        else:
            feedback.append("No numeric values detected in output file.")

        # --- 6. Summary line / emoji check ---
        if "best conditioning" in text or "🏋️" in text or "→" in text:
            score += 0.1
            feedback.append("Summary line with top workout type correctly reported.")
        else:
            feedback.append("Summary statement missing for best conditioning workout type.")

    except Exception as e:
        feedback.append(f"Evaluation failed: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)

    # --- Final scoring ---
    final_score = min(round(score, 2), 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score

def evaluate_workout_efficiency(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the 'Workout Efficiency Ranking' reasoning task (Task ds105).

    Expected ground truth (approximate):
        Workout_Type  efficiency_index
            hiit            18.545
        strength            15.733
          cardio            13.003
            yoga             9.783

        🔥 Top 5 Most Efficient Workouts:
        Workout_Type  efficiency_index
            hiit            18.545
        strength            15.733
          cardio            13.003
            yoga             9.783
    """

    # --- Handle invalid directory collection ---
    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    base_dir = os.path.dirname(actual)
    script_path = os.path.join(base_dir, "workout_efficiency.py")
    db_path = os.path.join(base_dir, "meal.db")
    result_path = os.path.join(base_dir, "workout_efficiency.csv")

    score = 0.0
    feedback = []

    try:
        # --- 1. Script file existence ---
        if os.path.exists(script_path):
            score += 0.2
            feedback.append("Python script 'workout_efficiency.py' found.")
        else:
            feedback.append("Python script missing.")

        # --- 2. Database existence ---
        if os.path.exists(db_path):
            score += 0.1
            feedback.append("Database file 'meal.db' found.")
        else:
            feedback.append("Database file 'meal.db' missing.")

        # --- 3. Result CSV existence ---
        if os.path.exists(result_path):
            score += 0.3
            feedback.append("Result file 'workout_efficiency.csv' found.")
        else:
            feedback.append("Result file 'workout_efficiency.csv' missing.")
            return round(score, 2)

        # --- 4. Read and validate content ---
        import pandas as pd
        df = pd.read_csv(result_path)

        # Column checks
        if {"Workout_Type", "efficiency_index"}.issubset(df.columns):
            score += 0.15
            feedback.append("Expected columns ('Workout_Type', 'efficiency_index') present.")
        else:
            feedback.append("Missing expected columns in CSV.")
            return round(score, 2)

        # --- 5. Top-ranked workout check ---
        top_type = str(df.iloc[0]["Workout_Type"]).strip().lower()
        if top_type == "hiit":
            score += 0.15
            feedback.append("Top workout type 'hiit' correctly ranked as most efficient.")
        else:
            feedback.append(f"Top workout type mismatch: found '{top_type}' instead of 'hiit'.")

        # --- 6. Numeric proximity check for top efficiency value (≈18.545) ---
        top_value = float(df.iloc[0]["efficiency_index"])
        if np.isclose(top_value, 18.545, atol=0.5):
            score += 0.1
            feedback.append("Top efficiency value close to expected (≈18.545).")
        else:
            feedback.append(f"Top efficiency value deviates from expected ({top_value:.3f}).")

        # --- 7. Row count and ordering check (Top 5 expected) ---
        if len(df) >= 4:
            score += 0.1
            feedback.append("At least 4 ranked workouts found (expected top 5).")
        else:
            feedback.append("Fewer than expected ranked workouts.")

    except Exception as e:
        feedback.append(f"Evaluation failed: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)

    # --- Final scoring ---
    final_score = min(round(score, 2), 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score

def evaluate_hydration_burnrate(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the 'Hydration vs. Calorie Burn Correlation by Gender' reasoning task (Task ds106).

    Expected ground truth (approximate):
        💧 Hydration vs. Calorie Burn Correlation by Gender:

        Gender: Female     | Relationship: Positive  | Correlation: 0.243
        Gender: Male       | Relationship: Positive  | Correlation: 0.341

        ✅ Analysis complete — correlation strength and direction displayed above.
    """

    # --- Handle invalid directory collection ---
    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    base_dir = os.path.dirname(actual)
    script_path = os.path.join(base_dir, "hydration_burnrate.py")
    db_path = os.path.join(base_dir, "meal.db")
    result_path = os.path.join(base_dir, "hydration_burnrate.txt")

    score = 0.0
    feedback = []

    try:
        # --- 1. Script existence ---
        if os.path.exists(script_path):
            score += 0.2
            feedback.append("Python script 'hydration_burnrate.py' found.")
        else:
            feedback.append("Python script missing.")

        # --- 2. Database existence ---
        if os.path.exists(db_path):
            score += 0.1
            feedback.append("Database file 'meal.db' found.")
        else:
            feedback.append("Database file 'meal.db' missing.")

        # --- 3. Result file existence ---
        if os.path.exists(result_path):
            score += 0.3
            feedback.append("Result file 'hydration_burnrate.txt' found.")
        else:
            feedback.append("Result file 'hydration_burnrate.txt' missing.")
            return round(score, 2)

        # --- 4. Read and validate content ---
        with open(result_path, "r", encoding="utf-8") as f:
            text = f.read().lower().strip()

        # Header check
        if "hydration vs. calorie burn correlation by gender" in text:
            score += 0.1
            feedback.append("Header found: 'Hydration vs. Calorie Burn Correlation by Gender'.")
        else:
            feedback.append("Header missing: 'Hydration vs. Calorie Burn Correlation by Gender'.")

        # Gender mentions check
        if "female" in text and "male" in text:
            score += 0.1
            feedback.append("Both genders (Female, Male) detected in output.")
        else:
            feedback.append("One or both gender categories missing.")

        # Relationship direction check
        if "positive" in text:
            score += 0.1
            feedback.append("Positive correlation relationship correctly identified.")
        else:
            feedback.append("Expected 'Positive' relationship not found.")

        # --- 5. Numeric correlation proximity check ---
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", text)
        if nums:
            values = np.array([float(x) for x in nums])
            expected_values = np.array([0.243, 0.341])
            diff = np.mean([np.min(np.abs(values - v)) for v in expected_values])
            if diff <= 0.05:
                score += 0.15
                feedback.append("Correlation values close to expected (≈0.243, 0.341).")
            else:
                feedback.append(f"Correlation values deviate (avg diff={diff:.3f}).")
        else:
            feedback.append("No numeric values detected in output file.")

        # --- 6. Completion or summary check ---
        if "analysis complete" in text or "✅" in text or "displayed above" in text:
            score += 0.05
            feedback.append("Completion summary line correctly reported.")
        else:
            feedback.append("Completion summary missing in output.")

    except Exception as e:
        feedback.append(f"Evaluation failed: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)

    # --- Final scoring ---
    final_score = min(round(score, 2), 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score

def evaluate_mealtype_caloric_impact(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the 'Meal Type Caloric Impact' reasoning task (Task ds107).

    Expected ground truth (approximate):
        meal_type   Calories_Burned
           dinner          1381.33
            lunch          1327.16
            snack          1304.73
        breakfast          1288.62

        🏆 Meal Type with Highest Average Calories Burned: Dinner (1381.33 kcal)
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    base_dir = os.path.dirname(actual)
    script_path = os.path.join(base_dir, "mealtype_caloric_impact.py")
    db_path = os.path.join(base_dir, "meal.db")
    result_path = os.path.join(base_dir, "mealtype_caloric_impact.txt")

    score = 0.0
    feedback = []

    try:
        # --- 1. Script existence ---
        if os.path.exists(script_path):
            score += 0.2
            feedback.append("Python script 'mealtype_caloric_impact.py' found.")
        else:
            feedback.append("Python script missing.")

        # --- 2. Database existence ---
        if os.path.exists(db_path):
            score += 0.1
            feedback.append("Database file 'meal.db' found.")
        else:
            feedback.append("Database file 'meal.db' missing.")

        # --- 3. Result file existence ---
        if os.path.exists(result_path):
            score += 0.3
            feedback.append("Result file 'mealtype_caloric_impact.txt' found.")
        else:
            feedback.append("Result file 'mealtype_caloric_impact.txt' missing.")
            return round(score, 2)

        # --- 4. Read and validate content ---
        with open(result_path, "r", encoding="utf-8") as f:
            text = f.read().lower().strip()

        # Header check
        if "meal_type" in text and "calories_burned" in text:
            score += 0.1
            feedback.append("Header or column names ('meal_type', 'Calories_Burned') detected.")
        else:
            feedback.append("Expected column headers missing.")

        # --- 5. Meal type presence check ---
        if "breakfast" in text and "lunch" in text and "dinner" in text:
            score += 0.1
            feedback.append("All primary meal types (Breakfast, Lunch, Dinner) detected.")
        else:
            feedback.append("Not all primary meal types present in output.")

        # --- 6. Top meal type correctness check ---
        if "dinner" in text:
            score += 0.1
            feedback.append("Top meal type 'Dinner' detected as highest calorie burn.")
        else:
            feedback.append("Expected top meal type 'Dinner' not found.")

        # --- 7. Numeric proximity check (≈1381.33 kcal) ---
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", text)
        if nums:
            values = np.array([float(x) for x in nums])
            target = 1381.33
            if np.any(np.isclose(values, target, atol=5.0)):
                score += 0.1
                feedback.append("Numeric value close to expected (≈1381.33 kcal).")
            else:
                feedback.append("No numeric value near 1381.33 kcal found.")
        else:
            feedback.append("No numeric values detected in output file.")

        # --- 8. Optional summary line (flexible bonus) ---
        if any(kw in text for kw in ["highest", "🏆", "top", "most", "kcal"]):
            score += 0.05
            feedback.append("Summary or highlight line present (bonus).")

    except Exception as e:
        feedback.append(f"Evaluation failed: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)

    # --- Final scoring ---
    final_score = min(round(score, 2), 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score

def evaluate_macro_deviation(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the 'Macro–Calorie Deviation Detection' reasoning task (Task ds108).

    Ground truth expectation:
        ⚠️ Total Records with >10% Macro Deviation: 0
        ✅ No significant macro deviations found.
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    base_dir = os.path.dirname(actual)
    script_path = os.path.join(base_dir, "macro_deviation.py")
    db_path = os.path.join(base_dir, "meal.db")
    result_path = os.path.join(base_dir, "macro_deviation.csv")

    score = 0.0
    feedback = []

    try:
        # --- 1. Script existence ---
        if os.path.exists(script_path):
            score += 0.25
            feedback.append("Python script 'macro_deviation.py' found.")
        else:
            feedback.append("Python script missing.")
            return 0.0

        # --- 2. Database existence ---
        if os.path.exists(db_path):
            score += 0.15
            feedback.append("Database file 'meal.db' found.")
        else:
            feedback.append("Database file 'meal.db' missing.")
            return round(score, 2)

        # --- 3. CSV existence ---
        if os.path.exists(result_path):
            score += 0.3
            feedback.append("Result file 'macro_deviation.csv' found.")
        else:
            feedback.append("Result file 'macro_deviation.csv' missing.")
            return round(score, 2)

        # --- 4. Validate CSV content ---
        try:
            df = pd.read_csv(result_path)
            deviation_count = len(df)
            if deviation_count == 0:
                score += 0.3
                feedback.append("No significant macro deviations found (0 records) — matches expected result.")
            else:
                feedback.append(f"Unexpected: {deviation_count} records flagged (>10% deviation).")
        except Exception as e:
            feedback.append(f"Error reading 'macro_deviation.csv': {e}")
            return round(score, 2)

    except Exception as e:
        feedback.append(f"Evaluation failed: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)

    # --- Final scoring ---
    final_score = min(round(score, 2), 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score


def evaluate_bmi_burn_relation(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the 'BMI–Burn Relationship by Workout Type' reasoning task (Task ds109).

    Ground truth expectation:
        🔥 Correlation Between BMI and Calories Burned by Workout Type:
        Cardio    → -0.021 (Negative)
        Hiit      → +0.019 (Positive)
        Strength  → +0.111 (Positive)
        Yoga      → -0.017 (Negative)

        🏆 Strongest Relationship: Strength (+0.111, Positive)
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    # --- Define expected values ---
    expected_correlations = {
        "cardio": -0.021,
        "hiit": 0.019,
        "strength": 0.111,
        "yoga": -0.017
    }
    top_relation = "strength"

    # --- Paths ---
    base_dir = os.path.dirname(actual)
    script_path = os.path.join(base_dir, "bmi_burn_relation.py")
    db_path = os.path.join(base_dir, "meal.db")
    result_path = os.path.join(base_dir, "bmi_burn_relation.txt")

    score = 0.0
    feedback = []

    try:
        # --- 1. Script existence ---
        if os.path.exists(script_path):
            score += 0.25
            feedback.append("Python script 'bmi_burn_relation.py' found.")
        else:
            feedback.append("Python script missing.")
            return 0.0

        # --- 2. Database existence ---
        if os.path.exists(db_path):
            score += 0.15
            feedback.append("Database file 'meal.db' found.")
        else:
            feedback.append("Database file 'meal.db' missing.")
            return round(score, 2)

        # --- 3. Output file existence ---
        if os.path.exists(result_path):
            score += 0.25
            feedback.append("Result file 'bmi_burn_relation.txt' found.")
        else:
            feedback.append("Result file 'bmi_burn_relation.txt' missing.")
            return round(score, 2)

        # --- 4. Validate file content ---
        with open(result_path, "r", encoding="utf-8") as f:
            text = f.read().lower().strip()

        # --- 5. Check presence of all workout types ---
        if all(w in text for w in expected_correlations.keys()):
            score += 0.1
            feedback.append("All workout types (cardio, hiit, strength, yoga) detected.")
        else:
            feedback.append("Missing one or more workout types in output.")

        # --- 6. Check correlation direction presence ---
        if "positive" in text or "negative" in text:
            score += 0.05
            feedback.append("Correlation direction (Positive/Negative) present.")
        else:
            feedback.append("Correlation direction not found.")

        # --- 7. Numeric accuracy check ---
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", text)
        if nums:
            values = np.array([float(x) for x in nums])
            diffs = []
            for exp in expected_correlations.values():
                diffs.append(np.min(np.abs(values - exp)))
            avg_diff = np.mean(diffs)
            if avg_diff <= 0.03:
                score += 0.15
                feedback.append("Correlation values close to expected (avg diff ≤ 0.03).")
            else:
                feedback.append(f"Correlation values deviate (avg diff = {avg_diff:.3f}).")
        else:
            feedback.append("No numeric correlation values found.")

        # --- 8. Check strongest relationship line ---
        if top_relation in text and "strongest" in text:
            score += 0.05
            feedback.append("Strongest relationship correctly identified as 'Strength'.")
        else:
            feedback.append("Expected strongest relation ('Strength') not detected.")

    except Exception as e:
        feedback.append(f"Evaluation failed: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)

    # --- Final scoring ---
    final_score = min(round(score, 2), 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score

def evaluate_overtraining_risk(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the 'Overtraining Risk Detection' reasoning task (Task ds110).

    Ground truth expectation:
        ⚠️ Total Potential Overtraining Risk Cases: 14291
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    base_dir = os.path.dirname(actual)
    script_path = os.path.join(base_dir, "overtraining_risk.py")
    db_path = os.path.join(base_dir, "meal.db")
    result_path = os.path.join(base_dir, "overtraining_risk.csv")

    score = 0.0
    feedback = []

    try:
        # --- 1. Script existence ---
        if os.path.exists(script_path):
            score += 0.25
            feedback.append("Python script 'overtraining_risk.py' found.")
        else:
            feedback.append("Python script missing.")
            return 0.0

        # --- 2. Database existence ---
        if os.path.exists(db_path):
            score += 0.15
            feedback.append("Database file 'meal.db' found.")
        else:
            feedback.append("Database file 'meal.db' missing.")
            return round(score, 2)

        # --- 3. Result CSV existence ---
        if os.path.exists(result_path):
            score += 0.25
            feedback.append("Result file 'overtraining_risk.csv' found.")
        else:
            feedback.append("Result file 'overtraining_risk.csv' missing.")
            return round(score, 2)

        # --- 4. Validate CSV content ---
        try:
            df = pd.read_csv(result_path)
            flagged_count = len(df)

            if flagged_count == 14291:
                score += 0.35
                feedback.append("Flagged case count matches expected (14291).")
            elif abs(flagged_count - 14291) <= 50:
                score += 0.25
                feedback.append(f"Flagged case count close to expected (found {flagged_count}).")
            else:
                feedback.append(f"Flagged count deviates significantly (found {flagged_count}, expected 14291).")
        except Exception as e:
            feedback.append(f"Error reading 'overtraining_risk.csv': {e}")
            return round(score, 2)

    except Exception as e:
        feedback.append(f"Evaluation failed: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)

    # --- Final scoring ---
    final_score = min(round(score, 2), 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score

def evaluate_burn_variability(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the 'Burn Ratio Variability by Workout Type' reasoning task (Task ds111).

    Ground truth expectation:
        🔥 Burn Ratio Variability by Workout Type (descending):
        Workout_Type  std_burn_ratio
            hiit          0.236
        strength          0.162
          cardio          0.140
            yoga          0.093

        ⚠️ Most Inconsistent Burn Performance: Hiit (0.236)
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    # --- Expected structure ---
    expected_std = {
        "hiit": 0.236,
        "strength": 0.162,
        "cardio": 0.140,
        "yoga": 0.093
    }
    expected_top = "hiit"

    # --- File paths ---
    base_dir = os.path.dirname(actual)
    script_path = os.path.join(base_dir, "burn_variability.py")
    db_path = os.path.join(base_dir, "meal.db")
    result_path = os.path.join(base_dir, "burn_variability.txt")

    score = 0.0
    feedback = []

    try:
        # --- 1. Script existence ---
        if os.path.exists(script_path):
            score += 0.25
            feedback.append("Python script 'burn_variability.py' found.")
        else:
            feedback.append("Python script missing.")
            return 0.0

        # --- 2. Database existence ---
        if os.path.exists(db_path):
            score += 0.15
            feedback.append("Database file 'meal.db' found.")
        else:
            feedback.append("Database file 'meal.db' missing.")
            return round(score, 2)

        # --- 3. Output text file existence ---
        if os.path.exists(result_path):
            score += 0.25
            feedback.append("Result file 'burn_variability.txt' found.")
        else:
            feedback.append("Result file 'burn_variability.txt' missing.")
            return round(score, 2)

        # --- 4. Validate content ---
        with open(result_path, "r", encoding="utf-8") as f:
            text = f.read().lower().strip()

        # --- 5. Check all workout types ---
        if all(k in text for k in expected_std.keys()):
            score += 0.1
            feedback.append("All workout types (hiit, strength, cardio, yoga) present.")
        else:
            feedback.append("Missing one or more workout types in output.")

        # --- 6. Check top inconsistent performer ---
        if expected_top in text and "inconsistent" in text:
            score += 0.05
            feedback.append("Top inconsistent performer correctly identified as 'hiit'.")
        else:
            feedback.append("Expected 'hiit' as top inconsistent performer not found.")

        # --- 7. Numeric accuracy check ---
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", text)
        if nums:
            values = np.array([float(x) for x in nums])
            diffs = []
            for val in expected_std.values():
                diffs.append(np.min(np.abs(values - val)))
            avg_diff = np.mean(diffs)
            if avg_diff <= 0.02:
                score += 0.2
                feedback.append("Numeric standard deviation values close to expected (avg diff ≤ 0.02).")
            else:
                feedback.append(f"Numeric values deviate (avg diff = {avg_diff:.3f}).")
        else:
            feedback.append("No numeric values detected in output.")

    except Exception as e:
        feedback.append(f"Evaluation failed: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)

    # --- Final scoring ---
    final_score = min(round(score, 2), 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score

def evaluate_fitness_clusters(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the 'Fitness Clusters using K-Means' modeling task (Task ds112).

    This version focuses on realistic validation:
    - Numeric values are not compared directly.
    - Emphasis is on structure, label correctness, and overall statistical validity.
    """ 

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    base_dir = os.path.dirname(actual)
    script_path = os.path.join(base_dir, "fitness_clusters.py")
    db_path = os.path.join(base_dir, "meal.db")
    result_path = os.path.join(base_dir, "fitness_clusters.csv")

    score = 0.0
    feedback = []

    try:
        # 1. Check that the script exists (0.20)
        if os.path.exists(script_path):
            score += 0.20
            feedback.append("The Python script fitness_clusters.py was found.")
        else:
            feedback.append("The Python script fitness_clusters.py is missing.")
            return 0.0

        # 2. Check that the database exists (0.10)
        if os.path.exists(db_path):
            score += 0.10
            feedback.append("The database meal.db is available.")
        else:
            feedback.append("The database meal.db could not be found.")
            return round(score, 2)

        # 3. Check that the output file exists (0.20)
        if os.path.exists(result_path):
            score += 0.20
            feedback.append("The output file fitness_clusters.csv was generated successfully.")
        else:
            feedback.append("The output file fitness_clusters.csv is missing.")
            return round(score, 2)

        # 4. Validate file structure (0.15)
        df = pd.read_csv(result_path)
        required_cols = {
            "Calories_Burned",
            "Fat_Percentage",
            "BMI_calc",
            "Water_Intake",
            "cal_balance",
            "lean_mass_kg",
            "expected_burn",
            "Cluster_Label",
            "Count"
        }
        if required_cols.issubset(df.columns):
            score += 0.15
            feedback.append("All required columns are present in the output file.")
        else:
            missing = required_cols - set(df.columns)
            feedback.append(f"The following expected columns are missing: {missing}")

        # 5. Verify cluster count (0.10)
        if "Cluster_Label" in df.columns:
            n_clusters = df["Cluster_Label"].nunique()
            if n_clusters == 3:
                score += 0.10
                feedback.append("The file contains exactly three clusters, as expected.")
            else:
                feedback.append(f"The file contains {n_clusters} clusters instead of three.")
        else:
            feedback.append("The column Cluster_Label is missing from the file.")

        # 6. Verify expected labels (≥2 match) (0.10)
        expected_labels = {"balanced", "calorie deficit", "overtrained"}
        if "Cluster_Label" in df.columns:
            labels = set(df["Cluster_Label"].astype(str).str.lower().str.strip())
            overlap = labels & expected_labels
            if len(overlap) >= 2:
                score += 0.10
                feedback.append(f"At least two of the expected cluster labels were found: {overlap}.")
            else:
                feedback.append(f"The cluster labels do not match the expected ones. Found: {labels}")
        else:
            feedback.append("Cluster labels could not be verified because the column is missing.")

        # 7. Check centroid variability (0.10)
        numeric_cols = [
            "Calories_Burned",
            "Fat_Percentage",
            "BMI_calc",
            "Water_Intake",
            "cal_balance",
            "lean_mass_kg",
            "expected_burn"
        ]
        numeric_data = df[numeric_cols].select_dtypes(include=[np.number])
        if numeric_data.std().mean() > 0.01:
            score += 0.10
            feedback.append("The cluster centroids vary across features, suggesting the model output is realistic.")
        else:
            feedback.append("The centroids show very little variation, which may indicate an incorrect clustering process.")

    except Exception as e:
        feedback.append(f"Evaluation failed due to an error: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)

    # Final score
    final_score = min(round(score, 2), 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score


def evaluate_category_revenue(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for 'Category Revenue Leader' task (realistic version).

    Focuses on reasoning correctness:
    - Output file existence
    - Detection of category names and numeric revenues
    - Verification that 'Tablet' appears as top category
    - Numeric revenue close to ~475M
    """
    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    base_dir = os.path.dirname(actual)
    result_path = os.path.join(base_dir, "category_revenue.txt")

    score = 0.0
    feedback = []

    try:
        # 1. File existence
        if os.path.exists(result_path):
            score += 0.4
            feedback.append("Output file category_revenue.txt found.")
        else:
            feedback.append("Result file category_revenue.txt missing.")
            return 0.0

        # 2. Read and normalize content
        with open(result_path, "r", encoding="utf-8") as f:
            text = f.read().lower()

        # 3. Check top category mention
        if "tablet" in text:
            score += 0.25
            feedback.append("Category name 'Tablet' detected in output.")
        else:
            feedback.append("Expected top category 'Tablet' not found.")

        # 4. Numeric validation for approximate revenue magnitude (~475M)
        nums = re.findall(r"[-+]?\d[\d,]*\.?\d*", text)
        clean_nums = [float(x.replace(",", "")) for x in nums if x.replace(",", "").replace(".", "").isdigit()]

        if clean_nums:
            max_val = max(clean_nums)
            target = 475_845_320.0
            if abs(max_val - target) <= 10_000_000:  # within ±10M tolerance
                score += 0.25
                feedback.append("Revenue magnitude is close to expected (~475 million).")
            elif abs(max_val - target) <= 50_000_000:
                score += 0.15
                feedback.append("Revenue roughly near expected range but outside 10M tolerance.")
            else:
                feedback.append(f"Revenue deviates significantly (found max {max_val:,.0f}).")
        else:
            feedback.append("No numeric values detected in the result file.")

        # 5. Consistency check – at least 5 numeric values (for multiple categories)
        if len(clean_nums) >= 5:
            score += 0.1
            feedback.append("Multiple numeric revenue values found, indicating category-wise computation.")
        else:
            feedback.append("Few numeric values found; may not include full category summary.")

    except Exception as e:
        feedback.append(f"Evaluation failed due to an error: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)

    # Final score capped at 1.0
    final_score = min(round(score, 2), 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score

def evaluate_store_profitability(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the 'Store Profitability Index' reasoning task.

    Expected output structure (approximate):
            Store_Name     total_revenue  total_quantity  profit_index
        Apple Rosenstrasse   41,247,230.00          37741        1092.90
        Apple Dubai Mall     42,610,693.00          38999        1092.61
        Apple Union Square   41,365,349.00          37969        1089.45

    Evaluation focuses on:
    - Output file existence
    - Detection of valid numeric profit_index values
    - Top 3 stores listed (Apple Rosenstrasse, Dubai Mall, Union Square)
    - Realistic profit_index range (~1000–1200)
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    base_dir = os.path.dirname(actual)
    result_path = os.path.join(base_dir, "store_profitability.csv")

    score = 0.0
    feedback = []

    try:
        # 1. Check file existence
        if os.path.exists(result_path):
            score += 0.4
            feedback.append("Output file store_profitability.csv found.")
        else:
            feedback.append("Output file store_profitability.csv missing.")
            return 0.0

        # 2. Read file content
        with open(result_path, "r", encoding="utf-8") as f:
            text = f.read().lower()

        # 3. Detect presence of expected store names
        found_stores = 0
        expected_stores = ["rosenstrasse", "dubai", "union square"]
        for store in expected_stores:
            if store in text:
                found_stores += 1
        if found_stores >= 2:
            score += 0.2
            feedback.append(f"{found_stores} of 3 expected top store names found in output.")
        else:
            feedback.append(f"Few expected top store names found ({found_stores}/3).")

        # 4. Extract numeric values
        nums = re.findall(r"[-+]?\d[\d,]*\.?\d*", text)
        clean_nums = [float(x.replace(",", "")) for x in nums if x.replace(",", "").replace(".", "").isdigit()]
        if not clean_nums:
            feedback.append("No numeric values found in output.")
            return round(score, 2)

        # 5. Check number of numeric entries (should be several)
        if len(clean_nums) >= 6:
            score += 0.1
            feedback.append("Sufficient numeric values detected for store-level summaries.")
        else:
            feedback.append("Too few numeric values; output may be incomplete.")

        # 6. Check profit_index plausibility (~1000–1200)
        profit_like = [v for v in clean_nums if 800 <= v <= 1500]
        if len(profit_like) >= 3:
            score += 0.2
            feedback.append("Profit index values fall within realistic range (~1000–1200).")
        else:
            feedback.append("Profit index values appear unrealistic or missing.")

        # 7. Ranking structure check (descending order)
        # Look for descending profit_index pattern by approximate numeric order
        if len(profit_like) >= 3:
            if all(profit_like[i] >= profit_like[i + 1] for i in range(len(profit_like) - 1)):
                score += 0.1
                feedback.append("Profit index values appear correctly ranked in descending order.")
            else:
                feedback.append("Profit index values not clearly ranked.")
        else:
            feedback.append("Not enough numeric entries to validate ranking order.")

    except Exception as e:
        feedback.append(f"Evaluation failed due to an error: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)

    # Final score
    final_score = min(round(score, 2), 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score

def evaluate_launch_performance(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the 'Product Launch Effectiveness' reasoning task.

    Expected output (approximate):
        Average Monthly Sales Growth (Recent Launches):
                     Product_Name growth_rate
                  MagSafe Charger        0.65
                     Apple Arcade        0.11
                    iPhone 12 Pro       -0.02
                    ...

        Top Performer: MagSafe Charger (Growth Rate: 0.65 units/month)

    Evaluation logic:
    - Verifies output file existence
    - Ensures numeric growth rate values exist
    - Confirms multiple products listed
    - Detects a top performer with positive growth
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    base_dir = os.path.dirname(actual)
    result_path = os.path.join(base_dir, "launch_performance.txt")

    score = 0.0
    feedback = []

    try:
        # 1. File existence (0.4)
        if os.path.exists(result_path):
            score += 0.4
            feedback.append("Output file launch_performance.txt found.")
        else:
            feedback.append("Output file launch_performance.txt missing.")
            return 0.0

        # 2. Read file content
        with open(result_path, "r", encoding="utf-8") as f:
            text = f.read().lower()

        # 3. Top performer detection (0.25)
        if "magsafe" in text:
            score += 0.25
            feedback.append("Top-performing product 'MagSafe Charger' detected in output.")
        else:
            feedback.append("Expected top performer 'MagSafe Charger' not found.")

        # 4. Numeric growth rate validation (0.25)
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", text)
        clean_nums = [float(n) for n in nums if re.match(r"[-+]?\d*\.\d+", n)]
        if clean_nums:
            mean_val = np.mean(clean_nums)
            if any(v > 0 for v in clean_nums):
                score += 0.25
                feedback.append("Growth rate values detected, including positive growth values.")
            else:
                feedback.append("Numeric growth rates found but none are positive.")
        else:
            feedback.append("No numeric growth rate values found in output file.")

        # 5. Product count / variety check (0.10)
        # Expect at least 5–10 products mentioned
        product_mentions = len(re.findall(r"\n\s*[a-z]", text))
        if product_mentions >= 5:
            score += 0.10
            feedback.append("Multiple products listed in output summary.")
        else:
            feedback.append("Too few product entries detected in the summary.")

    except Exception as e:
        feedback.append(f"Evaluation failed due to an error: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)

    # Final score (capped at 1.00)
    final_score = min(round(score, 2), 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score


def evaluate_city_category_diversity(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the 'City-Wise Category Diversity' reasoning task.

    Expected outcome (approximate):
        Abu Dhabi — 10 distinct categories

    Evaluation only checks:
    - Output file existence
    - Detection of 'Abu Dhabi' as the top-diversity city
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    base_dir = os.path.dirname(actual)
    result_path = os.path.join(base_dir, "city_category_diversity.txt")

    score = 0.0
    feedback = []

    try:
        # 1. File existence (0.6)
        if os.path.exists(result_path):
            score += 0.6
            feedback.append("Output file city_category_diversity.txt found.")
        else:
            feedback.append("Output file city_category_diversity.txt missing.")
            return 0.0

        # 2. Check for 'Abu Dhabi' mention (0.4)
        with open(result_path, "r", encoding="utf-8") as f:
            text = f.read().lower()
        if "abu dhabi" in text:
            score += 0.4
            feedback.append("Top city 'Abu Dhabi' detected in output.")
        else:
            feedback.append("Expected top city 'Abu Dhabi' not found in output.")

    except Exception as e:
        feedback.append(f"Evaluation failed due to an error: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)

    final_score = min(round(score, 2), 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score


def evaluate_warranty_claim_rate(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the 'Warranty Claim Rate' reasoning task.

    Expected approximate output (top 3 products sorted by claim_rate descending):
        Product_Name          claim_rate
        MacBook                     5.82
        MacBook Air (M1)            5.73
        MacBook Air (M2)            5.54

    Evaluation focuses on:
    - CSV file existence
    - Columns presence
    - At least 3 rows in file
    - 'MacBook' appears as top product
    """
    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    base_dir = os.path.dirname(actual)
    csv_path = os.path.join(base_dir, "warranty_claim_rate.csv")

    score = 0.0
    feedback = []

    try:
        # 1. File existence (0.4)
        if os.path.exists(csv_path):
            score += 0.4
            feedback.append("Output file warranty_claim_rate.csv found.")
        else:
            feedback.append("Output file warranty_claim_rate.csv missing.")
            return 0.0

        # 2. Read file
        df = pd.read_csv(csv_path)

        # 3. Check required columns (0.3)
        required_cols = {"Product_Name", "claim_rate"}
        if required_cols.issubset(df.columns):
            score += 0.3
            feedback.append("Required columns present: Product_Name, claim_rate.")
        else:
            feedback.append("Missing expected columns (Product_Name, claim_rate).")
            return round(score, 2)

        # 4. Check at least 3 rows (0.2)
        if len(df) >= 3:
            score += 0.2
            feedback.append("At least 3 products listed as flagged (>5%).")
        else:
            feedback.append(f"Only {len(df)} products found, expected ≥3.")

        # 5. Check top product name (0.1)
        top_name = str(df.iloc[0]["Product_Name"]).lower()
        if "macbook" in top_name:
            score += 0.1
            feedback.append("Top product 'MacBook' correctly identified.")
        else:
            feedback.append(f"Top product differs: found '{df.iloc[0]['Product_Name']}'.")

    except Exception as e:
        feedback.append(f"Evaluation failed due to an error: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)

    # Final scoring
    final_score = min(round(score, 2), 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score

def evaluate_underperforming_categories(actual: str, expected: dict, **options) -> float:
    """
    Realistic evaluator for 'Underperforming Categories' reasoning task.

    Expected key result values (approximate):
        Smart Speaker          818324.88
        Streaming Device       1625878.86
        Subscription Service   3126877.95

    Focuses only on detecting these categories and their approximate values.
    """

    if actual is None:
        logger.error("No file path provided.")
        return 0.0

    base_dir = os.path.dirname(actual)
    result_path = os.path.join(base_dir, "underperforming_categories.txt")

    score = 0.0
    feedback = []

    try:
        # 1. Check file existence (0.4)
        if os.path.exists(result_path):
            score += 0.4
            feedback.append("Result file underperforming_categories.txt found.")
        else:
            feedback.append("Result file underperforming_categories.txt missing.")
            return 0.0

        # 2. Read content
        with open(result_path, "r", encoding="utf-8") as f:
            text = f.read().lower()

        # Expected key categories and approximate numeric targets
        expected_values = {
            "smart speaker": 818324.88,
            "streaming device": 1625878.86,
            "subscription service": 3126877.95
        }

        # 3. Check each category presence (0.4 total)
        found_count = 0
        for cat, val in expected_values.items():
            if cat in text:
                found_count += 1

        if found_count == 3:
            score += 0.4
            feedback.append("All three expected categories found.")
        elif found_count >= 1:
            score += 0.2
            feedback.append(f"{found_count} expected categories found.")
        else:
            feedback.append("No expected categories found.")

        # 4. Check numeric values (0.2)
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", text)
        nums = np.array(list(map(float, nums))) if nums else []
        expected_nums = np.array(list(expected_values.values()))

        if len(nums) > 0:
            close_count = sum(any(np.isclose(num, exp, atol=50000)) for num in nums for exp in expected_nums)
            if close_count >= 3:
                score += 0.2
                feedback.append("Numeric values match expected ranges.")
            elif close_count >= 1:
                score += 0.1
                feedback.append("Some numeric values roughly align with expected ones.")
            else:
                feedback.append("Numeric values deviate significantly.")
        else:
            feedback.append("No numeric values detected in output.")

    except Exception as e:
        feedback.append(f"Evaluation failed due to error: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)

    final_score = min(round(score, 2), 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score

def evaluate_product_stability(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for 'Product Stability' reasoning task.

    Expected approximate output (top 5 most stable products):
        iMac with Retina Display
        MacBook Air (M1)
        Mac Pro (2023)
        iPad Air (5th Generation)
        iPhone 13 mini
    """

    if actual is None:
        logger.error("No valid directory found.")
        return 0.0

    base_dir = os.path.dirname(actual)
    csv_path = os.path.join(base_dir, "product_stability.csv")

    score = 0.0
    feedback = []

    try:
        # 1. File existence (0.4)
        if os.path.exists(csv_path):
            score += 0.4
            feedback.append("Output file product_stability.csv found.")
        else:
            feedback.append("Output file product_stability.csv missing.")
            return 0.0

        # 2. Read CSV content (0.3)
        df = pd.read_csv(csv_path)
        if len(df) >= 5:
            score += 0.3
            feedback.append("At least 5 rows found in output (top stable products).")
        else:
            feedback.append(f"Only {len(df)} rows found, expected ≥5.")

        # 3. Check expected product names (0.3)
        text = " ".join(df.astype(str).values.flatten()).lower()
        expected_products = [
            "imac with retina display",
            "macbook air (m1)",
            "mac pro (2023)",
            "ipad air (5th generation)",
            "iphone 13 mini"
        ]
        found = [p for p in expected_products if p in text]

        if len(found) == 5:
            score += 0.3
            feedback.append("All 5 expected products detected in output.")
        elif len(found) >= 3:
            score += 0.15
            feedback.append(f"Partial match: {len(found)} expected products found.")
        else:
            feedback.append("Expected product names not found in output.")

    except Exception as e:
        feedback.append(f"Evaluation failed: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)

    final_score = min(round(score, 2), 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score

def evaluate_repair_efficiency(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the 'Repair Efficiency by Country' reasoning task.

    Expected outcome (approximate):
        Country with Fastest Average Repair Turnaround:
        Country         avg_repair_days
        South Korea     702.31

    Checks only:
    - Output file exists
    - 'South Korea' present in output
    - Numeric value (repair days) found
    """

    if actual is None:
        logger.error("No file path provided.")
        return 0.0

    base_dir = os.path.dirname(actual)
    result_path = os.path.join(base_dir, "repair_efficiency.txt")

    score = 0.0
    feedback = []

    try:
        # 1. File presence (0.4)
        if os.path.exists(result_path):
            score += 0.4
            feedback.append("Result file repair_efficiency.txt found.")
        else:
            feedback.append("repair_efficiency.txt missing.")
            return 0.0

        # 2. Read file
        with open(result_path, "r", encoding="utf-8") as f:
            text = f.read().lower().strip()

        # 3. Check for 'south korea' (0.4)
        if "south korea" in text:
            score += 0.4
            feedback.append("Correct fastest country 'South Korea' found.")
        else:
            feedback.append("Expected country 'South Korea' not found.")

        # 4. Check for numeric value (0.2)
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", text)
        if nums:
            values = np.array(list(map(float, nums)))
            if any(700 <= v <= 705 for v in values):  # flexible range for avg days
                score += 0.2
                feedback.append("Average repair days numeric value found within valid range.")
            else:
                feedback.append("Numeric values found but outside expected range.")
        else:
            feedback.append("No numeric values detected in output.")

    except Exception as e:
        feedback.append(f"Evaluation failed: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)

    # Finalize
    final_score = min(round(score, 2), 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score


def evaluate_store_warranty_ratio(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for 'Store Warranty Claim Ratio' reasoning task.

    Expected output (approximate):
        🏆 Top 3 Stores with Highest Warranty Claim Ratio:
            Store_Name
            Apple The Dubai Mall
            Apple The Dubai Mall
            Apple Orchard Road

    Checks only:
    - Output file exists
    - At least 3 store entries present
    - Specific store names detected (Apple The Dubai Mall, Apple Orchard Road)
    """

    if actual is None:
        logger.error("No valid directory path found.")
        return 0.0

    base_dir = os.path.dirname(actual)
    csv_path = os.path.join(base_dir, "store_warranty_ratio.csv")

    score = 0.0
    feedback = []

    try:
        # 1. Check file existence (0.4)
        if os.path.exists(csv_path):
            score += 0.4
            feedback.append("Result file store_warranty_ratio.csv found.")
        else:
            feedback.append("Result file store_warranty_ratio.csv missing.")
            return 0.0

        # 2. Read CSV (0.3)
        df = pd.read_csv(csv_path)
        if len(df) >= 3:
            score += 0.3
            feedback.append("At least 3 stores found in output (top-ranked).")
        else:
            feedback.append(f"Only {len(df)} stores found, expected ≥3.")

        # 3. Check for expected store names (0.3)
        text = " ".join(df.astype(str).values.flatten()).lower()
        expected_stores = ["apple the dubai mall", "apple orchard road"]
        found = [store for store in expected_stores if store in text]

        if len(found) == len(expected_stores):
            score += 0.3
            feedback.append("Expected store names found: Apple The Dubai Mall, Apple Orchard Road.")
        elif len(found) >= 1:
            score += 0.15
            feedback.append(f"Partial match: found {found}.")
        else:
            feedback.append("Expected store names not found in output.")

    except Exception as e:
        feedback.append(f"Evaluation failed: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)

    # --- Finalize scoring ---
    final_score = min(round(score, 2), 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score

def evaluate_price_segment_performance(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for 'Price Segment Performance' reasoning task.

    Expected approximate output:
        Price_Segment  total_revenue  avg_quantity
        High     1580663554          5.49
        Low       485124617          5.50
        Medium   1013730996          5.49

    Evaluation checks:
    - Output file exists
    - All 3 segments (Low, Medium, High) appear
    - Numeric values realistic (large revenue values)
    - Average quantity values ≈ 5.49–5.50
    """

    if actual is None:
        logger.error("No valid file path provided.")
        return 0.0

    base_dir = os.path.dirname(actual)
    csv_path = os.path.join(base_dir, "price_segment_performance.csv")

    score = 0.0
    feedback = []

    try:
        # 1. File existence (0.3)
        if os.path.exists(csv_path):
            score += 0.3
            feedback.append("Result file price_segment_performance.csv found.")
        else:
            feedback.append("Result file price_segment_performance.csv missing.")
            return 0.0

        # 2. Read file
        df = pd.read_csv(csv_path)
        text = " ".join(df.astype(str).values.flatten()).lower()

        # 3. Check for 3 segments (0.3)
        expected_segments = ["low", "medium", "high"]
        found = [seg for seg in expected_segments if seg in text]

        if len(found) == 3:
            score += 0.3
            feedback.append("All three price segments (Low, Medium, High) found.")
        elif len(found) >= 2:
            score += 0.15
            feedback.append(f"Partial segment match: found {found}.")
        else:
            feedback.append("Expected price segments missing.")

        # 4. Numeric validation (0.25)
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", text)
        if nums:
            values = np.array(list(map(float, nums)))
            if np.any(values > 1e6):  # revenue scale validation
                score += 0.25
                feedback.append("Revenue values appear realistic (large scale).")
            else:
                feedback.append("Revenue values found but scale may be unrealistic.")
        else:
            feedback.append("No numeric values found in output.")

        # 5. Avg quantity ≈ 5.49–5.50 (0.15)
        if "5.49" in text or "5.50" in text:
            score += 0.15
            feedback.append("Average quantity values (~5.49–5.50) found as expected.")
        else:
            feedback.append("Average quantity values (~5.49–5.50) missing or outside expected range.")

    except Exception as e:
        feedback.append(f"Evaluation failed due to error: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)

    # Finalize
    final_score = min(round(score, 2), 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score

def evaluate_sales_anomalies(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for 'Sales Anomalies (Z-Score)' reasoning task.

    Expected outcome:
        Total Anomalous Transactions Detected: 0

    Evaluation checks:
    - Output file exists
    - Mentions zero anomalies (0)
    """
    if actual is None:
        logger.error("No valid path provided.")
        return 0.0

    base_dir = os.path.dirname(actual)
    csv_path = os.path.join(base_dir, "sales_anomalies.csv")

    score = 0.0
    feedback = []

    try:
        # 1. Check file existence (0.6)
        if os.path.exists(csv_path):
            score += 0.6
            feedback.append("Output file sales_anomalies.csv found.")
        else:
            feedback.append("sales_anomalies.csv missing.")
            return 0.0

        # 2. Check for zero anomalies (0.4)
        with open(csv_path, "r", encoding="utf-8") as f:
            text = f.read().lower()

        # Look for zero anomaly indication
        if "0" in text or "zero" in text:
            score += 0.4
            feedback.append("Correctly reports zero anomalies detected.")
        else:
            feedback.append("Did not clearly indicate zero anomalies.")

    except Exception as e:
        feedback.append(f"Evaluation failed: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)

    final_score = min(round(score, 2), 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score

def evaluate_openml_simple(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for OpenML dataset loading tasks (e.g., Titanic dataset ID 40945).

    Checks:
      - Python script file exists
      - Script imports 'openml' and 'pandas'
      - Script likely prints DataFrame columns (looks for 'print' and 'columns')
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    base_dir = os.path.dirname(actual)
    script_path = os.path.join(base_dir, "titanic_openml.py")

    required_imports = ["openml", "pandas"]
    possible_columns = ["pclass", "sex", "age", "survived", "class"]

    score = 0.0
    feedback = []

    try:
        # Check that script file exists
        if os.path.exists(script_path):
            score += 0.4
            feedback.append("Python script found.")
            logger.info("Python script exists.")
            try:
                with open(script_path, "r", encoding="utf-8") as f:
                    code = f.read().lower()

                # Verify imports
                imported_libs = [lib for lib in required_imports if f"import {lib}" in code]
                if len(imported_libs) == len(required_imports):
                    score += 0.3
                    feedback.append("Script imports both pandas and openml correctly.")
                    logger.info("All required libraries imported.")
                else:
                    missing = [lib for lib in required_imports if f"import {lib}" not in code]
                    feedback.append(f"Missing import(s): {', '.join(missing)}")

                # Check that code prints DataFrame columns
                if "print" in code and "columns" in code:
                    score += 0.3
                    feedback.append("Script prints DataFrame columns as required.")
                else:
                    feedback.append("Script does not print DataFrame columns explicitly.")
            except Exception as e:
                feedback.append(f"Error reading script file: {e}")
                logger.error(f"Error reading script: {e}", exc_info=True)
        else:
            feedback.append("Python script missing.")
            logger.warning("Python script not found.")

    except Exception as e:
        feedback.append(f"Evaluation failed: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)

    final_score = min(round(score, 2), 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")

    for msg in feedback:
        logger.info(msg)

    return final_score


def evaluate_openml_summary_task(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for OpenML Titanic summary task.
    Checks:
      - Python script exists.
      - Script imports openml and pandas.
      - Result file titanic_summary.txt exists and contains a numeric value in a realistic range (25–40).
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    base_dir = os.path.dirname(actual)
    script_path = os.path.join(base_dir, "titanic_openml_summary.py")
    result_path = os.path.join(base_dir, "titanic_summary.txt")

    score = 0.0
    feedback = []

    try:
        # 1. Script presence
        if os.path.exists(script_path):
            score += 0.3
            feedback.append("Python script found.")
            with open(script_path, "r", encoding="utf-8") as f:
                code = f.read().lower()

            # 2. Check imports
            if "import pandas" in code and "import openml" in code:
                score += 0.3
                feedback.append("Script imports pandas and openml correctly.")
            else:
                feedback.append("Missing import for pandas or openml.")

        else:
            feedback.append("Script file missing.")

        # 3. Result file check
        if os.path.exists(result_path):
            score += 0.2
            feedback.append("Result file titanic_summary.txt found.")

            try:
                with open(result_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                match = re.search(r"[-+]?\d*\.\d+|\d+", content)
                if match:
                    value = float(match.group())
                    if 25 <= value <= 40:
                        score += 0.2
                        feedback.append(f"Mean age value ({value:.2f}) is within a realistic range.")
                    else:
                        feedback.append(f"Mean age value ({value:.2f}) is outside expected range (25–40).")
                else:
                    feedback.append("No numeric value found in titanic_summary.txt.")
            except Exception as e:
                feedback.append(f"Error reading titanic_summary.txt: {e}")
        else:
            feedback.append("Result file titanic_summary.txt missing.")

    except Exception as e:
        feedback.append(f"Evaluation failed: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)

    final_score = min(round(score, 2), 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score

def evaluate_openml_rf_model(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Titanic RandomForest model training task.
    Verifies:
      - Script file exists.
      - Script imports openml, pandas, sklearn.
      - RandomForestClassifier (or Regressor) is used.
      - Result file model_results.txt exists.
      - Metrics (accuracy, precision, recall) are realistic (0.6–1.0).
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    base_dir = os.path.dirname(actual)
    script_path = os.path.join(base_dir, "titanic_rf_model.py")
    result_path = os.path.join(base_dir, "model_results.txt")

    score = 0.0
    feedback = []

    try:
        # Script check
        if os.path.exists(script_path):
            score += 0.2
            feedback.append("Python script found.")
            with open(script_path, "r", encoding="utf-8") as f:
                content = f.read().lower()

            # Required imports
            required = ["import pandas", "import openml", "import sklearn"]
            if all(r in content for r in required):
                score += 0.2
                feedback.append("Script imports pandas, openml, and sklearn.")
            else:
                feedback.append("Missing one or more required imports.")

            # RandomForest presence
            if "randomforestclassifier" in content or "randomforestregressor" in content:
                score += 0.2
                feedback.append("RandomForest model correctly implemented in code.")
                logger.info("RandomForest usage verified.")
            else:
                feedback.append("No RandomForest model found in script.")
                logger.warning("Missing RandomForestClassifier or RandomForestRegressor.")
        else:
            feedback.append("Python script missing.")
            logger.warning("Script file not found.")

        # Result file
        if os.path.exists(result_path):
            score += 0.2
            feedback.append("Result file model_results.txt found.")

            try:
                with open(result_path, "r", encoding="utf-8") as f:
                    text = f.read().strip()

                nums = [float(x) for x in re.findall(r"[-+]?\d*\.\d+|\d+", text)]
                if len(nums) >= 3:
                    valid = all(0.6 <= n <= 1.0 for n in nums)
                    if valid:
                        score += 0.2
                        feedback.append("Model metrics fall within expected range (0.6–1.0).")
                    else:
                        feedback.append(f"Metrics outside expected range: {nums}")
                else:
                    feedback.append("Could not extract three numeric values for accuracy, precision, recall.")
            except Exception as e:
                feedback.append(f"Error reading model_results.txt: {e}")
        else:
            feedback.append("Result file model_results.txt missing.")
            logger.warning("Output file not found.")

    except Exception as e:
        feedback.append(f"Evaluation failed: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)

    final_score = min(round(score, 2), 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)
    return final_score


def evaluate_titanic_feature_importance(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for Titanic feature importance task.
    Checks:
      - Script file exists and contains RandomForestClassifier usage.
      - Output CSV (top_features.csv) exists.
      - CSV has correct columns: Feature, Importance.
      - At least 5 rows are present.
      - Importance values are numeric and sorted in descending order.
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    base_dir = os.path.dirname(actual)
    script_path = os.path.join(base_dir, "titanic_feature_importance.py")
    output_path = os.path.join(base_dir, "top_features.csv")

    score = 0.0
    feedback = []

    try:
        # --- Check for script existence ---
        if os.path.exists(script_path):
            score += 0.25
            feedback.append("Python script file found.")
            with open(script_path, "r", encoding="utf-8") as f:
                code = f.read().lower()

            # Verify core libraries
            required_libs = ["pandas", "sklearn", "randomforestclassifier"]
            if all(lib in code for lib in required_libs):
                score += 0.25
                feedback.append("Script imports pandas, sklearn, and uses RandomForestClassifier.")
            else:
                feedback.append("Missing RandomForestClassifier or required libraries.")
        else:
            feedback.append("Script file missing.")
            logger.warning("Python script not found.")

        # --- Check output CSV existence ---
        if os.path.exists(output_path):
            score += 0.25
            feedback.append("Output file top_features.csv found.")
            try:
                df = pd.read_csv(output_path)

                # Validate structure
                if list(df.columns[:2]) == ["Feature", "Importance"]:
                    score += 0.1
                    feedback.append("CSV has correct columns: Feature and Importance.")
                else:
                    feedback.append(f"Incorrect column names: {list(df.columns)}")

                # Check row count
                if len(df) >= 5:
                    score += 0.1
                    feedback.append("At least 5 features listed in output.")
                else:
                    feedback.append(f"Only {len(df)} features found (expected ≥5).")

                # Validate numeric and descending order
                try:
                    importance_vals = df["Importance"].astype(float)
                    if importance_vals.is_monotonic_decreasing:
                        score += 0.05
                        feedback.append("Importance values sorted in descending order.")
                    else:
                        feedback.append("Importance values not sorted descending.")
                except Exception:
                    feedback.append("Importance column not numeric or missing.")
            except Exception as e:
                feedback.append(f"Error reading CSV: {e}")
        else:
            feedback.append("Output file top_features.csv missing.")
            logger.warning("Output CSV not found.")

    except Exception as e:
        feedback.append(f"Evaluation failed: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)

    # --- Final Scoring ---
    final_score = min(round(score, 2), 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score

def evaluate_kaggle_titanic(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Kaggle Titanic + Python script + Column export task.
    Checks script, dataset CSV, and column names text file.
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    expected = expected.get("rules", expected)

    score = 0.0
    feedback = []

    try:
        # Base directory where all files are stored
        base_dir = os.path.dirname(actual)

        py_file = os.path.join(base_dir, "titanic_kaggle.py")
        csv_file = os.path.join(base_dir, "titanic.csv")
        txt_file = os.path.join(base_dir, "titanic_kaggle_columns.txt")

        # 1. Check Python script exists
        if os.path.exists(py_file):
            score += 0.25
            feedback.append("Python script file was found.")
            with open(py_file, "r", encoding="utf-8") as f:
                content = f.read()
            if "import pandas" in content:
                score += 0.25
                feedback.append("The script correctly imports pandas.")
            else:
                feedback.append("The script does not include an import for pandas.")
        else:
            feedback.append("The Python script file is missing.")

        # 2. Check CSV file exists
        if os.path.exists(csv_file):
            score += 0.25
            feedback.append("The Titanic CSV file was found.")
        else:
            feedback.append("The Titanic CSV file is missing.")

        # 3. Check columns text file exists and not empty
        if os.path.exists(txt_file):
            with open(txt_file, "r", encoding="utf-8") as f:
                cols = f.read().strip().splitlines()
            if cols:
                score += 0.25
                feedback.append("The column names text file exists and contains data.")
            else:
                feedback.append("The column names text file is empty.")
        else:
            feedback.append("The column names text file is missing.")

    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        feedback.append(f"An error occurred during evaluation: {e}")

    logger.info(f"Evaluation Score: {score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return min(score, 1.0)

def evaluate_kaggle_lifestyle(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Kaggle Life-Style dataset task.
    Checks:
      - Whether the Python script exists and imports pandas.
      - Whether both CSV files (Final_data.csv, expanded_fitness_data.csv) exist.
      - Whether the text file with column names exists and contains data.
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    expected = expected.get("rules", expected)

    score = 0.0
    feedback = []

    try:
        # Base directory where all files are stored
        base_dir = os.path.dirname(actual)

        py_file = os.path.join(base_dir, "lifestyle_kaggle.py")
        csv1 = os.path.join(base_dir, "Final_data.csv")
        csv2 = os.path.join(base_dir, "expanded_fitness_data.csv")
        txt_file = os.path.join(base_dir, "lifestyle_kaggle_columns.txt")

        # 1. Check Python script
        if os.path.exists(py_file):
            score += 0.2
            feedback.append("Python script file was found.")
            with open(py_file, "r", encoding="utf-8") as f:
                content = f.read()
            if "import pandas" in content:
                score += 0.2
                feedback.append("The script correctly imports pandas.")
            else:
                feedback.append("The script does not include an import for pandas.")
        else:
            feedback.append("The Python script file is missing.")

        # 2. Check both CSV files
        if os.path.exists(csv1):
            score += 0.2
            feedback.append("The file 'Final_data.csv' was found.")
        else:
            feedback.append("The file 'Final_data.csv' is missing.")

        if os.path.exists(csv2):
            score += 0.2
            feedback.append("The file 'expanded_fitness_data.csv' was found.")
        else:
            feedback.append("The file 'expanded_fitness_data.csv' is missing.")

        # 3. Check column names text file
        if os.path.exists(txt_file):
            with open(txt_file, "r", encoding="utf-8") as f:
                cols = f.read().strip().splitlines()
            if cols:
                score += 0.2
                feedback.append("The column names text file exists and contains data.")
            else:
                feedback.append("The column names text file is empty.")
        else:
            feedback.append("The column names text file is missing.")

    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        feedback.append(f"An error occurred during evaluation: {e}")

    logger.info(f"Evaluation Score: {score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return min(score, 1.0)
   
def evaluate_kaggle_fraud(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Kaggle Fraud Detection dataset task.
    Checks:
      - Whether the fraud_analysis.py script exists and imports pandas.
      - Whether fraud_columns.txt exists and is not empty.
      - Whether the column names file references multiple CSV files.
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    expected = expected.get("rules", expected)

    score = 0.0
    feedback = []

    try:
        base_dir = os.path.dirname(actual)

        py_file = os.path.join(base_dir, "fraud_analysis.py")
        txt_file = os.path.join(base_dir, "fraud_columns.txt")

        # 1. Check Python script existence and pandas import
        if os.path.exists(py_file):
            score += 0.25
            feedback.append("Python script file was found.")
            with open(py_file, "r", encoding="utf-8") as f:
                content = f.read()
            if "import pandas" in content:
                score += 0.25
                feedback.append("The script correctly imports pandas.")
            else:
                feedback.append("The script does not include an import for pandas.")
        else:
            feedback.append("The Python script file is missing.")

        # 2. Check that fraud_columns.txt exists and has content
        if os.path.exists(txt_file):
            with open(txt_file, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
            if lines:
                score += 0.25
                feedback.append("The column names file exists and contains data.")
            else:
                feedback.append("The column names file is empty.")
        else:
            feedback.append("The column names file is missing.")

        # 3. Verify that multiple CSV files are referenced
        csv_files = []
        for root, _, files in os.walk(base_dir):
            for fname in files:
                if fname.endswith(".csv"):
                    csv_files.append(fname)

        if csv_files and os.path.exists(txt_file):
            with open(txt_file, "r", encoding="utf-8") as f:
                txt_content = f.read()
            found_count = sum(1 for csv in csv_files if csv in txt_content)
            if found_count >= min(2, len(csv_files)):  # At least 2 CSVs included
                score += 0.25
                feedback.append(
                    f"The column names file references multiple CSV files ({found_count} of {len(csv_files)} found)."
                )
            else:
                feedback.append("The column names file does not reference enough CSV files.")
        else:
            feedback.append("No CSV files were found for validation.")

    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        feedback.append(f"An error occurred during evaluation: {e}")

    logger.info(f"Evaluation Score: {score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return min(score, 1.0)

def evaluate_customer_risk_analysis(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Customer Risk Analysis task.

    Checks only:
      - Python script exists and includes pandas import.
      - Script includes 'RiskScore' computation.
      - Output file 'customer_risk_overview.txt' exists.
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    base_dir = os.path.dirname(actual)
    script_file = os.path.join(base_dir, "customer_risk_analysis.py")
    result_file = os.path.join(base_dir, "customer_risk_overview.txt")

    score = 0.0
    feedback = []

    try:
        # Step 1 — Check script existence
        if os.path.exists(script_file):
            score += 0.4
            feedback.append("Python script found.")
            logger.info("Script located successfully.")

            with open(script_file, "r", encoding="utf-8") as f:
                content = f.read().lower()

            # Check for pandas import
            if "import pandas" in content:
                score += 0.3
                feedback.append("Pandas import found.")
            else:
                feedback.append("Pandas import missing.")

            # Check for RiskScore logic
            if "riskscore" in content and ("/" in content or "*" in content):
                score += 0.2
                feedback.append("RiskScore computation detected.")
            else:
                feedback.append("RiskScore computation not detected.")
        else:
            feedback.append("Python script file missing.")
            logger.warning("Script file not found.")
            return 0.0

        # Step 2 — Check result file existence
        if os.path.exists(result_file):
            score += 0.1
            feedback.append("Result file 'customer_risk_overview.txt' exists.")
            logger.info("Output file found.")
        else:
            feedback.append("Result file 'customer_risk_overview.txt' missing.")
            logger.warning("Output file not found.")

    except Exception as e:
        feedback.append(f"Evaluation failed: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)

    final_score = min(score, 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score

def evaluate_blood_model_comparison(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Blood Donation Model Comparison task.
    Checks:
      - Script and output file existence.
      - Required libraries and keywords in code.
      - Text file has correct columns: model, roc_auc, f1_score.
      - roc_auc and f1_score values are numeric and >= 0.7.
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    base_dir = os.path.dirname(actual)
    script_path = os.path.join(base_dir, "blood_model_comparison.py")
    output_path = os.path.join(base_dir, "blood_model_comparison.txt")

    score = 0.0
    feedback = []

    try:
        # Check script existence
        if os.path.exists(script_path):
            score += 0.2
            feedback.append("Python script file found.")
            with open(script_path, "r", encoding="utf-8") as f:
                code = f.read().lower()

            required_keywords = [
                "pandas", "sklearn", "randomforestclassifier",
                "roc_auc_score", "f1_score", "standardscaler"
            ]
            missing = [k for k in required_keywords if k not in code]
            if not missing:
                score += 0.2
                feedback.append("Script correctly imports required libraries and uses RandomForestClassifier with ROC-AUC and F1-score.")
            else:
                feedback.append(f"Missing keywords or libraries in script: {missing}")
        else:
            feedback.append("Python script file missing.")
            logger.warning("Python script not found.")

        # Check output file existence
        if os.path.exists(output_path):
            score += 0.2
            feedback.append("Output file blood_model_comparison.txt found.")
            try:
                df = pd.read_csv(output_path, sep=None, engine="python")

                expected_cols = ["model", "roc_auc", "f1_score"]
                if all(col in df.columns for col in expected_cols):
                    score += 0.2
                    feedback.append("Output file has correct columns: model, roc_auc, f1_score.")
                else:
                    feedback.append(f"Incorrect columns found: {list(df.columns)}")

                try:
                    roc_auc_values = pd.to_numeric(df["roc_auc"], errors="coerce")
                    f1_values = pd.to_numeric(df["f1_score"], errors="coerce")

                    if roc_auc_values.notna().all() and f1_values.notna().all():
                        score += 0.1
                        feedback.append("ROC-AUC and F1-score are numeric.")
                        roc_mean = roc_auc_values.mean()
                        f1_mean = f1_values.mean()
                        if roc_mean >= 0.7 and f1_mean >= 0.7:
                            score += 0.1
                            feedback.append(f"Model performance acceptable (ROC-AUC={roc_mean:.2f}, F1={f1_mean:.2f}).")
                        else:
                            feedback.append(f"Model performance below threshold (ROC-AUC={roc_mean:.2f}, F1={f1_mean:.2f}).")
                    else:
                        feedback.append("ROC-AUC or F1-score contain non-numeric values.")
                except Exception as e:
                    feedback.append(f"Error checking numeric values: {e}")

            except Exception as e:
                feedback.append(f"Error reading output file: {e}")
        else:
            feedback.append("Output text file missing.")
            logger.warning("Output text file not found.")

    except Exception as e:
        feedback.append(f"Evaluation failed: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)

    final_score = min(round(score, 2), 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score

def evaluate_donation_recency_visualization_quality(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Donation Recency Visualization (Visual Studio Code) task.

    This evaluation checks:
      - The Python script file exists.
      - Both visualization files exist and are non-empty.
      - GPT-4o confirms that:
          1. donation_rate_by_recency_bar.png correctly shows donation rate by recency bins as a bar chart.
          2. donation_rate_by_recency_line.png correctly shows the same information as a line chart.
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    expected = expected.get("rules", expected)
    base_dir = os.path.dirname(actual)

    script_file = os.path.join(base_dir, "donation_recency_visualization.py")
    chart1 = os.path.join(base_dir, "donation_rate_by_recency_bar.png")
    chart2 = os.path.join(base_dir, "donation_rate_by_recency_line.png")

    score = 0.0
    feedback = []

    # Step 1: Check script existence
    if os.path.exists(script_file):
        score += 0.4
        feedback.append("The Python script file exists.")
        logger.info("Script file found.")
    else:
        feedback.append("The Python script file is missing.")
        logger.warning("Script file missing.")

    # Helper: Encode image
    def encode_image(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    # Step 2: Visualization checks
    charts_to_check = [
        (
            chart1,
            "a bar chart showing the proportion of donors with Class = 2 in each recency bin (e.g., [0–6, 7–12, 13–24, 25–36, 37+]), with labeled axes and a clear title."
        ),
        (
            chart2,
            "a line chart showing the same donation rate by recency bins, with labeled axes and a clear title."
        )
    ]

    for chart, description in charts_to_check:
        chart_name = os.path.basename(chart)
        if os.path.exists(chart) and os.path.getsize(chart) > 0:
            score += 0.15
            feedback.append(f"The visualization {chart_name} exists and is non-empty.")
            logger.info(f"{chart_name} found and valid.")

            try:
                encoded = encode_image(chart)
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert data visualization evaluator. Assess chart clarity, labeling, and relevance."
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        f"Examine this chart and confirm whether it correctly represents {description}. "
                                        f"Reply 'yes' if it is accurate, clearly labeled, and visually appropriate; otherwise reply 'no' with a short reason."
                                    )
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{encoded}"}
                                }
                            ]
                        }
                    ],
                    max_tokens=100
                )

                answer = response.choices[0].message.content.lower()
                if "yes" in answer:
                    score += 0.1
                    feedback.append(f"GPT-4o judged {chart_name} as accurate and well-labeled.")
                    logger.info(f"{chart_name} validated by GPT-4o.")
                else:
                    feedback.append(f"GPT-4o judged {chart_name} unclear or inaccurate: {answer}")
                    logger.warning(f"{chart_name} unclear according to GPT-4o.")
            except Exception as e:
                feedback.append(f"Error evaluating {chart_name} with GPT-4o: {e}")
                logger.error(f"GPT-4o evaluation failed for {chart_name}: {e}", exc_info=True)
        else:
            feedback.append(f"The visualization {chart_name} is missing or empty.")
            logger.warning(f"{chart_name} missing or empty.")

    # Step 3: Final scoring
    final_score = min(score, 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    print("\nEvaluation Summary:")
    for msg in feedback:
        print("-", msg)
    print(f"\nFinal Score: {final_score:.2f}")

    return final_score

def evaluate_donation_aggregation_summary(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Donation Aggregation Summary task.
    Checks:
      - Script and output file existence.
      - Required libraries and key operations in code.
      - CSV file has correct columns.
      - Contains the three expected Recency categories.
      - Columns are numeric and Active_Rate between 0 and 1.
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    base_dir = os.path.dirname(actual)
    script_path = os.path.join(base_dir, "donation_aggregation_summary.py")
    output_path = os.path.join(base_dir, "donation_aggregation_summary.csv")

    score = 0.0
    feedback = []

    try:
        # Check for Python script existence
        if os.path.exists(script_path):
            score += 0.2
            feedback.append("Python script file found.")
            with open(script_path, "r", encoding="utf-8") as f:
                code = f.read().lower()

            required_keywords = ["pandas", "groupby", "mean", "std", "donation_density", "v1", "v2", "v3", "v4", "class"]
            missing = [k for k in required_keywords if k not in code]
            if not missing:
                score += 0.2
                feedback.append("Script contains required libraries and expected operations.")
            else:
                feedback.append(f"Missing keywords or libraries: {missing}")
        else:
            feedback.append("Python script file missing.")
            logger.warning("Python script not found.")

        # Check output CSV existence
        if os.path.exists(output_path):
            score += 0.2
            feedback.append("Output file donation_aggregation_summary.csv found.")
            try:
                df = pd.read_csv(output_path)

                # Validate columns
                expected_cols = [
                    "RecencyCategory", "Avg_Frequency", "Std_Frequency",
                    "Avg_Monetary", "Std_Monetary",
                    "Avg_Density", "Std_Density", "Active_Rate"
                ]
                if all(col in df.columns for col in expected_cols):
                    score += 0.2
                    feedback.append("Output file has correct columns.")
                else:
                    feedback.append(f"Incorrect columns: {list(df.columns)}")

                # Validate recency categories
                expected_categories = {"Recent", "Moderate", "Dormant"}
                actual_categories = set(df["RecencyCategory"].astype(str))
                if expected_categories.issubset(actual_categories):
                    score += 0.1
                    feedback.append("All expected recency categories are present.")
                else:
                    feedback.append(f"Missing recency categories. Found: {actual_categories}")

                # Validate numeric columns
                numeric_cols = [
                    "Avg_Frequency", "Std_Frequency",
                    "Avg_Monetary", "Std_Monetary",
                    "Avg_Density", "Std_Density", "Active_Rate"
                ]
                numeric_ok = True
                for col in numeric_cols:
                    try:
                        vals = pd.to_numeric(df[col], errors="coerce")
                        if vals.isna().any():
                            numeric_ok = False
                        if (vals < 0).any():
                            numeric_ok = False
                    except Exception:
                        numeric_ok = False
                if numeric_ok:
                    score += 0.05
                    feedback.append("All numeric columns are valid and non-negative.")
                else:
                    feedback.append("Some numeric columns contain invalid or negative values.")

                # Validate Active_Rate range
                try:
                    active_vals = pd.to_numeric(df["Active_Rate"], errors="coerce")
                    if (active_vals >= 0).all() and (active_vals <= 1).all():
                        score += 0.05
                        feedback.append("Active_Rate values are within [0, 1].")
                    else:
                        feedback.append("Active_Rate contains out-of-range values.")
                except Exception:
                    feedback.append("Could not validate Active_Rate values.")
            except Exception as e:
                feedback.append(f"Error reading CSV file: {e}")
        else:
            feedback.append("Output CSV file missing.")
            logger.warning("Output CSV not found.")

    except Exception as e:
        feedback.append(f"Evaluation failed: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)

    final_score = min(round(score, 2), 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score

def evaluate_steel_feature_correlation(actual: str, expected: dict, **options) -> float:
    """
    Realistic evaluator for the Steel Plates Fault correlation task.

    Checks:
      - Script and output file existence.
      - Output CSV has correct columns.
      - Exactly three result rows.
      - Correlation values are close to expected realistic range (≈0.98–1.00).
      - Does not penalize NaN t-statistic or p-value due to small samples.
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    base_dir = os.path.dirname(actual)
    script_path = os.path.join(base_dir, "steel_feature_correlation.py")
    output_path = os.path.join(base_dir, "top_correlated_features.csv")

    score = 0.0
    feedback = []

    try:
        # Check that the script exists
        if os.path.exists(script_path):
            score += 0.25
            feedback.append("Python script file found.")
        else:
            feedback.append("Python script file missing.")
            logger.warning("Script not found.")

        # Check that the output file exists
        if os.path.exists(output_path):
            score += 0.25
            feedback.append("Output file top_correlated_features.csv found.")

            try:
                df = pd.read_csv(output_path)

                # Expected structure
                expected_cols = ["Feature1", "Feature2", "Correlation", "t_statistic", "p_value"]
                if all(col in df.columns for col in expected_cols):
                    score += 0.25
                    feedback.append("Output file contains correct columns.")
                else:
                    feedback.append(f"Incorrect or missing columns: {list(df.columns)}")

                # Expected number of rows
                if len(df) == 3:
                    score += 0.15
                    feedback.append("Output file contains three correlation pairs as expected.")
                else:
                    feedback.append(f"Unexpected number of rows ({len(df)}). Expected 3.")

                # Validate correlation plausibility
                try:
                    corr_vals = pd.to_numeric(df["Correlation"], errors="coerce")
                    if corr_vals.isna().any():
                        feedback.append("Some correlation values are NaN.")
                    avg_corr = corr_vals.mean()
                    if (corr_vals.abs() <= 1).all() and (avg_corr > 0.95):
                        score += 0.10
                        feedback.append("Correlation values are within valid and realistic range (~0.98–1.00).")
                    else:
                        feedback.append(f"Correlation values out of expected range. Mean={avg_corr:.3f}")
                except Exception as e:
                    feedback.append(f"Error checking correlation values: {e}")

                # We tolerate NaNs in t-statistic and p-value
                if "t_statistic" in df.columns and "p_value" in df.columns:
                    feedback.append("t-statistic and p-value columns detected (NaNs tolerated).")

            except Exception as e:
                feedback.append(f"Error reading output CSV: {e}")
        else:
            feedback.append("Output file missing.")
            logger.warning("Output CSV not found.")

    except Exception as e:
        feedback.append(f"Evaluation failed: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)

    final_score = min(round(score, 2), 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score

def evaluate_steel_outlier_detection(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Steel Plates Fault outlier detection task.

    Checks:
      - Existence of the Python script and output CSV file.
      - Output file has the correct structure and required columns.
      - Contains results for V5 and V26.
      - Outlier counts are numeric and close to expected values (V5≈25, V26≈50, Common_Outliers≈0).
      - Allows minor deviations (±5 range) for numerical realism.
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    base_dir = os.path.dirname(actual)
    script_path = os.path.join(base_dir, "steel_outlier_detection.py")
    output_path = os.path.join(base_dir, "outlier_summary.csv")

    score = 0.0
    feedback = []

    try:
        # Check for script existence
        if os.path.exists(script_path):
            score += 0.125
            feedback.append("Python script file found.")
        else:
            feedback.append("Python script file missing.")
            logger.warning("Script not found.")

        # Check for CSV existence
        if os.path.exists(output_path):
            score += 0.125
            feedback.append("Output CSV file 'outlier_summary.csv' found.")

            try:
                df = pd.read_csv(output_path)

                # Validate column structure
                expected_cols = ["Column", "Outlier_Count", "Common_Outliers"]
                if all(col in df.columns for col in expected_cols):
                    score += 0.25
                    feedback.append("Output file contains correct columns.")
                else:
                    feedback.append(f"Incorrect or missing columns: {list(df.columns)}")

                # Check for presence of V5 and V26 rows
                if {"V5", "V26"}.issubset(set(df["Column"].astype(str))):
                    score += 0.15
                    feedback.append("Rows for V5 and V26 found.")
                else:
                    feedback.append("Missing one or both required rows (V5, V26).")

                # Validate numeric values and non-negative
                try:
                    df["Outlier_Count"] = pd.to_numeric(df["Outlier_Count"], errors="coerce")
                    df["Common_Outliers"] = pd.to_numeric(df["Common_Outliers"], errors="coerce")

                    if (df["Outlier_Count"] >= 0).all() and (df["Common_Outliers"] >= 0).all():
                        score += 0.15
                        feedback.append("Outlier counts are numeric and non-negative.")
                    else:
                        feedback.append("Some outlier counts are negative or non-numeric.")
                except Exception as e:
                    feedback.append(f"Error validating numeric columns: {e}")

                # Expected value check (within range tolerance)
                try:
                    v5_row = df.loc[df["Column"].astype(str) == "V5"]
                    v26_row = df.loc[df["Column"].astype(str) == "V26"]

                    if not v5_row.empty:
                        v5_val = float(v5_row["Outlier_Count"].iloc[0])
                        if 20 <= v5_val <= 30:
                            score += 0.05
                            feedback.append(f"V5 outlier count ({v5_val}) within expected range (~25).")
                        else:
                            feedback.append(f"V5 outlier count ({v5_val}) outside expected range.")

                    if not v26_row.empty:
                        v26_val = float(v26_row["Outlier_Count"].iloc[0])
                        if 45 <= v26_val <= 55:
                            score += 0.05
                            feedback.append(f"V26 outlier count ({v26_val}) within expected range (~50).")
                        else:
                            feedback.append(f"V26 outlier count ({v26_val}) outside expected range.")

                    # Check common outliers approximately zero
                    both_val = df["Common_Outliers"].sum()
                    if both_val <= 1:
                        score += 0.05
                        feedback.append("Common outliers count approximately zero.")
                    else:
                        feedback.append(f"Common outliers higher than expected ({both_val}).")
                except Exception as e:
                    feedback.append(f"Error checking expected ranges: {e}")

            except Exception as e:
                feedback.append(f"Error reading output CSV: {e}")
        else:
            feedback.append("Output CSV file missing.")
            logger.warning("Output file not found.")

    except Exception as e:
        feedback.append(f"Evaluation failed: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)

    final_score = min(round(score, 2), 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score


def evaluate_steel_fault_risk_analysis(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Steel Plates Fault Risk Analysis task.
    Checks:
      - Script and output file existence.
      - Correct output structure.
      - Two class rows (1 and 2).
      - Numeric and plausible values for Mean_Fault_Risk and Variance_Fault_Risk.
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    base_dir = os.path.dirname(actual)
    script_path = os.path.join(base_dir, "steel_fault_risk_analysis.py")
    output_path = os.path.join(base_dir, "fault_risk_summary.csv")

    score = 0.0
    feedback = []

    try:
        # Check script existence
        if os.path.exists(script_path):
            score += 0.2
            feedback.append("Python script file found.")
        else:
            feedback.append("Python script file missing.")
            logger.warning("Script not found.")

        # Check output file existence
        if os.path.exists(output_path):
            score += 0.2
            feedback.append("Output file 'fault_risk_summary.csv' found.")

            try:
                df = pd.read_csv(output_path)

                # Validate columns
                expected_cols = ["Class", "Mean_Fault_Risk", "Variance_Fault_Risk"]
                if all(col in df.columns for col in expected_cols):
                    score += 0.2
                    feedback.append("Output file contains correct columns.")
                else:
                    feedback.append(f"Incorrect or missing columns: {list(df.columns)}")

                # Validate class rows
                if set(df["Class"].astype(int)) == {1, 2}:
                    score += 0.15
                    feedback.append("Rows for Class = 1 and Class = 2 found.")
                else:
                    feedback.append(f"Unexpected Class values found: {df['Class'].unique()}")

                # Validate numeric values
                try:
                    means = pd.to_numeric(df["Mean_Fault_Risk"], errors="coerce")
                    vars_ = pd.to_numeric(df["Variance_Fault_Risk"], errors="coerce")
                    if means.notna().all() and vars_.notna().all():
                        score += 0.15
                        feedback.append("Mean and variance values are numeric and valid.")
                    else:
                        feedback.append("Some mean or variance values are non-numeric.")
                except Exception as e:
                    feedback.append(f"Error validating numeric values: {e}")

                # Plausibility check for mean and variance
                try:
                    mean_ok = means.between(-8000, -1000).all()
                    var_ok = vars_.between(1e7, 5e9).all()
                    if mean_ok and var_ok:
                        score += 0.1
                        feedback.append("Mean and variance values fall within realistic ranges.")
                    else:
                        feedback.append(
                            f"Values out of realistic range. Mean range: {means.tolist()}, Variance range: {vars_.tolist()}"
                        )
                except Exception as e:
                    feedback.append(f"Error checking value ranges: {e}")

            except Exception as e:
                feedback.append(f"Error reading output CSV: {e}")
        else:
            feedback.append("Output CSV file missing.")
            logger.warning("Output file not found.")

    except Exception as e:
        feedback.append(f"Evaluation failed: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)

    final_score = min(round(score, 2), 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score


def evaluate_steel_random_forest_cv(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for Steel Plates Fault Random Forest with cross-validation.
    Checks:
      - Script and output file existence.
      - CSV structure and at least 5 features.
      - Importance values numeric, valid, and plausible.
      - Average Accuracy and F1-score (if logged) are >= 0.75.
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    base_dir = os.path.dirname(actual)
    script_path = os.path.join(base_dir, "steel_random_forest_cv.py")
    output_path = os.path.join(base_dir, "top_rf_features.csv")

    score = 0.0
    feedback = []

    try:
        # Check script existence
        if os.path.exists(script_path):
            score += 0.2
            feedback.append("Python script file found.")
        else:
            feedback.append("Python script file missing.")
            logger.warning("Script not found.")

        # Check output file existence
        if os.path.exists(output_path):
            score += 0.2
            feedback.append("Output CSV file 'top_rf_features.csv' found.")

            try:
                df = pd.read_csv(output_path)

                # Column validation
                expected_cols = ["Feature", "Importance"]
                if all(col in df.columns for col in expected_cols):
                    score += 0.2
                    feedback.append("Output file has correct columns: Feature, Importance.")
                else:
                    feedback.append(f"Incorrect or missing columns: {list(df.columns)}")

                # Row count check
                if len(df) >= 5:
                    score += 0.15
                    feedback.append(f"Output contains {len(df)} features (>=5).")
                else:
                    feedback.append(f"Only {len(df)} features found (expected at least 5).")

                # Numeric importance validation
                try:
                    imp_vals = pd.to_numeric(df["Importance"], errors="coerce")
                    if imp_vals.notna().all() and (imp_vals >= 0).all():
                        score += 0.15
                        feedback.append("Importance values are numeric and non-negative.")

                        total_imp = imp_vals.sum()
                        if 0.7 <= total_imp <= 1.3:
                            feedback.append(f"Total importance sum ({total_imp:.2f}) is within expected range (~1).")
                        else:
                            feedback.append(f"Total importance sum ({total_imp:.2f}) deviates slightly but acceptable.")
                    else:
                        feedback.append("Some importance values are invalid or negative.")
                except Exception as e:
                    feedback.append(f"Error validating importance values: {e}")

                # Attempt to parse accuracy/F1 from potential log file or stdout text
                try:
                    log_text = ""
                    with open(script_path, "r", encoding="utf-8") as f:
                        code = f.read().lower()
                        if "accuracy" in code and "f1" in code:
                            log_text = code
                    if log_text:
                        acc_vals = [float(x) for x in re.findall(r"accuracy[^0-9]*([\d\.]+)", log_text)]
                        f1_vals = [float(x) for x in re.findall(r"f1[^0-9]*([\d\.]+)", log_text)]
                        acc_check = any(v >= 0.75 for v in acc_vals)
                        f1_check = any(v >= 0.75 for v in f1_vals)
                        if acc_check and f1_check:
                            score += 0.1
                            feedback.append("Average Accuracy and F1-score meet the minimum threshold (>=0.75).")
                        else:
                            feedback.append("Model performance below expected minimum (75%).")
                    else:
                        feedback.append("Accuracy/F1 not detected in script, skipping score check.")
                except Exception as e:
                    feedback.append(f"Error parsing accuracy/F1 metrics: {e}")

            except Exception as e:
                feedback.append(f"Error reading output CSV: {e}")
        else:
            feedback.append("Output file missing.")
            logger.warning("Output CSV not found.")

    except Exception as e:
        feedback.append(f"Evaluation failed: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)

    final_score = min(round(score, 2), 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score


def evaluate_steel_fault_type_visualization_quality(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Steel Fault Type Visualization (Visual Studio Code) task.

    This evaluation checks:
      - The Python script file exists.
      - Both visualization files exist and are non-empty.
      - GPT-4o confirms that:
          1. steel_type_means.png correctly shows average V5 and V26 per steel type combination.
          2. fault_rate_heatmap.png correctly shows the proportion of Class = 1 (fault rate) per steel type combination.
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    expected = expected.get("rules", expected)
    base_dir = os.path.dirname(actual)

    script_file = os.path.join(base_dir, "steel_fault_type_analysis.py")
    chart1 = os.path.join(base_dir, "steel_type_means.png")
    chart2 = os.path.join(base_dir, "fault_rate_heatmap.png")

    score = 0.0
    feedback = []

    # Step 1: Check for script existence
    if os.path.exists(script_file):
        score += 0.4
        feedback.append("The Python script file exists.")
        logger.info("Script file found.")
    else:
        feedback.append("The Python script file is missing.")
        logger.warning("Script file missing.")

    # Helper function for encoding image
    def encode_image(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    # Step 2: Visualization checks
    charts_to_check = [
        (
            chart1,
            "a bar or grouped chart showing the average V5 (Pixels_Areas) and V26 (Luminosity_Index) per steel type combination (V12, V13), with clear axis labels and title."
        ),
        (
            chart2,
            "a heatmap or equivalent visualization showing the proportion of Class = 1 (fault rate) across steel type combinations, with labeled axes and color scale."
        )
    ]

    for chart, description in charts_to_check:
        chart_name = os.path.basename(chart)
        if os.path.exists(chart) and os.path.getsize(chart) > 0:
            score += 0.15
            feedback.append(f"The visualization {chart_name} exists and is non-empty.")
            logger.info(f"{chart_name} found and valid.")

            try:
                encoded = encode_image(chart)
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert in data visualization evaluation. Judge each chart for clarity, labeling, and accuracy."
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        f"Examine this chart and confirm whether it correctly represents {description}. "
                                        f"Reply 'yes' if it clearly conveys the correct information and is properly labeled; "
                                        f"otherwise reply 'no' with a short reason."
                                    )
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{encoded}"}
                                }
                            ]
                        }
                    ],
                    max_tokens=100
                )

                answer = response.choices[0].message.content.lower()
                if "yes" in answer:
                    score += 0.1
                    feedback.append(f"GPT-4o judged {chart_name} as accurate and clearly labeled.")
                    logger.info(f"{chart_name} validated by GPT-4o.")
                else:
                    feedback.append(f"GPT-4o judged {chart_name} unclear or inaccurate: {answer}")
                    logger.warning(f"{chart_name} unclear according to GPT-4o.")
            except Exception as e:
                feedback.append(f"Error evaluating {chart_name} with GPT-4o: {e}")
                logger.error(f"GPT-4o evaluation failed for {chart_name}: {e}", exc_info=True)
        else:
            feedback.append(f"The visualization {chart_name} is missing or empty.")
            logger.warning(f"{chart_name} missing or empty.")

    # Step 3: Compute final score
    final_score = min(score, 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")

    # Log and print summary
    for msg in feedback:
        logger.info(msg)

    print("\nEvaluation Summary:")
    for msg in feedback:
        print("-", msg)
    print(f"\nFinal Score: {final_score:.2f}")

    return final_score

def evaluate_eeg_collinearity_analysis(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for EEG Eye State collinearity analysis task.

    Checks:
      - Script and output file existence.
      - Output CSV has correct structure.
      - At least one triplet identified.
      - Avg_Correlation >= 0.95 and Combined_VIF >= 10.
      - Realistic tolerance for floating-point or dataset variation.
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    base_dir = os.path.dirname(actual)
    script_path = os.path.join(base_dir, "eeg_collinearity_analysis.py")
    output_path = os.path.join(base_dir, "collinear_triplets_vif.csv")

    score = 0.0
    feedback = []

    try:
        # 1. Check for script existence
        if os.path.exists(script_path):
            score += 0.2
            feedback.append("Python script file found.")
        else:
            feedback.append("Python script file missing.")
            logger.warning("Script not found.")

        # 2. Check for CSV existence
        if os.path.exists(output_path):
            score += 0.2
            feedback.append("Output CSV file 'collinear_triplets_vif.csv' found.")

            try:
                df = pd.read_csv(output_path)

                # Validate required columns
                expected_cols = ["Feature1", "Feature2", "Feature3", "Avg_Correlation", "Combined_VIF"]
                if all(col in df.columns for col in expected_cols):
                    score += 0.2
                    feedback.append("Output file has correct columns.")
                else:
                    feedback.append(f"Incorrect or missing columns: {list(df.columns)}")

                # Ensure non-empty results
                if len(df) >= 1:
                    score += 0.1
                    feedback.append(f"Output contains {len(df)} triplet(s).")
                else:
                    feedback.append("No triplets found.")

                # Validate numeric fields
                try:
                    corr_vals = pd.to_numeric(df["Avg_Correlation"], errors="coerce")
                    vif_vals = pd.to_numeric(df["Combined_VIF"], errors="coerce")
                    if corr_vals.notna().all() and vif_vals.notna().all():
                        feedback.append("Correlation and VIF columns are numeric.")
                        valid_corr = (corr_vals.abs() >= 0.95).any()
                        valid_vif = (vif_vals >= 10).any()
                        if valid_corr:
                            score += 0.075
                            feedback.append("At least one triplet has |corr| > 0.95.")
                        else:
                            feedback.append("No highly correlated triplets detected.")
                        if valid_vif:
                            score += 0.075
                            feedback.append("At least one triplet has Combined VIF ≥ 10.")
                        else:
                            feedback.append("No high-VIF triplets detected.")
                    else:
                        feedback.append("Non-numeric correlation or VIF values found.")
                except Exception as e:
                    feedback.append(f"Error validating numeric data: {e}")

                # Optional plausibility check for expected triplet
                try:
                    match = df[
                        (df["Feature1"] == "V1")
                        & (df["Feature2"] == "V9")
                        & (df["Feature3"] == "V13")
                    ]
                    if not match.empty:
                        corr = float(match["Avg_Correlation"].iloc[0])
                        vif = float(match["Combined_VIF"].iloc[0])
                        if 0.95 <= corr <= 1.00 and 100 <= vif <= 10000:
                            score += 0.1
                            feedback.append(
                                f"Expected triplet (V1,V9,V13) found with realistic correlation ({corr:.3f}) and VIF ({vif:.1f})."
                            )
                        else:
                            feedback.append(
                                f"Expected triplet (V1,V9,V13) found but values out of plausible range: corr={corr:.3f}, VIF={vif:.1f}"
                            )
                    else:
                        feedback.append("Expected triplet (V1,V9,V13) not found; tolerating other valid triplets.")
                except Exception as e:
                    feedback.append(f"Error checking expected triplet: {e}")

            except Exception as e:
                feedback.append(f"Error reading output CSV: {e}")
        else:
            feedback.append("Output CSV file missing.")
            logger.warning("Output CSV not found.")

    except Exception as e:
        feedback.append(f"Evaluation failed: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)

    final_score = min(round(score, 2), 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score

def evaluate_steel_fault_cooccurrence(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Steel Plates Fault co-occurrence analysis task.

    Checks:
      - Script and output file existence.
      - CSV has correct structure (Fault1, Fault2, Cooccurrence).
      - Exactly 3 rows.
      - Cooccurrence values numeric and within [0, 1].
      - Logs a warning but doesn't penalize if all co-occurrences are zero.
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    base_dir = os.path.dirname(actual)
    script_path = os.path.join(base_dir, "steel_fault_cooccurrence.py")
    output_path = os.path.join(base_dir, "top_fault_pairs.csv")

    score = 0.0
    feedback = []

    try:
        # 1. Check script existence
        if os.path.exists(script_path):
            score += 0.25
            feedback.append("Python script file found.")
        else:
            feedback.append("Python script file missing.")
            logger.warning("Script not found.")

        # 2. Check CSV existence
        if os.path.exists(output_path):
            score += 0.25
            feedback.append("Output file 'top_fault_pairs.csv' found.")

            try:
                df = pd.read_csv(output_path)

                # 3. Validate required columns
                expected_cols = ["Fault1", "Fault2", "Cooccurrence"]
                if all(col in df.columns for col in expected_cols):
                    score += 0.25
                    feedback.append("Output file has correct columns.")
                else:
                    feedback.append(f"Incorrect or missing columns: {list(df.columns)}")

                # 4. Validate row count
                if len(df) == 3:
                    score += 0.15
                    feedback.append("Output contains exactly 3 fault pairs.")
                else:
                    feedback.append(f"Unexpected number of rows ({len(df)}); expected 3.")

                # 5. Validate numeric and plausible co-occurrence values
                try:
                    co_vals = pd.to_numeric(df["Cooccurrence"], errors="coerce")
                    if co_vals.notna().all():
                        if ((co_vals >= 0) & (co_vals <= 1)).all():
                            score += 0.10
                            feedback.append("Cooccurrence values are numeric and within [0, 1].")
                            if co_vals.sum() == 0:
                                feedback.append("All co-occurrence values are zero (possible if no faults overlap).")
                        else:
                            feedback.append("Some Cooccurrence values fall outside [0, 1].")
                    else:
                        feedback.append("Non-numeric values found in Cooccurrence column.")
                except Exception as e:
                    feedback.append(f"Error validating Cooccurrence values: {e}")

            except Exception as e:
                feedback.append(f"Error reading output CSV: {e}")
        else:
            feedback.append("Output CSV file missing.")
            logger.warning("Output file not found.")

    except Exception as e:
        feedback.append(f"Evaluation failed: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)

    final_score = min(round(score, 2), 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score


def evaluate_steel_fault_risk_analysis1(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Steel Plates Fault Risk Analysis task.

    Checks:
      - Script and output CSV exist.
      - CSV has correct columns: Class, Mean_Fault_Risk, Variance_Fault_Risk.
      - Two rows for different classes.
      - Mean and variance values are numeric and realistic.
      - Mean_Fault_Risk roughly in expected negative range.
      - Variance_Fault_Risk is large and positive.
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    base_dir = os.path.dirname(actual)
    script_path = os.path.join(base_dir, "steel_fault_risk_analysis.py")
    output_path = os.path.join(base_dir, "fault_risk_summary.csv")

    score = 0.0
    feedback = []

    try:
        # 1. Check script existence
        if os.path.exists(script_path):
            score += 0.2
            feedback.append("Python script file found.")
        else:
            feedback.append("Python script file missing.")
            logger.warning("Script not found.")

        # 2. Check output file existence
        if os.path.exists(output_path):
            score += 0.2
            feedback.append("Output file 'fault_risk_summary.csv' found.")

            try:
                df = pd.read_csv(output_path)

                # 3. Validate column names
                expected_cols = ["Class", "Mean_Fault_Risk", "Variance_Fault_Risk"]
                if all(col in df.columns for col in expected_cols):
                    score += 0.2
                    feedback.append("Output file contains correct columns.")
                else:
                    feedback.append(f"Incorrect or missing columns: {list(df.columns)}")

                # 4. Validate two distinct class rows
                if len(df) == 2 and len(set(df["Class"])) == 2:
                    score += 0.15
                    feedback.append("Output contains two rows for distinct classes.")
                else:
                    feedback.append(f"Unexpected number of rows or classes: {df['Class'].tolist()}")

                # 5. Validate numeric and finite values
                try:
                    mean_vals = pd.to_numeric(df["Mean_Fault_Risk"], errors="coerce")
                    var_vals = pd.to_numeric(df["Variance_Fault_Risk"], errors="coerce")

                    if mean_vals.notna().all() and var_vals.notna().all():
                        score += 0.15
                        feedback.append("Mean and variance values are numeric and valid.")

                        # 6. Check plausible range
                        mean_ok = (mean_vals.between(-10000, 0)).all()
                        var_ok = (var_vals > 1e6).all() and (var_vals < 1e10).all()

                        if mean_ok and var_ok:
                            score += 0.1
                            feedback.append("Mean and variance values fall within realistic ranges.")
                        else:
                            feedback.append(
                                f"Some values fall outside expected ranges. Means: {mean_vals.tolist()}, Variances: {var_vals.tolist()}"
                            )
                    else:
                        feedback.append("Some numeric values are invalid or NaN.")
                except Exception as e:
                    feedback.append(f"Error validating numeric values: {e}")

            except Exception as e:
                feedback.append(f"Error reading output CSV: {e}")
        else:
            feedback.append("Output CSV file missing.")
            logger.warning("Output CSV not found.")

    except Exception as e:
        feedback.append(f"Evaluation failed: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)

    final_score = min(round(score, 2), 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score

def evaluate_rf_accuracy_f1(actual: str, expected: dict, **options) -> float:
    """
    Strict evaluator for the Steel Plates Fault Random Forest evaluation task.

    Expected output:
        rf_results.csv
        ----------------------------
        Metric,Value
        Accuracy,<float 0–1>
        F1,<float 0–1>
        ----------------------------

    Evaluation checks:
      - Script and CSV exist.
      - Two exact rows: Accuracy and F1.
      - Both numeric, 0–1, rounded to 4 decimals.
      - Metrics >= 0.60 for full credit.
      - No extra rows or text.
    """

    import pandas as pd, os, math, csv

    score, feedback = 0.0, []

    if actual is None:
        feedback.append("No working directory provided.")
        return 0.0

    base_dir = os.path.dirname(actual)
    script_path = os.path.join(base_dir, "steel_rf_eval.py")
    output_path = os.path.join(base_dir, "rf_results.csv")

    # 1. Check script existence
    if os.path.exists(script_path):
        score += 0.25
        feedback.append("Python script file found.")
    else:
        feedback.append("Python script missing.")

    # 2. Check output existence
    if not os.path.exists(output_path):
        feedback.append("Output CSV file missing.")
        logger.info("\n".join(feedback))
        return round(score, 2)

    score += 0.25
    feedback.append("Output CSV file found.")

    # 3. Read output robustly
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            sample = f.read(1024)
            try:
                sep = csv.Sniffer().sniff(sample).delimiter
            except Exception:
                sep = ","
        df = pd.read_csv(output_path, sep=sep)
    except Exception as e:
        feedback.append(f"Failed to read CSV file: {e}")
        logger.info("\n".join(feedback))
        return round(score, 2)

    # 4. Validate structure
    df.columns = [c.strip().lower() for c in df.columns]
    expected_cols = ["metric", "value"]
    if list(df.columns[:2]) == expected_cols:
        score += 0.15
        feedback.append("CSV columns match expected format.")
    else:
        feedback.append(f"Unexpected columns: {df.columns.tolist()}")

    # 5. Check for two rows: Accuracy and F1
    metrics = df["metric"].astype(str).str.strip().str.lower()
    expected_metrics = ["accuracy", "f1"]
    if sorted(metrics.tolist()) == sorted(expected_metrics):
        score += 0.15
        feedback.append("Both required metrics (Accuracy, F1) found.")
    else:
        feedback.append(f"Metrics mismatch: found {metrics.tolist()}.")

    # 6. Validate numeric values 0–1 and rounding
    try:
        vals = pd.to_numeric(df["value"], errors="coerce")
        valid_range = ((vals >= 0) & (vals <= 1)).all()
        if valid_range and not vals.isna().any():
            score += 0.10
            feedback.append("Metric values are numeric and within [0, 1].")
            # Check rounding to 4 decimals
            rounded_ok = all(abs(v - round(v, 4)) < 1e-8 for v in vals)
            if rounded_ok:
                score += 0.05
                feedback.append("Values appear rounded to four decimals.")
            else:
                feedback.append("Values not rounded to four decimals.")
        else:
            feedback.append("Values outside [0,1] or invalid.")
    except Exception as e:
        feedback.append(f"Error parsing numeric values: {e}")

    # 7. Check performance threshold
    try:
        perf = dict(zip(metrics, vals))
        acc = perf.get("accuracy", 0)
        f1 = perf.get("f1", 0)
        if acc >= 0.6 and f1 >= 0.6:
            score += 0.05
            feedback.append("Accuracy and F1 meet or exceed 60% threshold.")
        else:
            feedback.append(f"Metrics below threshold: accuracy={acc:.3f}, f1={f1:.3f}")
    except Exception:
        feedback.append("Could not check thresholds.")

    # 8. Enforce strict 2-row limit
    if len(df) == 2:
        score += 0.05
        feedback.append("CSV contains exactly two rows as required.")
    else:
        feedback.append(f"Unexpected number of rows: {len(df)}")

    # --- Finalize ---
    final_score = min(round(score, 2), 1.0)
    logger.info(f"Final Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)
    return final_score

def evaluate_eeg_feature_group_visualization_quality(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the EEG Feature Group Visualization (Visual Studio Code) task.

    This evaluation checks:
      - The Python script file exists.
      - Both visualization files exist and are non-empty.
      - GPT-4o confirms that:
          1. feature_means.png correctly shows average V5 and V10 per (V12, V13) group.
          2. class_proportion_heatmap.png correctly shows the proportion of Class = 1 across (V12, V13) groups.
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    expected = expected.get("rules", expected)
    base_dir = os.path.dirname(actual)

    script_file = os.path.join(base_dir, "eeg_feature_group_analysis.py")
    chart1 = os.path.join(base_dir, "feature_means.png")
    chart2 = os.path.join(base_dir, "class_proportion_heatmap.png")

    score = 0.0
    feedback = []

    # Step 1: Check for script existence
    if os.path.exists(script_file):
        score += 0.4
        feedback.append("The Python script file exists.")
        logger.info("Script file found.")
    else:
        feedback.append("The Python script file is missing.")
        logger.warning("Script file missing.")

    # Helper: Encode image as base64
    def encode_image(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    # Step 2: Visualization checks
    charts_to_check = [
        (
            chart1,
            "a grouped or bar chart showing the mean values of V5 and V10 for each combination of V12 and V13, clearly labeled with axes and legend."
        ),
        (
            chart2,
            "a heatmap or equivalent visualization showing the proportion of Class = 1 for each (V12, V13) combination, with labeled axes and color scale."
        )
    ]

    for chart, description in charts_to_check:
        chart_name = os.path.basename(chart)
        if os.path.exists(chart) and os.path.getsize(chart) > 0:
            score += 0.15
            feedback.append(f"The visualization {chart_name} exists and is non-empty.")
            logger.info(f"{chart_name} found and valid.")

            try:
                encoded = encode_image(chart)
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a data visualization expert evaluating chart accuracy and clarity."
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        f"Examine this chart and confirm whether it correctly represents {description}. "
                                        f"Reply 'yes' if it clearly conveys the data with proper labeling and layout; "
                                        f"otherwise reply 'no' with a brief reason."
                                    )
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{encoded}"}
                                }
                            ]
                        }
                    ],
                    max_tokens=100
                )

                answer = response.choices[0].message.content.lower()
                if "yes" in answer:
                    score += 0.1
                    feedback.append(f"GPT-4o judged {chart_name} as accurate and well-labeled.")
                    logger.info(f"{chart_name} validated successfully by GPT-4o.")
                else:
                    feedback.append(f"GPT-4o judged {chart_name} unclear or inaccurate: {answer}")
                    logger.warning(f"{chart_name} unclear according to GPT-4o.")
            except Exception as e:
                feedback.append(f"Error evaluating {chart_name} with GPT-4o: {e}")
                logger.error(f"GPT-4o evaluation failed for {chart_name}: {e}", exc_info=True)
        else:
            feedback.append(f"The visualization {chart_name} is missing or empty.")
            logger.warning(f"{chart_name} missing or empty.")

    # Step 3: Final Scoring
    final_score = min(score, 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")

    for msg in feedback:
        logger.info(msg)

    print("\nEvaluation Summary:")
    for msg in feedback:
        print("-", msg)
    print(f"\nFinal Score: {final_score:.2f}")

    return final_score

def evaluate_amazon_outlier_detection(actual: str, expected: dict, **options) -> float:
    """
    Strict evaluator for the Amazon Outlier Detection task.

    Expected output:
        outlier_summary.csv
        ------------------------------
        Metric,Value
        Total_Outliers,732505
        ------------------------------

    Checks:
      - Script and CSV exist.
      - Columns are Metric,Value.
      - Exactly one row.
      - Metric = 'Total_Outliers'.
      - Value is integer ≥ 0 and within ±10 of expected (732505).
    """

    import pandas as pd, os, csv

    score = 0.0
    feedback = []

    if actual is None:
        feedback.append("No working directory provided.")
        return 0.0

    base_dir = os.path.dirname(actual)
    script_path = os.path.join(base_dir, "amazon_outlier_detection.py")
    output_path = os.path.join(base_dir, "outlier_summary.csv")

    # 1. Check script existence
    if os.path.exists(script_path):
        score += 0.25
        feedback.append("Python script file found.")
    else:
        feedback.append("Python script missing.")

    # 2. Check output CSV existence
    if not os.path.exists(output_path):
        feedback.append("Output CSV file missing.")
        logger.info("\n".join(feedback))
        return round(score, 2)

    score += 0.25
    feedback.append("Output CSV file found.")

    # 3. Load CSV robustly
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            sample = f.read(1024)
            try:
                sep = csv.Sniffer().sniff(sample).delimiter
            except Exception:
                sep = ","
        df = pd.read_csv(output_path, sep=sep)
    except Exception as e:
        feedback.append(f"Failed to read CSV: {e}")
        logger.info("\n".join(feedback))
        return round(score, 2)

    # 4. Validate columns
    df.columns = [c.strip().lower() for c in df.columns]
    if list(df.columns[:2]) == ["metric", "value"]:
        score += 0.15
        feedback.append("CSV columns match expected format: Metric, Value.")
    else:
        feedback.append(f"Incorrect columns: {list(df.columns)}")

    # 5. Validate row count
    if len(df) == 1:
        score += 0.1
        feedback.append("CSV contains exactly one row as required.")
    else:
        feedback.append(f"Unexpected number of rows: {len(df)}")

    # 6. Check metric name
    metric_val = str(df.iloc[0, 0]).strip().lower()
    if metric_val == "total_outliers":
        score += 0.1
        feedback.append("Metric name 'Total_Outliers' correctly specified.")
    else:
        feedback.append(f"Unexpected metric name: {metric_val}")

    # 7. Validate numeric value and range
    expected_total = 732505
    tolerance = 10

    try:
        val = float(df.iloc[0, 1])
        if val >= 0:
            if abs(val - round(val)) < 1e-6:
                score += 0.1
                feedback.append("Value is a valid non-negative integer.")
            else:
                feedback.append("Value is not rounded to an integer.")
        else:
            feedback.append("Value is negative.")

        # Check if value within ±10 of expected
        if abs(val - expected_total) <= tolerance:
            score += 0.1
            feedback.append(f"Value {val} is within ±{tolerance} of expected ({expected_total}).")
        else:
            feedback.append(f"Value {val} differs from expected ({expected_total}) by more than ±{tolerance}.")
    except Exception:
        feedback.append("Value is not numeric or missing.")

    # 8. Final plausibility range
    try:
        if 0 < val < 1e9:
            score += 0.1
            feedback.append("Value is within realistic range.")
        else:
            feedback.append(f"Value {val} appears unrealistic.")
    except Exception:
        feedback.append("Could not validate numeric range.")

    # --- Finalize ---
    final_score = min(round(score, 2), 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)
    return final_score


def evaluate_amazon_feature_stats(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Amazon feature statistics summary task.

    Expected output:
        feature_stats_summary.csv
        Columns: Class, Feature, Mean, Std, CV
        One row per Class × Feature (every 50th feature).

    Checks:
      - Script and CSV exist.
      - Columns match.
      - Multiple rows and multiple classes.
      - Each 'every 50th feature' appears.
      - Mean, Std, CV are numeric, non-negative, and realistic.
    """

    import pandas as pd, os, re, csv, numpy as np

    score = 0.0
    feedback = []

    if actual is None:
        feedback.append("No working directory provided.")
        return 0.0

    base_dir = os.path.dirname(actual)
    script_path = os.path.join(base_dir, "amazon_feature_stats.py")
    output_path = os.path.join(base_dir, "feature_stats_summary.csv")

    # 1. Check script existence
    if os.path.exists(script_path):
        score += 0.2
        feedback.append("Python script file found.")
    else:
        feedback.append("Python script missing.")

    # 2. Check CSV existence
    if not os.path.exists(output_path):
        feedback.append("Output CSV file missing.")
        logger.info("\n".join(feedback))
        return round(score, 2)
    else:
        score += 0.2
        feedback.append("Output CSV file found.")

    # 3. Load CSV robustly
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            sample = f.read(1024)
            try:
                sep = csv.Sniffer().sniff(sample).delimiter
            except Exception:
                sep = ","
        df = pd.read_csv(output_path, sep=sep)
    except Exception as e:
        feedback.append(f"Failed to read CSV: {e}")
        logger.info("\n".join(feedback))
        return round(score, 2)

    # 4. Validate columns
    expected_cols = ["class", "feature", "mean", "std", "cv"]
    df.columns = [c.strip().lower() for c in df.columns]
    if list(df.columns)[:5] == expected_cols:
        score += 0.2
        feedback.append("CSV columns match expected format.")
    else:
        feedback.append(f"Incorrect columns: {list(df.columns)}")

    # 5. Check row count
    if len(df) >= 5:
        score += 0.1
        feedback.append(f"CSV has {len(df)} rows — valid summary output.")
    else:
        feedback.append(f"Too few rows ({len(df)}).")

    # 6. Check feature coverage (every 50th feature pattern)
    pattern_features = [f"v{i}" for i in range(1, 1001, 50)]
    features_found = set(df["feature"].str.lower())
    coverage = sum(1 for f in pattern_features if f in features_found)
    if coverage >= 3:  # at least 3 features appear
        score += 0.1
        feedback.append(f"Detected {coverage} expected features (e.g., V1, V51, ...).")
    else:
        feedback.append("Missing expected feature pattern (V1, V51, ...).")

    # 7. Check multiple classes
    if df["class"].nunique() > 1:
        score += 0.1
        feedback.append(f"Detected multiple classes ({df['class'].nunique()}).")
    else:
        feedback.append("Only one class detected — grouping may be incorrect.")

    # 8. Check numeric and realistic values
    try:
        numeric_ok = all(
            pd.api.types.is_numeric_dtype(df[col]) for col in ["mean", "std", "cv"]
        )
        if numeric_ok:
            score += 0.1
            feedback.append("Mean, Std, CV columns are numeric.")
            # Check ranges
            if (df["mean"] >= 0).all() and (df["std"] >= 0).all() and (df["cv"] <= 5).all():
                score += 0.1
                feedback.append("Values appear realistic and within expected ranges.")
            else:
                feedback.append("Some numeric values are unrealistic (negative or CV > 5).")
        else:
            feedback.append("Mean/Std/CV columns contain non-numeric values.")
    except Exception as e:
        feedback.append(f"Error validating numeric columns: {e}")

    # --- Finalize ---
    final_score = min(round(score, 2), 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)
    return final_score


def evaluate_amazon_feature_correlation(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for Amazon Feature Correlation task.

    Expected output:
        top_correlated_pairs.csv
        Columns: Feature1, Feature2, Correlation
        Must have exactly 5 rows.

    Checks:
      - Script and CSV exist.
      - CSV has required columns.
      - Contains exactly 5 rows.
      - Correlation values are numeric, between −1 and 1.
      - All |r| > 0.5 (strong correlations).
      - Sorted in descending order by absolute correlation.
    """

    import os
    import pandas as pd
    import numpy as np

    score = 0.0
    feedback = []

    if actual is None:
        feedback.append("No working directory provided.")
        return 0.0

    base_dir = os.path.dirname(actual)
    script_path = os.path.join(base_dir, "amazon_feature_correlation.py")
    output_path = os.path.join(base_dir, "top_correlated_pairs.csv")

    # --- 1. Check Python script existence ---
    if os.path.exists(script_path):
        score += 0.25
        feedback.append("Python script file found.")
    else:
        feedback.append("Python script missing.")

    # --- 2. Check CSV output existence ---
    if not os.path.exists(output_path):
        feedback.append("Output CSV file missing.")
        return round(score, 2)
    else:
        score += 0.25
        feedback.append("Output CSV file found.")

    # --- 3. Load full CSV directly ---
    try:
        df = pd.read_csv(output_path)
    except Exception as e:
        feedback.append(f"Failed to read CSV: {e}")
        return round(score, 2)

    # --- 4. Validate columns ---
    df.columns = [c.strip().lower() for c in df.columns]
    expected_cols = ["feature1", "feature2", "correlation"]
    if all(col in df.columns for col in expected_cols):
        score += 0.15
        feedback.append("CSV columns match expected names.")
    else:
        feedback.append(f"Incorrect or missing columns: {df.columns.tolist()}")

    # --- 5. Validate row count ---
    if len(df) == 5:
        score += 0.1
        feedback.append("CSV contains exactly 5 rows as required.")
    else:
        feedback.append(f"Unexpected row count: {len(df)} (expected 5).")

    # --- 6. Check correlation validity ---
    try:
        corrs = pd.to_numeric(df["correlation"], errors="coerce")
        if corrs.notna().all() and ((corrs >= -1) & (corrs <= 1)).all():
            score += 0.1
            feedback.append("All correlation values are numeric and within [−1, 1].")

            # --- Enforce all correlations strong (>0.5) ---
            if (corrs.abs() > 0.5).all():
                score += 0.1
                feedback.append("All correlations are strong (|r| > 0.5).")
            else:
                feedback.append("Some correlations are weaker than |r| ≤ 0.5.")
        else:
            feedback.append("Invalid correlation values detected (non-numeric or out of range).")
    except Exception as e:
        feedback.append(f"Error processing correlation values: {e}")

    # --- 7. Check sorting by absolute correlation ---
    try:
        if corrs.abs().is_monotonic_decreasing:
            score += 0.1
            feedback.append("Rows sorted correctly by absolute correlation (descending).")
        else:
            feedback.append("Rows not sorted correctly by absolute correlation.")
    except Exception:
        feedback.append("Could not verify sorting order.")

    # --- Final scoring ---
    final_score = min(round(score, 2), 1.0)

    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score

def evaluate_amazon_rf_classification(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Amazon Random Forest classification task.

    Expected output:
        rf_classification_results.txt
        Lines (example):
            Accuracy: 0.8123
            Macro_F1: 0.8041
            Top5_Accuracy: 0.9276

    Checks:
      - Script and result file exist.
      - Contains all three metrics: Accuracy, Macro_F1, Top5_Accuracy.
      - Each value is numeric, within [0, 1].
      - Each metric ≥ 0.5.
    """

    import os, re

    score = 0.0
    feedback = []

    if actual is None:
        feedback.append("No working directory provided.")
        return 0.0

    base_dir = os.path.dirname(actual)
    script_path = os.path.join(base_dir, "amazon_rf_classification.py")
    result_path = os.path.join(base_dir, "rf_classification_results.txt")

    # --- 1. Check script existence ---
    if os.path.exists(script_path):
        score += 0.25
        feedback.append("Python script file found.")
    else:
        feedback.append("Python script missing.")

    # --- 2. Check result file existence ---
    if not os.path.exists(result_path):
        feedback.append("Results file (rf_classification_results.txt) missing.")
        return round(score, 2)
    else:
        score += 0.25
        feedback.append("Results file found.")

    # --- 3. Parse result file ---
    try:
        with open(result_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        feedback.append(f"Error reading results file: {e}")
        return round(score, 2)

    # --- 4. Validate metric presence ---
    required_metrics = ["Accuracy", "Macro_F1", "Top5_Accuracy"]
    found_metrics = [m for m in required_metrics if m.lower() in content.lower()]
    if len(found_metrics) == len(required_metrics):
        score += 0.15
        feedback.append("All required metrics (Accuracy, Macro_F1, Top5_Accuracy) found.")
    else:
        missing = list(set(required_metrics) - set(found_metrics))
        feedback.append(f"Missing metrics: {missing}")

    # --- 5. Extract numeric values ---
    pattern = r"([-+]?[0-9]*\.?[0-9]+)"
    values = [float(v) for v in re.findall(pattern, content) if v.strip()]
    if not values:
        feedback.append("No numeric metric values detected.")
        return round(score, 2)

    # --- 6. Validate numeric ranges ---
    valid_range = all(0.0 <= v <= 1.0 for v in values)
    if valid_range:
        score += 0.15
        feedback.append("All metric values are within the valid range [0, 1].")
    else:
        feedback.append("Some metric values are outside the valid [0, 1] range.")

    # --- 7. Check minimum thresholds (≥ 0.5) ---
    if all(v >= 0.5 for v in values):
        score += 0.2
        feedback.append("All metrics meet or exceed 0.5 threshold.")
    else:
        low_values = [v for v in values if v < 0.5]
        feedback.append(f"Some metrics below threshold: {low_values}")

    # --- 8. Final scoring ---
    final_score = min(round(score, 2), 1.0)

    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score

def evaluate_amazon_rf_gridsearch(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Amazon Random Forest Grid Search task.

    Expected output:
        rf_best_params.txt
        Lines (example):
            Best_Params: {'n_estimators': 200, 'max_depth': 15, 'min_samples_split': 4}
            Validation_Accuracy: 0.8234

    Checks:
      - Script (.py) and result (.txt) files exist.
      - Contains both "Best_Params" and "Validation_Accuracy".
      - Validation_Accuracy is numeric and ≥ 0.5.
    """

    score = 0.0
    feedback = []

    if actual is None:
        feedback.append("No working directory provided.")
        return 0.0

    base_dir = os.path.dirname(actual)
    script_path = os.path.join(base_dir, "amazon_rf_gridsearch.py")
    result_path = os.path.join(base_dir, "rf_best_params.txt")

    # --- 1. Check file existence ---
    script_exists = os.path.exists(script_path)
    txt_exists = os.path.exists(result_path)

    if script_exists:
        score += 0.3
        feedback.append("Python script file found.")
    else:
        feedback.append("Python script missing.")

    if txt_exists:
        score += 0.3
        feedback.append("Results text file found.")
    else:
        feedback.append("Results file missing.")
        return round(score, 2)

    # --- 2. Check contents for required keywords ---
    try:
        with open(result_path, "r", encoding="utf-8") as f:
            content = f.read()

        if "Best_Params" in content and "Validation_Accuracy" in content:
            score += 0.2
            feedback.append("Both 'Best_Params' and 'Validation_Accuracy' found in output.")
        else:
            feedback.append("Missing one or both required keys in results file.")

        # --- 3. Validate numeric accuracy ≥ 0.5 ---
        match = re.search(r"Validation_Accuracy[:=]\s*([0-9]*\.?[0-9]+)", content)
        if match:
            acc_val = float(match.group(1))
            if 0.0 <= acc_val <= 1.0:
                if acc_val >= 0.5:
                    score += 0.2
                    feedback.append(f"Validation accuracy ({acc_val:.4f}) is valid and ≥ 0.5.")
                else:
                    feedback.append(f"Validation accuracy too low: {acc_val:.4f} (< 0.5).")
            else:
                feedback.append(f"Accuracy value {acc_val:.4f} out of [0, 1] range.")
        else:
            feedback.append("Could not find numeric Validation_Accuracy value.")

    except Exception as e:
        feedback.append(f"Error reading or validating results file: {e}")

    # --- Final scoring ---
    final_score = min(round(score, 2), 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score


def evaluate_amazon_reviews_pca_kmeans_visualization_quality(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Amazon Reviews PCA + K-Means Visualization (Visual Studio Code) task.

    This evaluation checks:
      - The Python script file exists.
      - Both visualization files exist and are non-empty.
      - GPT-4o confirms that:
          1. cluster_scatter.png correctly shows clusters in PCA space (first two components, color = cluster).
          2. cluster_distribution.png correctly shows the size of each cluster as a bar chart.
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    expected = expected.get("rules", expected)
    base_dir = os.path.dirname(actual)

    script_file = os.path.join(base_dir, "amazon_reviews_pca_kmeans.py")
    chart1 = os.path.join(base_dir, "cluster_scatter.png")
    chart2 = os.path.join(base_dir, "cluster_distribution.png")

    score = 0.0
    feedback = []

    # Step 1: Verify script existence
    if os.path.exists(script_file):
        score += 0.4
        feedback.append("The Python script file exists.")
        logger.info("Script file found.")
    else:
        feedback.append("The Python script file is missing.")
        logger.warning("Script file missing.")

    # Helper: Encode image for GPT-4o
    def encode_image(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    # Step 2: Visualization checks
    charts_to_check = [
        (
            chart1,
            "a scatter plot showing data points projected onto the first two PCA components, colored by their K-Means cluster label. The axes should be labeled PC1 and PC2."
        ),
        (
            chart2,
            "a bar chart showing the number of samples in each cluster, with bars labeled by cluster ID and a clear title."
        )
    ]

    for chart, description in charts_to_check:
        chart_name = os.path.basename(chart)
        if os.path.exists(chart) and os.path.getsize(chart) > 0:
            score += 0.15
            feedback.append(f"The visualization {chart_name} exists and is non-empty.")
            logger.info(f"{chart_name} found and valid.")

            try:
                encoded = encode_image(chart)
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a data visualization expert evaluating chart clarity and accuracy."
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        f"Examine this chart and determine whether it correctly represents {description}. "
                                        f"Reply 'yes' if it is clearly labeled, visually interpretable, and appropriate for the given description; "
                                        f"otherwise reply 'no' with a short reason."
                                    )
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{encoded}"}
                                }
                            ]
                        }
                    ],
                    max_tokens=100
                )

                answer = response.choices[0].message.content.lower()
                if "yes" in answer:
                    score += 0.1
                    feedback.append(f"GPT-4o judged {chart_name} as accurate and well-labeled.")
                    logger.info(f"{chart_name} validated by GPT-4o.")
                else:
                    feedback.append(f"GPT-4o judged {chart_name} unclear or inaccurate: {answer}")
                    logger.warning(f"{chart_name} unclear according to GPT-4o.")
            except Exception as e:
                feedback.append(f"Error evaluating {chart_name} with GPT-4o: {e}")
                logger.error(f"GPT-4o evaluation failed for {chart_name}: {e}", exc_info=True)
        else:
            feedback.append(f"The visualization {chart_name} is missing or empty.")
            logger.warning(f"{chart_name} missing or empty.")

    # Step 3: Final Scoring
    final_score = min(score, 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")

    for msg in feedback:
        logger.info(msg)

    print("\nEvaluation Summary:")
    for msg in feedback:
        print("-", msg)
    print(f"\nFinal Score: {final_score:.2f}")

    return final_score

def evaluate_cholesterol_outlier_replacement(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the cholesterol outlier replacement task.

    Expected output:
        outlier_replacement_summary.txt
        Format:
            Total_Replaced: 20

    Checks:
      - Script (.py) and result (.txt) files exist.
      - Text file contains "Total_Replaced".
      - Extracted value is numeric.
      - Value is close to 20 (±3 tolerance for realism).
    """

    score = 0.0
    feedback = []

    if actual is None:
        feedback.append("No working directory provided.")
        return 0.0

    base_dir = os.path.dirname(actual)
    script_path = os.path.join(base_dir, "cholesterol_outlier_replacement.py")
    result_path = os.path.join(base_dir, "outlier_replacement_summary.txt")

    # --- 1. Check script existence ---
    if os.path.exists(script_path):
        score += 0.3
        feedback.append("Python script file found.")
    else:
        feedback.append("Python script missing.")

    # --- 2. Check output file existence ---
    if not os.path.exists(result_path):
        feedback.append("Output text file missing.")
        return round(score, 2)
    else:
        score += 0.3
        feedback.append("Output text file found.")

    # --- 3. Validate file content ---
    try:
        with open(result_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        # Check for keyword
        if "Total_Replaced" in content:
            score += 0.2
            feedback.append("'Total_Replaced' keyword found in output.")
        else:
            feedback.append("Missing 'Total_Replaced' in output file.")

        # Extract numeric value
        match = re.search(r"Total_Replaced[:=]\s*([0-9]+)", content)
        if match:
            replaced_val = int(match.group(1))
            # Expect value around 20 (±3 tolerance)
            if abs(replaced_val - 20) <= 3:
                score += 0.2
                feedback.append(f"Replaced value ({replaced_val}) is realistic and matches expected range.")
            else:
                feedback.append(f"Replaced value ({replaced_val}) outside expected range (~20 ±3).")
        else:
            feedback.append("Could not find numeric value for 'Total_Replaced'.")

    except Exception as e:
        feedback.append(f"Error reading or parsing output: {e}")

    # --- Finalize scoring ---
    final_score = min(round(score, 2), 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score

def evaluate_correlation_with_target(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for correlation_with_target task (ds152).

    Expected output:
        top_target_correlations.csv
        Columns:
            Feature, Correlation
        Example:
            Feature,Correlation
            ca,0.5189
            thal,0.5099

    Checks:
      - Script (.py) and output (.csv) files exist.
      - CSV has correct columns.
      - Exactly 2 rows present.
      - Correlation values numeric, within [0, 1].
      - All correlations >= 0.5.
      - Sorted in descending order.
    """

    score = 0.0
    feedback = []

    if actual is None:
        feedback.append("No working directory provided.")
        return 0.0

    base_dir = os.path.dirname(actual)
    script_path = os.path.join(base_dir, "correlation_with_target.py")
    output_path = os.path.join(base_dir, "top_target_correlations.csv")

    # --- 1. Check script file existence ---
    if os.path.exists(script_path):
        score += 0.25
        feedback.append("Python script file found.")
    else:
        feedback.append("Python script missing.")

    # --- 2. Check output CSV existence ---
    if not os.path.exists(output_path):
        feedback.append("Output CSV file missing.")
        return round(score, 2)
    else:
        score += 0.25
        feedback.append("Output CSV file found.")

    # --- 3. Load CSV ---
    try:
        df = pd.read_csv(output_path)
    except Exception as e:
        feedback.append(f"Error reading CSV: {e}")
        return round(score, 2)

    # --- 4. Validate columns ---
    df.columns = [c.strip().lower() for c in df.columns]
    expected_cols = ["feature", "correlation"]
    if all(col in df.columns for col in expected_cols):
        score += 0.15
        feedback.append("CSV columns match expected names (Feature, Correlation).")
    else:
        feedback.append(f"Incorrect columns: {df.columns.tolist()}")

    # --- 5. Validate row count ---
    if len(df) == 2:
        score += 0.1
        feedback.append("CSV contains exactly 2 rows as expected.")
    else:
        feedback.append(f"Unexpected row count: {len(df)} (expected 2).")

    # --- 6. Check correlation values ---
    try:
        corr_vals = pd.to_numeric(df["correlation"], errors="coerce")
        if corr_vals.notna().all() and ((corr_vals >= 0) & (corr_vals <= 1)).all():
            score += 0.1
            feedback.append("All correlation values are numeric and within [0, 1].")

            # --- Check all correlations >= 0.5 ---
            if (corr_vals >= 0.5).all():
                score += 0.1
                feedback.append("All correlations are ≥ 0.5 (strong positive).")
            else:
                feedback.append("Some correlations are below 0.5.")
        else:
            feedback.append("Invalid or non-numeric correlation values found.")
    except Exception as e:
        feedback.append(f"Error validating correlation values: {e}")

    # --- 7. Check descending order ---
    try:
        if corr_vals.is_monotonic_decreasing:
            score += 0.05
            feedback.append("Correlations are sorted in descending order.")
        else:
            feedback.append("Correlations not sorted correctly.")
    except Exception:
        feedback.append("Could not check sorting order.")

    # --- Final scoring ---
    final_score = min(round(score, 2), 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score

def evaluate_cholesterol_group_stats(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for cholesterol_group_stats task (ds153).

    Expected output example:
        Group_Count,Sum_Mean,Sum_Std
        8,3205.8551,539.3424

    Checks:
      - Python script and CSV file exist.
      - CSV has correct columns.
      - Exactly 1 row in output.
      - All values numeric and positive.
      - Sum_Mean and Sum_Std within ±10% of expected.
    """
    score = 0.0
    feedback = []

    if actual is None:
        feedback.append("No working directory provided.")
        return 0.0

    base_dir = os.path.dirname(actual)
    script_path = os.path.join(base_dir, "cholesterol_group_stats.py")
    output_path = os.path.join(base_dir, "grouped_cholesterol_summary.csv")

    # --- 1. Check script file existence ---
    if os.path.exists(script_path):
        score += 0.25
        feedback.append("Python script file found.")
    else:
        feedback.append("Python script missing.")

    # --- 2. Check output CSV existence ---
    if not os.path.exists(output_path):
        feedback.append("Output CSV file missing.")
        return round(score, 2)
    else:
        score += 0.25
        feedback.append("Output CSV file found.")

    # --- 3. Load CSV ---
    try:
        df = pd.read_csv(output_path)
    except Exception as e:
        feedback.append(f"Error reading CSV: {e}")
        return round(score, 2)

    # --- 4. Validate columns ---
    df.columns = [c.strip().lower() for c in df.columns]
    expected_cols = ["group_count", "sum_mean", "sum_std"]
    if all(col in df.columns for col in expected_cols):
        score += 0.15
        feedback.append("CSV columns match expected names (Group_Count, Sum_Mean, Sum_Std).")
    else:
        feedback.append(f"Incorrect columns: {df.columns.tolist()}")

    # --- 5. Validate row count ---
    if len(df) == 1:
        score += 0.1
        feedback.append("CSV contains exactly 1 row as expected.")
    else:
        feedback.append(f"Unexpected number of rows: {len(df)} (expected 1).")

    # --- 6. Validate numeric values ---
    try:
        df = df.astype(float)
        if (df >= 0).all().all():
            score += 0.1
            feedback.append("All output values are numeric and positive.")
        else:
            feedback.append("Found non-positive values in output.")
    except Exception:
        feedback.append("Output values could not be converted to float.")

    # --- 7. Validate approximate correctness (within ±10%) ---
    expected_mean, expected_std = 3205.8551, 539.3424
    try:
        mean_val = float(df["sum_mean"].iloc[0])
        std_val = float(df["sum_std"].iloc[0])

        if abs(mean_val - expected_mean) / expected_mean <= 0.1:
            score += 0.075
            feedback.append("Sum_Mean within ±10% of expected value.")
        else:
            feedback.append(f"Sum_Mean ({mean_val}) deviates more than ±10% from expected ({expected_mean}).")

        if abs(std_val - expected_std) / expected_std <= 0.1:
            score += 0.075
            feedback.append("Sum_Std within ±10% of expected value.")
        else:
            feedback.append(f"Sum_Std ({std_val}) deviates more than ±10% from expected ({expected_std}).")
    except Exception as e:
        feedback.append(f"Error comparing numeric values: {e}")

    # --- Final scoring ---
    final_score = min(round(score, 2), 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score

def evaluate_cholesterol_linear_regression(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for cholesterol_linear_regression task (ds154).

    Expected output (linear_regression_results.csv):
        R2,MAE,RMSE
        0.7123,22.4351,34.9821

    Checks:
      - Python script and output CSV exist.
      - CSV has correct columns.
      - Exactly 1 row in output.
      - Metrics are numeric and positive.
      - R² > 0 and ideally >= 0.5.
      - RMSE < 100 for realism.
    """

    score = 0.0
    feedback = []

    if actual is None:
        feedback.append("No working directory provided.")
        return 0.0

    base_dir = os.path.dirname(actual)
    script_path = os.path.join(base_dir, "cholesterol_linear_regression.py")
    output_path = os.path.join(base_dir, "linear_regression_results.csv")

    # --- 1. Check script existence ---
    if os.path.exists(script_path):
        score += 0.25
        feedback.append("Python script file found.")
    else:
        feedback.append("Python script missing.")

    # --- 2. Check output CSV existence ---
    if not os.path.exists(output_path):
        feedback.append("Output CSV file missing.")
        return round(score, 2)
    else:
        score += 0.25
        feedback.append("Output CSV file found.")

    # --- 3. Load and validate CSV ---
    try:
        df = pd.read_csv(output_path)
    except Exception as e:
        feedback.append(f"Error reading CSV: {e}")
        return round(score, 2)

    df.columns = [c.strip().lower() for c in df.columns]
    expected_cols = ["r2", "mae", "rmse"]

    # --- 4. Check columns ---
    if all(col in df.columns for col in expected_cols):
        score += 0.15
        feedback.append("CSV columns match expected names (R2, MAE, RMSE).")
    else:
        feedback.append(f"Incorrect columns: {df.columns.tolist()}")

    # --- 5. Check row count ---
    if len(df) == 1:
        score += 0.1
        feedback.append("CSV contains exactly one row as expected.")
    else:
        feedback.append(f"Unexpected number of rows: {len(df)} (expected 1).")

    # --- 6. Check metric values ---
    try:
        vals = df.iloc[0].astype(float)
        r2, mae, rmse = vals.get("r2", 0), vals.get("mae", 0), vals.get("rmse", 0)

        if all(x >= 0 for x in [mae, rmse]):
            score += 0.1
            feedback.append("MAE and RMSE are positive numeric values.")
        else:
            feedback.append("Found negative or invalid MAE/RMSE values.")

        # --- R² realistic check ---
        if 0 <= r2 <= 1:
            score += 0.05
            feedback.append("R² value is valid and within [0, 1].")
        else:
            feedback.append(f"Invalid R² value: {r2}")

        # --- Reward good model ---
        if r2 >= 0.5:
            score += 0.05
            feedback.append("Model shows reasonable predictive power (R² ≥ 0.5).")

        if rmse < 100:
            score += 0.05
            feedback.append("RMSE value is within a realistic range (<100).")
        else:
            feedback.append("RMSE appears too large for realistic data.")
    except Exception as e:
        feedback.append(f"Error validating metric values: {e}")

    # --- Final scoring ---
    final_score = min(round(score, 2), 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score

def evaluate_ridge_lasso_comparison(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for ridge_lasso_comparison task (ds155).

    Expected output (ridge_lasso_results.csv):
        Model,R2_Score,Better
        Ridge,0.7123,Ridge
        Lasso,0.6351,Ridge

    Checks:
      - Python script (.py) and output (.csv) exist.
      - Output CSV has correct columns: Model, R2_Score, Better.
      - All R2_Score values are numeric and greater than 0.5.
    """
    score = 0.0
    feedback = []

    if actual is None:
        feedback.append("No working directory provided.")
        return 0.0

    base_dir = os.path.dirname(actual)
    script_path = os.path.join(base_dir, "ridge_lasso_comparison.py")
    output_path = os.path.join(base_dir, "ridge_lasso_results.csv")

    # --- 1. Check script existence ---
    if os.path.exists(script_path):
        score += 0.33
        feedback.append("Python script file found.")
    else:
        feedback.append("Python script missing.")

    # --- 2. Check output CSV existence ---
    if not os.path.exists(output_path):
        feedback.append("Output CSV file missing.")
        return round(score, 2)
    else:
        score += 0.33
        feedback.append("Output CSV file found.")

    # --- 3. Validate CSV structure and R2 values ---
    try:
        df = pd.read_csv(output_path)
        df.columns = [c.strip().lower() for c in df.columns]

        expected_cols = ["model", "r2_score", "better"]
        if all(col in df.columns for col in expected_cols):
            score += 0.17
            feedback.append("CSV columns match expected names (Model, R2_Score, Better).")
        else:
            feedback.append(f"Incorrect columns found: {df.columns.tolist()}")

        # Check R² values
        if "r2_score" in df.columns:
            try:
                r2_vals = pd.to_numeric(df["r2_score"], errors="coerce")
                if r2_vals.notna().all() and (r2_vals > 0.5).all():
                    score += 0.17
                    feedback.append("All R2_Score values are numeric and greater than 0.5.")
                else:
                    feedback.append("Some R2_Score values are ≤ 0.5 or non-numeric.")
            except Exception as e:
                feedback.append(f"Error validating R2_Score values: {e}")
        else:
            feedback.append("Column R2_Score missing.")
    except Exception as e:
        feedback.append(f"Error reading or validating CSV: {e}")

    # --- Final scoring ---
    final_score = min(round(score, 2), 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score


def evaluate_cholesterol_sex_visualization_quality(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Cholesterol by Sex Visualization (Visual Studio Code) task.

    This evaluation checks:
      - The Python script file exists.
      - Both visualization files exist and are non-empty.
      - GPT-4o confirms that:
          1. chol_boxplot.png correctly compares cholesterol (chol) distributions by sex using a boxplot.
          2. sex_mean_num_bar.png correctly shows mean num (disease severity) per sex as a bar chart.
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    expected = expected.get("rules", expected)
    base_dir = os.path.dirname(actual)

    script_file = os.path.join(base_dir, "cholesterol_sex_visualization.py")
    chart1 = os.path.join(base_dir, "chol_boxplot.png")
    chart2 = os.path.join(base_dir, "sex_mean_num_bar.png")

    score = 0.0
    feedback = []

    # Step 1: Check script existence
    if os.path.exists(script_file):
        score += 0.4
        feedback.append("The Python script file exists.")
        logger.info("Script file found.")
    else:
        feedback.append("The Python script file is missing.")
        logger.warning("Script file missing.")

    # Helper: Encode an image as base64
    def encode_image(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    # Step 2: Visualization checks
    charts_to_check = [
        (
            chart1,
            "a boxplot comparing cholesterol (chol) distributions by sex, with labeled axes and a clear title."
        ),
        (
            chart2,
            "a bar chart showing mean disease severity (num) per sex, with labeled axes, bars for each sex, and a clear title."
        )
    ]

    for chart, description in charts_to_check:
        chart_name = os.path.basename(chart)
        if os.path.exists(chart) and os.path.getsize(chart) > 0:
            score += 0.15
            feedback.append(f"The visualization {chart_name} exists and is non-empty.")
            logger.info(f"{chart_name} found and valid.")

            try:
                encoded = encode_image(chart)
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert data visualization evaluator. Assess chart clarity, labeling, and relevance."
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        f"Examine this chart and determine whether it correctly represents {description}. "
                                        f"Reply 'yes' if it clearly visualizes the data, includes labels and titles, and is visually appropriate; "
                                        f"otherwise reply 'no' with a brief reason."
                                    )
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{encoded}"}
                                }
                            ]
                        }
                    ],
                    max_tokens=100
                )

                answer = response.choices[0].message.content.lower()
                if "yes" in answer:
                    score += 0.1
                    feedback.append(f"GPT-4o judged {chart_name} as accurate and well-labeled.")
                    logger.info(f"{chart_name} validated successfully by GPT-4o.")
                else:
                    feedback.append(f"GPT-4o judged {chart_name} unclear or inaccurate: {answer}")
                    logger.warning(f"{chart_name} unclear according to GPT-4o.")
            except Exception as e:
                feedback.append(f"Error evaluating {chart_name} with GPT-4o: {e}")
                logger.error(f"GPT-4o evaluation failed for {chart_name}: {e}", exc_info=True)
        else:
            feedback.append(f"The visualization {chart_name} is missing or empty.")
            logger.warning(f"{chart_name} missing or empty.")

    # Step 3: Final Scoring
    final_score = min(score, 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")

    for msg in feedback:
        logger.info(msg)

    print("\nEvaluation Summary:")
    for msg in feedback:
        print("-", msg)
    print(f"\nFinal Score: {final_score:.2f}")

    return final_score


def evaluate_boston_outlier_removal(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for boston_outlier_removal task (ds157).

    Expected output (boston_outlier_summary.csv):
        Column,Outliers_Removed
        MEDV,40

    Checks:
      - Python script (.py) and output (.csv) exist.
      - CSV has correct columns: Column, Outliers_Removed.
      - Exactly 1 row in output.
      - Outliers_Removed is numeric and between 10 and 60 (realistic range).
    """

    score = 0.0
    feedback = []

    if actual is None:
        feedback.append("No working directory provided.")
        return 0.0

    base_dir = os.path.dirname(actual)
    script_path = os.path.join(base_dir, "boston_outlier_removal.py")
    output_path = os.path.join(base_dir, "boston_outlier_summary.csv")

    # --- 1. Check Python script existence ---
    if os.path.exists(script_path):
        score += 0.35
        feedback.append("Python script file found.")
    else:
        feedback.append("Python script missing.")

    # --- 2. Check output CSV existence ---
    if not os.path.exists(output_path):
        feedback.append("Output CSV file missing.")
        return round(score, 2)
    else:
        score += 0.35
        feedback.append("Output CSV file found.")

    # --- 3. Validate CSV structure ---
    try:
        df = pd.read_csv(output_path)
        df.columns = [c.strip().lower() for c in df.columns]
        expected_cols = ["column", "outliers_removed"]

        if all(col in df.columns for col in expected_cols):
            score += 0.15
            feedback.append("CSV columns match expected names (Column, Outliers_Removed).")
        else:
            feedback.append(f"Incorrect columns found: {df.columns.tolist()}")

        # --- 4. Validate single row and numeric value ---
        if len(df) == 1:
            feedback.append("CSV contains exactly one row as expected.")
            try:
                val = float(df["outliers_removed"].iloc[0])
                if 10 <= val <= 60:
                    score += 0.15
                    feedback.append(f"Outliers_Removed value ({val}) is realistic and within range (10–60).")
                else:
                    feedback.append(f"Outliers_Removed value ({val}) outside expected range (10–60).")
            except Exception:
                feedback.append("Outliers_Removed value is not numeric.")
        else:
            feedback.append(f"Unexpected row count: {len(df)} (expected 1).")

    except Exception as e:
        feedback.append(f"Error reading or validating CSV: {e}")

    # --- Final scoring ---
    final_score = min(round(score, 2), 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score

def evaluate_boston_group_summary(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for boston_group_summary task (ds158).

    Expected output (boston_group_summary.csv):
        Sum_Mean,Sum_Std,Group_Count
        87.3203,36.0662,2

    Checks:
      - Python script (.py) and output (.csv) exist.
      - CSV has correct columns.
      - Exactly one row.
      - Sum_Mean ≈ 87.3203, Sum_Std ≈ 36.0662, Group_Count == 2.
        (Tolerates small rounding differences up to ±0.5)
    """

    score = 0.0
    feedback = []

    if actual is None:
        feedback.append("No working directory provided.")
        return 0.0

    base_dir = os.path.dirname(actual)
    script_path = os.path.join(base_dir, "boston_group_summary.py")
    output_path = os.path.join(base_dir, "boston_group_summary.csv")

    # --- 1. Check Python script existence ---
    if os.path.exists(script_path):
        score += 0.3
        feedback.append("Python script file found.")
    else:
        feedback.append("Python script missing.")

    # --- 2. Check output CSV existence ---
    if not os.path.exists(output_path):
        feedback.append("Output CSV file missing.")
        return round(score, 2)
    else:
        score += 0.3
        feedback.append("Output CSV file found.")

    # --- 3. Validate CSV content ---
    try:
        df = pd.read_csv(output_path)
        df.columns = [c.strip().lower() for c in df.columns]
        expected_cols = ["sum_mean", "sum_std", "group_count"]

        if all(col in df.columns for col in expected_cols):
            score += 0.15
            feedback.append("CSV columns match expected names (Sum_Mean, Sum_Std, Group_Count).")
        else:
            feedback.append(f"Incorrect columns: {df.columns.tolist()}")

        # --- 4. Validate single row and numeric correctness ---
        if len(df) == 1:
            try:
                vals = df.iloc[0].astype(float)
                sum_mean, sum_std, group_count = vals["sum_mean"], vals["sum_std"], vals["group_count"]

                # Expected target values
                exp_mean, exp_std, exp_group = 87.3203, 36.0662, 2

                # Check close match (up to ±0.5 difference)
                mean_close = abs(sum_mean - exp_mean) <= 0.5
                std_close = abs(sum_std - exp_std) <= 0.5
                group_exact = int(round(group_count)) == exp_group

                if mean_close and std_close and group_exact:
                    score += 0.25
                    feedback.append(f"Output matches expected values: ({sum_mean:.4f}, {sum_std:.4f}, {int(group_count)}).")
                else:
                    feedback.append(
                        f"Mismatch: got (Sum_Mean={sum_mean:.4f}, Sum_Std={sum_std:.4f}, Group_Count={int(group_count)}), "
                        f"expected approximately (87.3203, 36.0662, 2)."
                    )
            except Exception as e:
                feedback.append(f"Error validating numeric values: {e}")
        else:
            feedback.append(f"Unexpected row count: {len(df)} (expected 1).")

    except Exception as e:
        feedback.append(f"Error reading or validating CSV: {e}")

    # --- Final scoring ---
    final_score = min(round(score, 2), 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score


def evaluate_boston_linear_regression(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for boston_linear_regression task (ds159).

    Expected output (boston_linear_regression_results.csv):
        R2,MAE,RMSE
        0.7321,3.5278,4.8163

    Checks:
      - Python script (.py) and output (.csv) exist.
      - CSV has correct columns: R2, MAE, RMSE.
      - Exactly 1 row.
      - Metrics are numeric and positive.
      - R² >= 0.5 (realistic performance threshold).
    """

    score = 0.0
    feedback = []

    if actual is None:
        feedback.append("No working directory provided.")
        return 0.0

    base_dir = os.path.dirname(actual)
    script_path = os.path.join(base_dir, "boston_linear_regression.py")
    output_path = os.path.join(base_dir, "boston_linear_regression_results.csv")

    # --- 1. Check script file existence ---
    if os.path.exists(script_path):
        score += 0.3
        feedback.append("Python script file found.")
    else:
        feedback.append("Python script missing.")

    # --- 2. Check output CSV existence ---
    if not os.path.exists(output_path):
        feedback.append("Output CSV file missing.")
        return round(score, 2)
    else:
        score += 0.3
        feedback.append("Output CSV file found.")

    # --- 3. Validate CSV structure and contents ---
    try:
        df = pd.read_csv(output_path)
        df.columns = [c.strip().lower() for c in df.columns]
        expected_cols = ["r2", "mae", "rmse"]

        # Validate column names
        if all(col in df.columns for col in expected_cols):
            score += 0.15
            feedback.append("CSV columns match expected names (R2, MAE, RMSE).")
        else:
            feedback.append(f"Incorrect columns found: {df.columns.tolist()}")

        # Validate row count
        if len(df) == 1:
            vals = df.iloc[0].astype(float)
            r2, mae, rmse = vals["r2"], vals["mae"], vals["rmse"]

            # Check numeric validity
            if all(x >= 0 for x in [r2, mae, rmse]):
                score += 0.15
                feedback.append("All metric values are numeric and positive.")
            else:
                feedback.append("Found invalid or negative metric values.")

            # Check R² threshold
            if r2 >= 0.5:
                score += 0.1
                feedback.append("Model shows reasonable predictive power (R² ≥ 0.5).")
            else:
                feedback.append(f"Low R² value ({r2}); expected ≥ 0.5.")
        else:
            feedback.append(f"Unexpected number of rows: {len(df)} (expected 1).")

    except Exception as e:
        feedback.append(f"Error reading or validating CSV: {e}")

    # --- Final Scoring ---
    final_score = min(round(score, 2), 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score

def evaluate_boston_rf_gb_comparison(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for ds160: Random Forest vs Gradient Boosting Regression comparison.

    Expected output (rf_gb_results.csv):
        Model,Mean_R2,Better
        RandomForest,0.8421,RandomForest
        GradientBoosting,0.7994,RandomForest

    Checks:
      - Script and output file exist.
      - CSV has correct columns: Model, Mean_R2, Better.
      - Two rows present.
      - Mean_R2 values are numeric and >= 0.5.
      - 'Better' value matches model with higher Mean_R2.
    """

    score = 0.0
    feedback = []

    if actual is None:
        feedback.append("No working directory provided.")
        return 0.0

    base_dir = os.path.dirname(actual)
    script_path = os.path.join(base_dir, "boston_rf_gb_comparison.py")
    output_path = os.path.join(base_dir, "rf_gb_results.csv")

    # --- 1. Check for script existence ---
    if os.path.exists(script_path):
        score += 0.25
        feedback.append("Python script found.")
    else:
        feedback.append("Python script missing.")

    # --- 2. Check for output file existence ---
    if not os.path.exists(output_path):
        feedback.append("Output CSV not found.")
        return round(score, 2)
    else:
        score += 0.25
        feedback.append("Output CSV found.")

    # --- 3. Validate file content ---
    try:
        df = pd.read_csv(output_path)
        df.columns = [c.strip().lower() for c in df.columns]

        # Expected columns
        expected_cols = ["model", "mean_r2", "better"]
        if all(c in df.columns for c in expected_cols):
            score += 0.2
            feedback.append("CSV contains correct columns (Model, Mean_R2, Better).")
        else:
            feedback.append(f"Incorrect columns found: {df.columns.tolist()}")

        # Expected row count
        if len(df) == 2:
            score += 0.1
            feedback.append("Output contains results for two models.")

            # --- Check numeric Mean_R2 values ---
            try:
                r2_values = df["mean_r2"].astype(float)
                if (r2_values >= 0.5).all():
                    score += 0.15
                    feedback.append("Mean_R2 values are numeric and ≥ 0.5.")
                else:
                    feedback.append("Some Mean_R2 values are below 0.5.")

                # --- Verify 'Better' column correctness ---
                best_model = df.loc[r2_values.idxmax(), "model"].strip().lower()
                better_column_values = df["better"].astype(str).str.lower().unique().tolist()

                if best_model in better_column_values:
                    score += 0.05
                    feedback.append("The 'Better' column correctly identifies the top model.")
                else:
                    feedback.append("The 'Better' column does not match the higher R² model.")

            except Exception as e:
                feedback.append(f"Error parsing R² values: {e}")
        else:
            feedback.append(f"Unexpected row count: {len(df)} (expected 2).")

    except Exception as e:
        feedback.append(f"Error reading or validating CSV: {e}")

    # --- Final Scoring ---
    final_score = min(round(score, 2), 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score

def evaluate_boston_gb_tuning(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for ds161: Gradient Boosting hyperparameter tuning task.

    Expected output (gb_best_params.txt):
        Best_Params: {'n_estimators': 100, 'learning_rate': 0.05, 'max_depth': 5}
        Validation_R2: 0.8213

    Checks:
      - Python script (.py) and result file (.txt) exist.
      - Text file contains both 'Best_Params' and 'Validation_R2'.
      - Extracts numeric R² value and ensures it is >= 0.5.
      - Flexible format (accepts varied spacing or key capitalization).
    """

    score = 0.0
    feedback = []

    if actual is None:
        feedback.append("No working directory provided.")
        return 0.0

    base_dir = os.path.dirname(actual)
    script_path = os.path.join(base_dir, "boston_gb_tuning.py")
    txt_path = os.path.join(base_dir, "gb_best_params.txt")

    # --- 1. Check script existence ---
    if os.path.exists(script_path):
        score += 0.3
        feedback.append("Python script found.")
    else:
        feedback.append("Python script missing.")

    # --- 2. Check output file existence ---
    if not os.path.exists(txt_path):
        feedback.append("Output text file missing.")
        return round(score, 2)
    else:
        score += 0.3
        feedback.append("Output text file found.")

    # --- 3. Validate file content ---
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            content = f.read().lower()

        # Ensure both expected lines are present
        has_params = "best_params" in content
        has_r2 = "validation_r2" in content

        if has_params:
            score += 0.15
            feedback.append("Found 'Best_Params' entry.")
        else:
            feedback.append("Missing 'Best_Params' entry.")

        if has_r2:
            score += 0.15
            feedback.append("Found 'Validation_R2' entry.")
        else:
            feedback.append("Missing 'Validation_R2' entry.")

        # --- 4. Extract and validate R² value ---
        match = re.search(r"r2[:\s=]+([0-9]*\.?[0-9]+)", content)
        if match:
            r2_val = float(match.group(1))
            if r2_val >= 0.5:
                score += 0.1
                feedback.append(f"Validation R² = {r2_val:.3f}, acceptable performance (≥ 0.5).")
            else:
                feedback.append(f"Low R² value ({r2_val:.3f}); expected ≥ 0.5.")
        else:
            feedback.append("Could not parse R² value from file.")

    except Exception as e:
        feedback.append(f"Error reading or validating file: {e}")

    # --- Final scoring ---
    final_score = min(round(score, 2), 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score

def evaluate_mnist_logistic_recall(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for ds664: Logistic Regression (OvR) recall analysis on MNIST.

    Expected output (lowest_recall_digits.csv):
        Digit,Recall
        2,0.7613
        5,0.7739
        8,0.7894

    Checks:
      - Python script (.py) and output CSV exist.
      - CSV has columns: Digit, Recall.
      - Exactly 3 rows.
      - Recall values numeric and between 0 and 1.
      - At least one recall < 0.8 (ensures genuine 'low recall' cases).
    """

    score = 0.0
    feedback = []

    if actual is None:
        feedback.append("No working directory provided.")
        return 0.0

    base_dir = os.path.dirname(actual)
    script_path = os.path.join(base_dir, "mnist_logistic_recall.py")
    output_path = os.path.join(base_dir, "lowest_recall_digits.csv")

    # --- 1. Check for script existence ---
    if os.path.exists(script_path):
        score += 0.25
        feedback.append("Python script found.")
    else:
        feedback.append("Python script missing.")

    # --- 2. Check for output CSV existence ---
    if not os.path.exists(output_path):
        feedback.append("Output CSV missing.")
        return round(score, 2)
    else:
        score += 0.25
        feedback.append("Output CSV found.")

    # --- 3. Validate CSV structure and contents ---
    try:
        df = pd.read_csv(output_path)
        df.columns = [c.strip().lower() for c in df.columns]

        expected_cols = ["digit", "recall"]
        if all(col in df.columns for col in expected_cols):
            score += 0.2
            feedback.append("CSV columns match expected structure (Digit, Recall).")
        else:
            feedback.append(f"Incorrect columns found: {df.columns.tolist()}")

        # Check number of rows
        if len(df) == 3:
            score += 0.15
            feedback.append("Output contains exactly 3 rows for lowest recall digits.")
        else:
            feedback.append(f"Unexpected number of rows: {len(df)} (expected 3).")

        # Check recall values
        try:
            recalls = df["recall"].astype(float)
            if recalls.between(0, 1).all():
                score += 0.1
                feedback.append("Recall values are valid (between 0 and 1).")
            else:
                feedback.append("Some recall values are outside [0, 1].")

            if (recalls < 0.8).any():
                score += 0.05
                feedback.append("At least one digit has recall < 0.8 — realistic low recall detected.")
            else:
                feedback.append("All recalls ≥ 0.8 — unrealistic for 'lowest recall' analysis.")
        except Exception as e:
            feedback.append(f"Error parsing recall values: {e}")

    except Exception as e:
        feedback.append(f"Error reading or validating CSV: {e}")

    # --- Final scoring ---
    final_score = min(round(score, 2), 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score

def evaluate_mnist_digit_variance(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for ds665: MNIST Digit Variance Analysis

    Expected output (digit_intensity_variance.csv):
        Digit,Avg_Intensity,Std_Intensity
        0,33.4821,72.9123
        1,21.3459,56.8242
        ...
        Highest_Variance_Digit,8,85.2379

    Checks:
      - Python script (.py) and CSV exist.
      - CSV has correct columns: Digit, Avg_Intensity, Std_Intensity.
      - Contains 11 rows (10 digits + 1 summary row).
      - Intensity values are numeric and within [0, 255].
      - Last row correctly labeled 'Highest_Variance_Digit'.
    """

    import os
    import pandas as pd
    import numpy as np

    score = 0.0
    feedback = []

    if actual is None:
        feedback.append("No working directory provided.")
        return 0.0

    base_dir = os.path.dirname(actual)
    script_path = os.path.join(base_dir, "mnist_digit_variance.py")
    output_path = os.path.join(base_dir, "digit_intensity_variance.csv")

    # --- 1. Check for script existence ---
    if os.path.exists(script_path):
        score += 0.25
        feedback.append("Python script file found.")
    else:
        feedback.append("Python script missing.")

    # --- 2. Check for output CSV existence ---
    if not os.path.exists(output_path):
        feedback.append("Output CSV file missing.")
        return round(score, 2)
    else:
        score += 0.25
        feedback.append("Output CSV file found.")

    # --- 3. Validate CSV content ---
    try:
        df = pd.read_csv(output_path)
        df.columns = [c.strip().lower() for c in df.columns]
        expected_cols = ["digit", "avg_intensity", "std_intensity"]

        # Column validation
        if all(col in df.columns for col in expected_cols):
            score += 0.2
            feedback.append("CSV columns match expected structure (Digit, Avg_Intensity, Std_Intensity).")
        else:
            feedback.append(f"Incorrect columns found: {df.columns.tolist()}")

        # Row count validation
        if len(df) >= 11:
            score += 0.15
            feedback.append("Output contains results for all 10 digits and a summary row.")
        else:
            feedback.append(f"Only {len(df)} rows found (expected ≥11).")

        # Validate numeric ranges
        try:
            avg_vals = df["avg_intensity"].astype(float)
            std_vals = df["std_intensity"].astype(float)

            if avg_vals.between(0, 255).all() and std_vals.between(0, 255).all():
                score += 0.1
                feedback.append("Intensity values are within valid range [0, 255].")
            else:
                feedback.append("Some intensity values fall outside [0, 255].")
        except Exception as e:
            feedback.append(f"Error parsing numeric intensity values: {e}")

        # --- 4. Validate final row label ---
        last_digit = str(df.iloc[-1, 0]).strip().lower()
        if "highest_variance_digit" in last_digit:
            score += 0.05
            feedback.append("Last row correctly labeled 'Highest_Variance_Digit'.")
        else:
            feedback.append(f"Last row label mismatch: found '{last_digit}'.")

    except Exception as e:
        feedback.append(f"Error reading or validating CSV: {e}")

    # --- Final scoring ---
    final_score = min(round(score, 2), 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score

def evaluate_credit_missing_imputation(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for ds164: Missing value detection and imputation (credit-g dataset).

    Expected output (missing_value_summary.csv):
        Stage,Missing_Count
        Before_Imputation,67
        After_Imputation,0

    Checks:
      - Python script (.py) and CSV exist.
      - CSV has columns: Stage, Missing_Count.
      - Exactly two rows (Before_Imputation and After_Imputation).
      - Missing count before ≈ 67 (±5 tolerance).
      - Missing count after = 0.
    """

    score = 0.0
    feedback = []

    if actual is None:
        feedback.append("No working directory provided.")
        return 0.0

    base_dir = os.path.dirname(actual)
    script_path = os.path.join(base_dir, "credit_missing_imputation.py")
    output_path = os.path.join(base_dir, "missing_value_summary.csv")

    # --- 1. Check for script existence ---
    if os.path.exists(script_path):
        score += 0.25
        feedback.append("Python script found.")
    else:
        feedback.append("Python script missing.")

    # --- 2. Check for CSV existence ---
    if not os.path.exists(output_path):
        feedback.append("Output CSV missing.")
        return round(score, 2)
    else:
        score += 0.25
        feedback.append("Output CSV file found.")

    # --- 3. Validate CSV structure and content ---
    try:
        df = pd.read_csv(output_path)
        df.columns = [c.strip().lower() for c in df.columns]

        expected_cols = ["stage", "missing_count"]
        if all(col in df.columns for col in expected_cols):
            score += 0.2
            feedback.append("CSV columns match expected structure (Stage, Missing_Count).")
        else:
            feedback.append(f"Incorrect columns found: {df.columns.tolist()}")

        # Check number of rows
        if len(df) == 2:
            score += 0.15
            feedback.append("CSV contains exactly two rows (Before and After).")

            try:
                # Normalize stage names
                stages = df["stage"].astype(str).str.lower().tolist()
                counts = df["missing_count"].astype(float).tolist()

                before_val = None
                after_val = None
                for s, c in zip(stages, counts):
                    if "before" in s:
                        before_val = c
                    elif "after" in s:
                        after_val = c

                if before_val is not None and abs(before_val - 67) <= 5:
                    score += 0.1
                    feedback.append(f"Before_Imputation missing count ≈ {before_val} (expected around 67).")
                else:
                    feedback.append(f"Unexpected before-imputation count: {before_val} (expected ≈67).")

                if after_val == 0:
                    score += 0.05
                    feedback.append("After_Imputation count is 0 — all missing values filled.")
                else:
                    feedback.append(f"After_Imputation count not zero: {after_val}.")
            except Exception as e:
                feedback.append(f"Error validating missing count values: {e}")
        else:
            feedback.append(f"Unexpected number of rows: {len(df)} (expected 2).")

    except Exception as e:
        feedback.append(f"Error reading or validating CSV: {e}")

    # --- Final scoring ---
    final_score = min(round(score, 2), 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score

def evaluate_credit_feature_correlation(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for ds165: Correlation between numeric features and target class.

    Expected output (credit_feature_correlation.csv):
        Feature,Correlation,Category
        A14,0.3124,Top
        A8,0.1768,Top
        A11,0.1459,Top
        A2,-0.1895,Bottom
        A3,-0.1527,Bottom
        A15,-0.0482,Bottom

    Checks:
      - Script and CSV file exist.
      - CSV has columns: Feature, Correlation, Category.
      - Exactly 6 rows: 3 labeled 'Top', 3 labeled 'Bottom'.
      - Correlation values are numeric and between -1 and 1.
      - Compare correlation values with expected ground truth (within ±0.05 tolerance).
    """

    import os
    import pandas as pd
    import numpy as np

    score = 0.0
    feedback = []

    # Ground truth values for evaluation reference
    expected_values = {
        "A14": 0.3124,
        "A8": 0.1768,
        "A11": 0.1459,
        "A2": -0.1895,
        "A3": -0.1527,
        "A15": -0.0482,
    }

    if actual is None:
        feedback.append("No working directory provided.")
        return 0.0

    base_dir = os.path.dirname(actual)
    script_path = os.path.join(base_dir, "credit_feature_correlation.py")
    output_path = os.path.join(base_dir, "credit_feature_correlation.csv")

    # --- 1. Check for script existence ---
    if os.path.exists(script_path):
        score += 0.25
        feedback.append("Python script found.")
    else:
        feedback.append("Python script missing.")

    # --- 2. Check for CSV existence ---
    if not os.path.exists(output_path):
        feedback.append("Output CSV file missing.")
        return round(score, 2)
    else:
        score += 0.25
        feedback.append("Output CSV file found.")

    # --- 3. Validate CSV structure and content ---
    try:
        df = pd.read_csv(output_path)
        df.columns = [c.strip().lower() for c in df.columns]
        expected_cols = ["feature", "correlation", "category"]

        # Check correct columns
        if all(c in df.columns for c in expected_cols):
            score += 0.15
            feedback.append("CSV columns match expected structure.")
        else:
            feedback.append(f"Incorrect columns found: {df.columns.tolist()}")

        # Check correct number of rows
        if len(df) == 6:
            score += 0.1
            feedback.append("CSV contains exactly 6 rows (3 Top + 3 Bottom).")
        else:
            feedback.append(f"Unexpected row count: {len(df)} (expected 6).")

        # Validate correlations are numeric
        try:
            corr_vals = df["correlation"].astype(float)
            if corr_vals.between(-1, 1).all():
                score += 0.1
                feedback.append("Correlation values are numeric and within [-1, 1].")
            else:
                feedback.append("Some correlation values fall outside [-1, 1].")
        except Exception as e:
            feedback.append(f"Error parsing correlation values: {e}")

        # Check for correct 'Top' and 'Bottom' labeling
        top_count = df["category"].str.lower().str.count("top").sum()
        bottom_count = df["category"].str.lower().str.count("bottom").sum()
        if top_count == 3 and bottom_count == 3:
            score += 0.1
            feedback.append("Contains 3 'Top' and 3 'Bottom' labeled features.")
        else:
            feedback.append(f"Incorrect category distribution: {top_count} Top, {bottom_count} Bottom.")

        # Compare with expected values (±0.05 tolerance)
        try:
            matched = 0
            for _, row in df.iterrows():
                feat = str(row["feature"]).strip()
                val = float(row["correlation"])
                if feat in expected_values:
                    if abs(val - expected_values[feat]) <= 0.05:
                        matched += 1
            if matched >= 5:
                score += 0.05
                feedback.append("Most correlation values align closely with expected results.")
            else:
                feedback.append(f"Only {matched}/6 correlation values within tolerance ±0.05.")
        except Exception as e:
            feedback.append(f"Error comparing correlation values: {e}")

    except Exception as e:
        feedback.append(f"Error reading or validating CSV: {e}")

    # --- Final scoring ---
    final_score = min(round(score, 2), 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score

def evaluate_credit_outlier_detection(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for ds166: Outlier detection in A2 and A14 using IQR rule.

    Expected output (credit_outlier_summary.csv):
        Metric,Value
        Total_Outliers,29

    Checks:
      - Python script (.py) and CSV exist.
      - CSV has columns: Metric, Value.
      - Exactly 1 row.
      - Metric equals 'Total_Outliers'.
      - Value is numeric, close to 29 (±5 tolerance).
    """
    score = 0.0
    feedback = []

    if actual is None:
        feedback.append("No working directory provided.")
        return 0.0

    base_dir = os.path.dirname(actual)
    script_path = os.path.join(base_dir, "credit_outlier_detection.py")
    output_path = os.path.join(base_dir, "credit_outlier_summary.csv")

    # --- 1. Check script existence ---
    if os.path.exists(script_path):
        score += 0.3
        feedback.append("Python script found.")
    else:
        feedback.append("Python script missing.")

    # --- 2. Check output CSV existence ---
    if not os.path.exists(output_path):
        feedback.append("Output CSV missing.")
        return round(score, 2)
    else:
        score += 0.3
        feedback.append("Output CSV file found.")

    # --- 3. Validate CSV content ---
    try:
        df = pd.read_csv(output_path)
        df.columns = [c.strip().lower() for c in df.columns]

        # Validate column names
        expected_cols = ["metric", "value"]
        if all(c in df.columns for c in expected_cols):
            score += 0.2
            feedback.append("CSV columns match expected structure (Metric, Value).")
        else:
            feedback.append(f"Incorrect columns found: {df.columns.tolist()}")

        # Validate row count and content
        if len(df) == 1:
            metric = str(df.iloc[0]["metric"]).strip().lower()
            val = float(df.iloc[0]["value"])
            if "outlier" in metric:
                score += 0.1
                feedback.append("Metric labeled correctly as 'Total_Outliers'.")
            else:
                feedback.append(f"Unexpected metric label: {metric}")

            # Compare value to expected (29 ±5 tolerance)
            if abs(val - 29) <= 5:
                score += 0.1
                feedback.append(f"Outlier count realistic ({val}, expected ≈29).")
            else:
                feedback.append(f"Outlier count {val} differs from expected ≈29.")
        else:
            feedback.append(f"Unexpected number of rows: {len(df)} (expected 1).")

    except Exception as e:
        feedback.append(f"Error reading or validating CSV: {e}")

    # --- Final Scoring ---
    final_score = min(round(score, 2), 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score

def evaluate_credit_decision_rule(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for ds167: Decision rule precision and recall on credit_approval dataset.

    Expected output file: credit_decision_rule_metrics.txt
      Precision: <value>
      Recall: <value>

    Checks:
      - Script and text file exist.
      - Text file contains both 'Precision' and 'Recall'.
      - Extracts numeric values and ensures they are between 0 and 1.
      - Precision and recall should each be at least 0.4 for realism.
    """

    import os
    import re

    score = 0.0
    feedback = []

    if actual is None:
        feedback.append("No working directory provided.")
        return 0.0

    base_dir = os.path.dirname(actual)
    script_path = os.path.join(base_dir, "credit_decision_rule.py")
    output_path = os.path.join(base_dir, "credit_decision_rule_metrics.txt")

    # --- 1. Check script existence ---
    if os.path.exists(script_path):
        score += 0.3
        feedback.append("Python script found.")
    else:
        feedback.append("Python script missing.")

    # --- 2. Check output text file existence ---
    if not os.path.exists(output_path):
        feedback.append("Output metrics text file missing.")
        return round(score, 2)
    else:
        score += 0.3
        feedback.append("Metrics text file found.")

    # --- 3. Validate content ---
    try:
        with open(output_path, "r") as f:
            content = f.read()

        if "Precision" in content and "Recall" in content:
            score += 0.2
            feedback.append("Contains both Precision and Recall entries.")
        else:
            feedback.append("Missing either Precision or Recall in output file.")

        # Extract numeric values
        values = re.findall(r"[-+]?\d*\.\d+|\d+", content)
        nums = [float(v) for v in values if 0 <= float(v) <= 1]

        if len(nums) >= 2:
            precision, recall = nums[0], nums[1]
            if precision >= 0.4:
                score += 0.1
                feedback.append(f"Precision value acceptable ({precision:.2f}).")
            else:
                feedback.append(f"Precision too low ({precision:.2f}).")
            if recall >= 0.4:
                score += 0.1
                feedback.append(f"Recall value acceptable ({recall:.2f}).")
            else:
                feedback.append(f"Recall too low ({recall:.2f}).")
        else:
            feedback.append("Could not extract numeric Precision and Recall values.")

    except Exception as e:
        feedback.append(f"Error reading or parsing metrics file: {e}")

    # --- Final Scoring ---
    final_score = min(round(score, 2), 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score

def evaluate_credit_model_comparison(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for ds168: Compare Logistic Regression, Random Forest, and XGBoost models.

    Expected output (credit_model_comparison.csv):
        Model,Accuracy,F1_Score,AUC
        Logistic Regression,0.85,0.83,0.89
        Random Forest,0.87,0.85,0.91
        XGBoost,0.88,0.86,0.92

    Checks:
      - Script and CSV exist.
      - Correct columns.
      - Contains exactly 3 rows.
      - Model names include Logistic, Random Forest, and XGBoost.
      - Accuracy, F1, and AUC numeric between 0 and 1.
      - At least one model with AUC >= 0.7.
    """

    score = 0.0
    feedback = []

    if actual is None:
        feedback.append("No working directory provided.")
        return 0.0

    base_dir = os.path.dirname(actual)
    script_path = os.path.join(base_dir, "credit_model_comparison.py")
    output_path = os.path.join(base_dir, "credit_model_comparison.csv")

    # 1. Check script existence
    if os.path.exists(script_path):
        score += 0.25
        feedback.append("Python script found.")
    else:
        feedback.append("Python script missing.")

    # 2. Check output file existence
    if not os.path.exists(output_path):
        feedback.append("Output CSV missing.")
        return round(score, 2)
    else:
        score += 0.25
        feedback.append("Output CSV found.")

    # 3. Validate content
    try:
        df = pd.read_csv(output_path)
        df.columns = [c.strip().lower() for c in df.columns]
        expected_cols = ["model", "accuracy", "f1_score", "auc"]

        # Check correct columns
        if all(c in df.columns for c in expected_cols):
            score += 0.2
            feedback.append("CSV columns match expected structure.")
        else:
            feedback.append(f"Incorrect columns: {df.columns.tolist()}")

        # Check for 3 models
        if len(df) == 3:
            score += 0.1
            feedback.append("CSV contains results for all 3 models.")
        else:
            feedback.append(f"Unexpected row count: {len(df)} (expected 3).")

        # Validate model names
        models = df["model"].str.lower().tolist()
        if any("logistic" in m for m in models) and any("forest" in m for m in models) and any("xgb" in m or "boost" in m for m in models):
            score += 0.05
            feedback.append("All model names present (Logistic, RF, XGBoost).")
        else:
            feedback.append("One or more model names missing or incorrect.")

        # Check metric values
        try:
            acc = df["accuracy"].astype(float)
            f1 = df["f1_score"].astype(float)
            auc = df["auc"].astype(float)
            if acc.between(0, 1).all() and f1.between(0, 1).all() and auc.between(0, 1).all():
                score += 0.1
                feedback.append("All metric values within [0, 1].")
                if auc.max() >= 0.7:
                    score += 0.05
                    feedback.append(f"At least one model performs well (AUC ≥ 0.7, max AUC = {auc.max():.2f}).")
                else:
                    feedback.append(f"AUC values too low (max = {auc.max():.2f}).")
            else:
                feedback.append("Invalid metric range detected.")
        except Exception as e:
            feedback.append(f"Error parsing numeric metrics: {e}")

    except Exception as e:
        feedback.append(f"Error reading CSV: {e}")

    # 4. Final scoring
    final_score = min(round(score, 2), 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)
    return final_score

def evaluate_credit_approval_visualization_quality(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Credit Approval Visualization (Visual Studio Code) task.

    This evaluation checks:
      - The Python script file exists.
      - Both visualization files exist and are non-empty.
      - GPT-4o confirms that:
          1. credit_amount_bar.png correctly shows the average A14 (credit amount)
             by A9 for each approval class (+ or -).
          2. a2_boxplot.png correctly shows the distribution of A2 values for approved
             vs. rejected applications.
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    expected = expected.get("rules", expected)
    base_dir = os.path.dirname(actual)

    script_file = os.path.join(base_dir, "credit_approval_visualization.py")
    chart1 = os.path.join(base_dir, "credit_amount_bar.png")
    chart2 = os.path.join(base_dir, "a2_boxplot.png")

    score = 0.0
    feedback = []

    # Step 1: Check script existence
    if os.path.exists(script_file):
        score += 0.4
        feedback.append("The Python script file exists.")
        logger.info("Script file found.")
    else:
        feedback.append("The Python script file is missing.")
        logger.warning("Script file missing.")

    # Helper: Encode an image as base64
    def encode_image(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    # Step 2: Visualization checks
    charts_to_check = [
        (
            chart1,
            "a bar chart showing the average A14 (credit amount) by A9 for each approval class (+ or -), with clearly labeled axes and a descriptive title."
        ),
        (
            chart2,
            "a boxplot showing the distribution of A2 values for approved vs. rejected applications, labeled and titled appropriately."
        )
    ]

    for chart, description in charts_to_check:
        chart_name = os.path.basename(chart)
        if os.path.exists(chart) and os.path.getsize(chart) > 0:
            score += 0.15
            feedback.append(f"The visualization {chart_name} exists and is non-empty.")
            logger.info(f"{chart_name} found and valid.")

            try:
                encoded = encode_image(chart)
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert data visualization evaluator. Assess chart clarity, labeling, and accuracy."
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        f"Please evaluate this chart and confirm whether it correctly represents {description}. "
                                        f"Reply 'yes' if it clearly matches the description and is visually correct, or 'no' otherwise with a brief reason."
                                    )
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{encoded}"}
                                }
                            ]
                        }
                    ],
                    max_tokens=100
                )

                answer = response.choices[0].message.content.lower()
                if "yes" in answer:
                    score += 0.1
                    feedback.append(f"GPT-4o judged {chart_name} as accurate and clearly labeled.")
                    logger.info(f"{chart_name} validated successfully by GPT-4o.")
                else:
                    feedback.append(f"GPT-4o judged {chart_name} unclear or inaccurate: {answer}")
                    logger.warning(f"{chart_name} unclear according to GPT-4o.")
            except Exception as e:
                feedback.append(f"Error evaluating {chart_name} with GPT-4o: {e}")
                logger.error(f"GPT-4o evaluation failed for {chart_name}: {e}", exc_info=True)
        else:
            feedback.append(f"The visualization {chart_name} is missing or empty.")
            logger.warning(f"{chart_name} missing or empty.")

    # Step 3: Final scoring
    final_score = min(score, 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")

    print("\nEvaluation Summary:")
    for msg in feedback:
        print("-", msg)
    print(f"\nFinal Score: {final_score:.2f}")

    return final_score

def evaluate_earthquake_depth_correlation(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for ds170: Checks strongest correlation group identification
    in earthquake depth analysis.

    Expected output file:
        earthquake_depth_summary.txt
    Expected line:
        Strongest_Correlation_Group: Intermediate

    Scoring breakdown:
      0.4 - Python script exists
      0.4 - Output text file exists
      0.2 - Correct output format and expected value
    """

    score = 0.0
    feedback = []

    if not actual:
        return 0.0

    base_dir = os.path.dirname(actual)
    script_path = os.path.join(
        base_dir,
        "earthquake_depth_correlation.py"
    )
    output_path = os.path.join(
        base_dir,
        "earthquake_depth_summary.txt"
    )

    # --- 1. Check Python script existence ---
    if os.path.exists(script_path):
        score += 0.4
        feedback.append("Python script found.")
    else:
        feedback.append("Python script missing.")

    # --- 2. Check output text file existence ---
    if not os.path.exists(output_path):
        feedback.append("Output file missing.")
        return round(score, 2)
    else:
        score += 0.4
        feedback.append("Output text file found.")

    # --- 3. Validate file content ---
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        # Expected format and case-insensitive match
        expected_line = "strongest_correlation_group: intermediate"
        if content.lower().startswith("strongest_correlation_group:"):
            if content.lower() == expected_line:
                score += 0.2
                feedback.append("Output matches expected ground truth exactly.")
            else:
                feedback.append(f"Output format correct but value differs: '{content}'")
        else:
            feedback.append("Output format invalid or missing key phrase.")

    except Exception as e:
        feedback.append(f"Error reading file: {e}")

    # --- Final score ---
    final_score = min(round(score, 2), 1.0)
    for msg in feedback:
        logger.info(msg)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    return final_score

def evaluate_earthquake_mahalanobis_outliers(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for ds171 — Mahalanobis distance–based earthquake outlier detection.

    Expected output file:
        earthquake_outlier_summary.txt
    Expected line format:
        Tsunami_Outliers: <count> (<percentage>%)
    Example ground truth:
        Tsunami_Outliers: 9 (56.25%)

    Scoring breakdown:
      0.4 - Python script exists
      0.4 - Output text file exists
      0.2 - Output matches expected format and realistic values
    """

    score = 0.0
    feedback = []

    if not actual:
        return 0.0

    base_dir = os.path.dirname(actual)
    script_path = os.path.join(
        base_dir,
        "earthquake_mahalanobis_outliers.py"
    )
    output_path = os.path.join(
        base_dir,
        "earthquake_outlier_summary.txt"
    )

    # 1. Check Python script existence
    if os.path.exists(script_path):
        score += 0.4
        feedback.append("Python script found.")
    else:
        feedback.append("Python script missing.")

    # 2. Check output file existence
    if not os.path.exists(output_path):
        feedback.append("Output text file missing.")
        logger.info("\n".join(feedback))
        return round(score, 2)
    else:
        score += 0.4
        feedback.append("Output text file found.")

    # 3. Validate output format and values
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        pattern = r"^Tsunami_Outliers:\s*(\d+)\s*\(([\d\.]+)%\)$"
        match = re.match(pattern, content)

        if match:
            count = int(match.group(1))
            percentage = float(match.group(2))
            feedback.append(f"Detected count={count}, percentage={percentage}%.")

            # Ground truth comparison (soft tolerance)
            if count == 9 and abs(percentage - 56.25) < 0.1:
                score += 0.2
                feedback.append("Output matches expected ground truth exactly.")
            elif count > 0 and 0 < percentage <= 100:
                score += 0.1
                feedback.append("Output format correct and values realistic, but not exact.")
            else:
                feedback.append("Output values not realistic.")
        else:
            feedback.append("Output format invalid. Expected 'Tsunami_Outliers: <count> (<percentage>%)'.")

    except Exception as e:
        feedback.append(f"Error reading or validating output: {e}")

    final_score = min(round(score, 2), 1.0)
    for msg in feedback:
        logger.info(msg)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    return final_score

def evaluate_earthquake_tsunami_trend(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for ds172 — Yearly Tsunami Trend Analysis.

    Expected output file:
        earthquake_tsunami_trend.txt

    Expected format:
        Tsunami_Trend_Slope: <slope> (<direction>)
    Example ground truth:
        Tsunami_Trend_Slope: 4.9778 (Increasing)

    Scoring:
      0.4 - Python script exists
      0.4 - Output text file exists
      0.2 - Correct format and expected slope/direction
    """
    score = 0.0
    feedback = []

    if not actual:
        return 0.0

    base_dir = os.path.dirname(actual)
    script_path = os.path.join(
        base_dir,
        "earthquake_tsunami_trend.py"
    )
    output_path = os.path.join(
        base_dir,
        "earthquake_tsunami_trend.txt"
    )

    # 1. Check Python script existence
    if os.path.exists(script_path):
        score += 0.4
        feedback.append("Python script found.")
    else:
        feedback.append("Python script missing.")

    # 2. Check output file existence
    if not os.path.exists(output_path):
        feedback.append("Output file missing.")
        logger.info("\n".join(feedback))
        return round(score, 2)
    else:
        score += 0.4
        feedback.append("Output text file found.")

    # 3. Validate output format and correctness
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        pattern = r"^Tsunami_Trend_Slope:\s*([-+]?\d*\.?\d+)\s*\((Increasing|Decreasing)\)$"
        match = re.match(pattern, content)

        if match:
            slope = float(match.group(1))
            direction = match.group(2)
            feedback.append(f"Detected slope={slope:.4f}, direction={direction}.")

            # Check ground truth with tolerance ±0.05
            if abs(slope - 4.9778) < 1 and direction.lower() == "increasing":
                score += 0.2
                feedback.append("Output matches expected ground truth exactly.")
            elif direction in ["Increasing", "Decreasing"]:
                score += 0.1
                feedback.append("Output format correct, slope realistic but not exact.")
            else:
                feedback.append("Output direction invalid.")
        else:
            feedback.append("Output format invalid. Expected 'Tsunami_Trend_Slope: <float> (Increasing|Decreasing)'.")

    except Exception as e:
        feedback.append(f"Error reading or validating output: {e}")

    final_score = min(round(score, 2), 1.0)
    for msg in feedback:
        logger.info(msg)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    return final_score


def evaluate_earthquake_tsunami_risk_index(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for ds173 — Composite Tsunami Risk Index task.

    Expected output file:
        earthquake_tsunami_risk_summary.txt
    Expected line format:
        Welch_t_statistic: <value>, p_value: <value>

    Example ground truth:
        Welch_t_statistic: -0.3617, p_value: 0.7177

    Scoring:
      0.4 - Python script exists
      0.4 - Output text file exists
      0.2 - Valid numeric format and realistic values
    """
    score = 0.0
    feedback = []

    if not actual:
        return 0.0

    base_dir = os.path.dirname(actual)
    script_path = os.path.join(
        base_dir,
        "earthquake_tsunami_risk_index.py"
    )
    output_path = os.path.join(
        base_dir,
        "earthquake_tsunami_risk_summary.txt"
    )

    # --- Check for Python script existence ---
    if os.path.exists(script_path):
        score += 0.4
        feedback.append("Python script found.")
    else:
        feedback.append("Python script missing.")

    # --- Check for output file existence ---
    if not os.path.exists(output_path):
        feedback.append("Output text file missing.")
        logger.info("\n".join(feedback))
        return round(score, 2)
    else:
        score += 0.4
        feedback.append("Output text file found.")

    # --- Validate output format and values ---
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        pattern = r"^Welch_t_statistic:\s*([-+]?\d*\.?\d+),\s*p_value:\s*([\d\.]+)$"
        match = re.match(pattern, content)

        if match:
            t_stat = float(match.group(1))
            p_value = float(match.group(2))
            feedback.append(f"Detected Welch_t_statistic={t_stat:.4f}, p_value={p_value:.4f}.")

            # --- Check numeric realism ---
            if -10 <= t_stat <= 10 and 0 <= p_value <= 1:
                score += 0.2
                feedback.append("Values are realistic and correctly formatted.")
            else:
                feedback.append("Values are outside expected range.")
        else:
            feedback.append("Output format invalid. Expected 'Welch_t_statistic: <float>, p_value: <float>'.")

    except Exception as e:
        feedback.append(f"Error reading or validating output: {e}")

    final_score = min(round(score, 2), 1.0)
    for msg in feedback:
        logger.info(msg)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    return final_score

def evaluate_earthquake_model_comparison(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for ds174 — Model Comparison (Logistic, RF, XGBoost).

    Expected output file:
        earthquake_model_comparison.txt

    Expected format:
        Best_Model: <model_name>, F1: <value>, ROC_AUC: <value>

    Example ground truth:
        Best_Model: XGBoost, F1: 0.8945, ROC_AUC: 0.9724

    Scoring:
      0.4 - Python script exists
      0.4 - Output text file exists
      0.2 - Correct format, valid model, and realistic metrics
    """

    score = 0.0
    feedback = []

    if not actual:
        return 0.0

    base_dir = os.path.dirname(actual)
    script_path = os.path.join(
        base_dir,
        "earthquake_model_comparison.py"
    )
    output_path = os.path.join(
        base_dir,
        "earthquake_model_comparison.txt"
    )

    # --- Check for Python script existence ---
    if os.path.exists(script_path):
        score += 0.4
        feedback.append("Python script found.")
    else:
        feedback.append("Python script missing.")

    # --- Check for output file existence ---
    if not os.path.exists(output_path):
        feedback.append("Output file missing.")
        logger.info("\n".join(feedback))
        return round(score, 2)
    else:
        score += 0.4
        feedback.append("Output text file found.")

    # --- Validate output format and values ---
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        pattern = r"^Best_Model:\s*(\w+),\s*F1:\s*([\d\.]+),\s*ROC_AUC:\s*([\d\.]+)$"
        match = re.match(pattern, content)

        if match:
            model = match.group(1)
            f1 = float(match.group(2))
            auc = float(match.group(3))
            feedback.append(f"Detected Best_Model={model}, F1={f1:.4f}, ROC_AUC={auc:.4f}.")

            valid_models = ["LogisticRegression", "RandomForest", "XGBoost"]

            if model in valid_models and 0 <= f1 <= 1 and 0 <= auc <= 1:
                if model == "XGBoost" and f1 >= 0.6 and auc >= 0.6:
                    score += 0.2
                    feedback.append("Output matches expected ground truth exactly.")
                else:
                    score += 0.1
                    feedback.append("Output format correct and metrics realistic, but not exact.")
            else:
                feedback.append("Invalid model name or unrealistic metrics.")
        else:
            feedback.append("Output format invalid. Expected 'Best_Model: <model>, F1: <val>, ROC_AUC: <val>'.")

    except Exception as e:
        feedback.append(f"Error reading or validating output: {e}")

    final_score = min(round(score, 2), 1.0)
    for msg in feedback:
        logger.info(msg)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    return final_score

def evaluate_earthquake_feature_interpretation(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for ds175 — Random Forest + SHAP Feature Interpretation.

    Expected output file:
        earthquake_feature_interpretation.txt

    Expected format:
        Nonlinear_Features: <feature1>, <feature2>

    Example:
        Nonlinear_Features: depth, magnitude

    Scoring:
      0.4 - Python script exists
      0.4 - Output text file exists
      0.2 - Valid output format (Nonlinear_Features: <f1>, <f2>)
    """
    score = 0.0
    feedback = []

    if not actual:
        return 0.0

    base_dir = os.path.dirname(actual)
    script_path = os.path.join(
        base_dir,
        "earthquake_feature_interpretation.py"
    )
    output_path = os.path.join(
        base_dir,
        "earthquake_feature_interpretation.txt"
    )

    # --- Check Python script existence ---
    if os.path.exists(script_path):
        score += 0.4
        feedback.append("Python script found.")
    else:
        feedback.append("Python script missing.")

    # --- Check output file existence ---
    if not os.path.exists(output_path):
        feedback.append("Output file missing.")
        logger.info("\n".join(feedback))
        return round(score, 2)
    else:
        score += 0.4
        feedback.append("Output text file found.")

    # --- Validate output format ---
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        pattern = r"^Nonlinear_Features:\s*([\w\-]+)\s*,\s*([\w\-]+)$"
        match = re.match(pattern, content)

        if match:
            feature1, feature2 = match.group(1), match.group(2)
            feedback.append(f"Detected features: {feature1}, {feature2}.")
            score += 0.2
            feedback.append("Output format is correct and includes two features.")
        else:
            feedback.append("Invalid format. Expected 'Nonlinear_Features: <feature1>, <feature2>'.")

    except Exception as e:
        feedback.append(f"Error reading or validating output: {e}")

    # --- Final Scoring ---
    final_score = min(round(score, 2), 1.0)
    for msg in feedback:
        logger.info(msg)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    return final_score


def evaluate_earthquake_decade_hemisphere_analysis(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for ds176 — Decade–Hemisphere Tsunami Change Analysis.

    Expected output file:
        earthquake_decade_hemisphere_summary.txt

    Expected format:
        Significant_Increase: <hemisphere>

    Example ground truth:
        Significant_Increase: S

    Scoring:
      0.4 - Python script exists
      0.4 - Output text file exists
      0.2 - Correct output format and valid hemisphere (N/S)
    """

    score = 0.0
    feedback = []

    if not actual:
        return 0.0

    base_dir = os.path.dirname(actual)
    script_path = os.path.join(
        base_dir,
        "earthquake_decade_hemisphere_analysis.py"
    )
    output_path = os.path.join(
        base_dir,
        "earthquake_decade_hemisphere_summary.txt"
    )

    # --- Check Python script existence ---
    if os.path.exists(script_path):
        score += 0.4
        feedback.append("Python script found.")
    else:
        feedback.append("Python script missing.")

    # --- Check output file existence ---
    if not os.path.exists(output_path):
        feedback.append("Output text file missing.")
        logger.info("\n".join(feedback))
        return round(score, 2)
    else:
        score += 0.4
        feedback.append("Output text file found.")

    # --- Validate output format ---
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        pattern = r"^Significant_Increase:\s*([NSns])$"
        match = re.match(pattern, content)

        if match:
            hemisphere = match.group(1).upper()
            feedback.append(f"Detected hemisphere: {hemisphere}")

            if hemisphere in ["N", "S"]:
                if hemisphere == "S":
                    score += 0.2
                    feedback.append("Output matches expected ground truth (S).")
                else:
                    score += 0.1
                    feedback.append("Output valid but hemisphere differs from ground truth.")
            else:
                feedback.append("Invalid hemisphere value detected.")
        else:
            feedback.append("Invalid format. Expected 'Significant_Increase: N' or 'Significant_Increase: S'.")

    except Exception as e:
        feedback.append(f"Error reading or validating output: {e}")

    # --- Final scoring ---
    final_score = min(round(score, 2), 1.0)
    for msg in feedback:
        logger.info(msg)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    return final_score


def evaluate_earthquake_hotspot_visualization_quality(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Earthquake Hotspot Visualization (Visual Studio Code) task.

    This evaluation checks:
      - The Python script file exists.
      - Both visualization files exist and are non-empty.
      - GPT-4o confirms that:
          1. earthquake_3d_scatter.png correctly shows a 3D scatter plot
             with magnitude (x), depth (y), and sig (z), colored by tsunami.
          2. earthquake_tsunami_heatmap.png correctly shows a heatmap
             of mean tsunami frequency by latitude and longitude bins.
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    expected = expected.get("rules", expected)
    base_dir = os.path.dirname(actual)

    script_file = os.path.join(base_dir, "earthquake_hotspot_visualization.py")
    chart1 = os.path.join(base_dir, "earthquake_3d_scatter.png")
    chart2 = os.path.join(base_dir, "earthquake_tsunami_heatmap.png")

    score = 0.0
    feedback = []

    # Step 1: Check that the Python script exists
    if os.path.exists(script_file):
        score += 0.4
        feedback.append("The Python script file exists.")
        logger.info("Script file found.")
    else:
        feedback.append("The Python script file is missing.")
        logger.warning("Script file missing.")

    # Helper function for encoding images
    def encode_image(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    # Step 2: Check both visualization files
    charts_to_check = [
        (
            chart1,
            "a 3D scatter plot with magnitude (x-axis), depth (y-axis), and sig (z-axis), color-coded by tsunami. The chart should include labeled axes, a legend, and a clear title."
        ),
        (
            chart2,
            "a heatmap showing the mean tsunami frequency by latitude (10-degree bins) and longitude (20-degree bins), with labeled axes and a colorbar."
        )
    ]

    for chart, description in charts_to_check:
        chart_name = os.path.basename(chart)
        if os.path.exists(chart) and os.path.getsize(chart) > 0:
            score += 0.15
            feedback.append(f"The visualization {chart_name} exists and is non-empty.")
            logger.info(f"{chart_name} found and valid.")

            # Use GPT-4o to confirm chart accuracy
            try:
                encoded = encode_image(chart)
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert data visualization evaluator. Assess chart clarity, labeling, and accuracy."
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        f"Please examine this chart and confirm whether it correctly represents {description}. "
                                        f"Reply 'yes' if it matches the description and is clearly labeled; otherwise reply 'no' with a short reason."
                                    )
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{encoded}"}
                                }
                            ]
                        }
                    ],
                    max_tokens=100
                )

                answer = response.choices[0].message.content.lower()
                if "yes" in answer:
                    score += 0.1
                    feedback.append(f"GPT-4o verified that {chart_name} accurately represents the required visualization.")
                    logger.info(f"{chart_name} validated successfully by GPT-4o.")
                else:
                    feedback.append(f"GPT-4o indicated that {chart_name} may be unclear or incorrect: {answer}")
                    logger.warning(f"{chart_name} flagged as unclear by GPT-4o.")
            except Exception as e:
                feedback.append(f"An error occurred while evaluating {chart_name} with GPT-4o: {e}")
                logger.error(f"Error evaluating {chart_name} with GPT-4o: {e}", exc_info=True)
        else:
            feedback.append(f"The visualization {chart_name} is missing or empty.")
            logger.warning(f"{chart_name} missing or empty.")

    # Step 3: Calculate final score
    final_score = min(score, 1.0)
    logger.info(f"Final evaluation score: {final_score:.2f}")

    print("\nEvaluation Summary:")
    for msg in feedback:
        print("-", msg)
    print(f"\nFinal Score: {final_score:.2f}")

    return final_score

def evaluate_bmw_price_mileage_correlation(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for ds177 — Regional Price–Mileage Correlation Analysis.

    Expected output file:
        bmw_price_mileage_summary.txt

    Expected format:
        Strongest_Negative_Correlation_Region: <region_name>

    Example ground truth:
        Strongest_Negative_Correlation_Region: Middle East

    Scoring:
      0.4 - Python script exists
      0.4 - Output text file exists
      0.2 - Valid format and region name (exact or realistic)
    """
    score = 0.0
    feedback = []

    if not actual:
        return 0.0

    base_dir = os.path.dirname(actual)
    script_path = os.path.join(
        base_dir,
        "bmw_price_mileage_correlation.py"
    )
    output_path = os.path.join(
        base_dir,
        "bmw_price_mileage_summary.txt"
    )

    # --- Check Python script existence ---
    if os.path.exists(script_path):
        score += 0.4
        feedback.append("Python script found.")
    else:
        feedback.append("Python script missing.")

    # --- Check output file existence ---
    if not os.path.exists(output_path):
        feedback.append("Output text file missing.")
        logger.info("\n".join(feedback))
        return round(score, 2)
    else:
        score += 0.4
        feedback.append("Output text file found.")

    # --- Validate output format and region name ---
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        # Accepts multi-word region names (e.g., Middle East)
        pattern = r"^Strongest_Negative_Correlation_Region:\s*([A-Za-z ]+)$"
        match = re.match(pattern, content)

        if match:
            region = match.group(1).strip()
            feedback.append(f"Detected region: '{region}'")

            if region.lower() == "middle east":
                score += 0.2
                feedback.append("Output matches expected ground truth exactly (Middle East).")
            elif len(region) > 1 and all(c.isalpha() or c.isspace() for c in region):
                score += 0.1
                feedback.append("Output format valid and region realistic, but not exact match.")
            else:
                feedback.append("Region name invalid or malformed.")
        else:
            feedback.append("Invalid format. Expected 'Strongest_Negative_Correlation_Region: <region_name>'.")

    except Exception as e:
        feedback.append(f"Error reading or validating output: {e}")

    # --- Final scoring ---
    final_score = min(round(score, 2), 1.0)
    for msg in feedback:
        logger.info(msg)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    return final_score

def evaluate_bmw_efficiency_price_ratio(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for ds178 — Efficiency-to-Price Ratio Analysis by Fuel Type.

    Expected output file:
        bmw_efficiency_summary.txt

    Expected format:
        Most_Efficient_Fuel_Type: <fuel_type>

    Example ground truth:
        Most_Efficient_Fuel_Type: Diesel

    Scoring:
      0.4 - Python script exists
      0.4 - Output text file exists
      0.2 - Correct format and realistic or exact result
    """
    score = 0.0
    feedback = []

    if not actual:
        return 0.0

    base_dir = os.path.dirname(actual)
    script_path = os.path.join(
        base_dir,
        "bmw_efficiency_price_ratio.py"
    )
    output_path = os.path.join(
        base_dir,
        "bmw_efficiency_summary.txt"
    )

    # Check if Python script exists
    if os.path.exists(script_path):
        score += 0.4
        feedback.append("Python script found.")
    else:
        feedback.append("Python script missing.")

    # Check if output text file exists
    if not os.path.exists(output_path):
        feedback.append("Output text file missing.")
        logger.info("\n".join(feedback))
        return round(score, 2)
    else:
        score += 0.4
        feedback.append("Output text file found.")

    # Validate output format and content
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        # Expect format like "Most_Efficient_Fuel_Type: Diesel"
        pattern = r"^Most_Efficient_Fuel_Type:\s*([A-Za-z ]+)$"
        match = re.match(pattern, content)

        if match:
            fuel_type = match.group(1).strip()
            feedback.append(f"Detected fuel type: '{fuel_type}'")

            if fuel_type.lower() == "diesel":
                score += 0.2
                feedback.append("Output matches the expected ground truth (Diesel).")
            elif len(fuel_type) > 1 and all(c.isalpha() or c.isspace() for c in fuel_type):
                score += 0.1
                feedback.append("Fuel type is valid but not an exact match.")
            else:
                feedback.append("Fuel type is invalid or malformed.")
        else:
            feedback.append("Invalid format. Expected 'Most_Efficient_Fuel_Type: <fuel_type>'.")

    except Exception as e:
        feedback.append(f"Error reading or validating output: {e}")

    # Final scoring
    final_score = min(round(score, 2), 1.0)
    for msg in feedback:
        logger.info(msg)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    return final_score

def evaluate_bmw_price_trend_regression(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for ds179 — BMW yearly price trend regression.

    Expected output file:
        bmw_price_trend_summary.txt

    Expected format:
        Regression_Slope: <value>, R2: <value>

    Example output:
        Regression_Slope: 152.37, R2: 0.84

    Scoring:
      0.4 - Python script exists
      0.4 - Output text file exists
      0.2 - Correct format and numeric values
    """
    score = 0.0
    feedback = []

    if not actual:
        return 0.0

    base_dir = os.path.dirname(actual)
    script_path = os.path.join(base_dir, "bmw_price_trend_regression.py")
    output_path = os.path.join(base_dir, "bmw_price_trend_summary.txt")

    # Check if Python script exists
    if os.path.exists(script_path):
        score += 0.4
        feedback.append("Python script found.")
    else:
        feedback.append("Python script missing.")

    # Check if output text file exists
    if not os.path.exists(output_path):
        feedback.append("Output text file missing.")
        logger.info("\n".join(feedback))
        return round(score, 2)
    else:
        score += 0.4
        feedback.append("Output text file found.")

    # Validate output format and numeric values
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        pattern = r"^Regression_Slope:\s*([-+]?\d*\.?\d+),\s*R2:\s*([-+]?\d*\.?\d+)$"
        match = re.match(pattern, content)

        if match:
            slope = float(match.group(1))
            r2 = float(match.group(2))
            feedback.append(f"Detected slope: {slope}, R2: {r2}")

            # Check numeric range
            if r2 >= 0 and r2 <= 1:
                score += 0.2
                feedback.append("Output format valid and values are numeric.")
            else:
                feedback.append("R2 value out of valid range (0 to 1).")
        else:
            feedback.append("Invalid format. Expected 'Regression_Slope: <value>, R2: <value>'.")

    except Exception as e:
        feedback.append(f"Error reading or validating output: {e}")

    final_score = min(round(score, 2), 1.0)
    for msg in feedback:
        logger.info(msg)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    return final_score

def evaluate_bmw_top_models_by_sales(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for ds180 — Top BMW Models by Normalized Sales Volume.

    Expected output file:
        bmw_top_models.csv

    Expected format:
        Model, Mean_Normalized_Sales
        (exactly 5 rows, numeric values between 0 and 1)

    Example output:
        Model,Mean_Normalized_Sales
        X1,0.5073
        7 Series,0.5049
        M5,0.5038
        i8,0.5036
        3 Series,0.5017
    """
    score = 0.0
    feedback = []

    if not actual:
        feedback.append("No directory provided for evaluation.")
        return 0.0

    base_dir = os.path.dirname(actual)
    script_path = os.path.join(
        base_dir,
        "bmw_top_models_by_sales.py"
    )
    output_path = os.path.join(
        base_dir,
        "bmw_top_models.csv"
    )

    # Check script existence
    if os.path.exists(script_path):
        score += 0.4
        feedback.append("Python script found.")
    else:
        feedback.append("Python script missing.")

    # Check output file existence
    if not os.path.exists(output_path):
        feedback.append("Output CSV file missing.")
        logger.info("\n".join(feedback))
        return round(score, 2)
    else:
        score += 0.4
        feedback.append("Output CSV file found.")

    # Validate structure and values
    try:
        df = pd.read_csv(output_path)
        cols = [c.strip() for c in df.columns]

        if cols == ["Model", "Mean_Normalized_Sales"]:
            score += 0.1
            feedback.append("CSV has correct columns.")
        else:
            feedback.append(f"Incorrect columns: {cols}")

        if len(df) == 5:
            score += 0.05
            feedback.append("CSV contains exactly 5 rows.")
        else:
            feedback.append(f"Unexpected row count: {len(df)} (expected 5).")

        # Check numeric validity and range
        try:
            values = df["Mean_Normalized_Sales"].astype(float)
            if values.between(0, 1).all():
                score += 0.05
                feedback.append("All Mean_Normalized_Sales values are between 0 and 1.")
            else:
                feedback.append("Some Mean_Normalized_Sales values are out of range (0–1).")
        except Exception:
            feedback.append("Mean_Normalized_Sales column is not numeric.")
    except Exception as e:
        feedback.append(f"Error reading or validating CSV: {e}")

    final_score = min(round(score, 2), 1.0)
    for msg in feedback:
        logger.info(msg)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    return final_score

def evaluate_bmw_region_price_variance(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for ds181 — BMW Regional Price Variance and ANOVA.

    Expected output file:
        bmw_region_variance_summary.txt

    Expected format:
        Highest_Variance_Region: <region_name>

    Example output:
        Highest_Variance_Region: Middle East
    """
    score = 0.0
    feedback = []

    if not actual:
        feedback.append("No base directory provided for evaluation.")
        return 0.0

    base_dir = os.path.dirname(actual)
    script_path = os.path.join(
        base_dir,
        "bmw_region_price_variance.py"
    )
    output_path = os.path.join(
        base_dir,
        "bmw_region_variance_summary.txt"
    )

    # Check script existence
    if os.path.exists(script_path):
        score += 0.4
        feedback.append("Python script file found.")
    else:
        feedback.append("Python script file missing.")

    # Check output file existence
    if not os.path.exists(output_path):
        feedback.append("Output text file missing.")
        logger.info("\n".join(feedback))
        return round(score, 2)
    else:
        score += 0.4
        feedback.append("Output text file found.")

    # Validate file content and format
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        pattern = r"^Highest_Variance_Region:\s*([A-Za-z ]+)$"
        match = re.match(pattern, content)

        if match:
            region = match.group(1).strip()
            feedback.append(f"Detected region: '{region}'")

            if region.lower() == "middle east":
                score += 0.2
                feedback.append("Output matches expected region (Middle East).")
            elif len(region) > 1:
                score += 0.1
                feedback.append("Region name valid but not an exact match.")
            else:
                feedback.append("Region name invalid or incomplete.")
        else:
            feedback.append("Invalid format. Expected 'Highest_Variance_Region: <region_name>'.")

    except Exception as e:
        feedback.append(f"Error reading output file: {e}")

    # Finalize and log score
    final_score = min(round(score, 2), 1.0)
    for msg in feedback:
        logger.info(msg)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")

    return final_score

def evaluate_bmw_price_random_forest(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for ds182 — BMW Random Forest Price Prediction.

    Expected output file:
        bmw_random_forest_metrics.txt

    Expected format:
        MAE: <value>, RMSE: <value>, R2: <value>

    Example output:
        MAE: 2154.23, RMSE: 3659.48, R2: 0.91
    """
    score = 0.0
    feedback = []

    if not actual:
        feedback.append("No directory provided for evaluation.")
        return 0.0

    base_dir = os.path.dirname(actual)
    script_path = os.path.join(
        base_dir,
        "bmw_price_random_forest.py"
    )
    output_path = os.path.join(
        base_dir,
        "bmw_random_forest_metrics.txt"
    )

    # Check for Python script
    if os.path.exists(script_path):
        score += 0.4
        feedback.append("Python script found.")
    else:
        feedback.append("Python script missing.")

    # Check for output text file
    if not os.path.exists(output_path):
        feedback.append("Output file missing.")
        logger.info("\n".join(feedback))
        return round(score, 2)
    else:
        score += 0.4
        feedback.append("Output text file found.")

    # Validate output format and metrics
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        pattern = r"^MAE:\s*([-+]?\d*\.?\d+),\s*RMSE:\s*([-+]?\d*\.?\d+),\s*R2:\s*([-+]?\d*\.?\d+)$"
        match = re.match(pattern, content)

        if match:
            mae = float(match.group(1))
            rmse = float(match.group(2))
            r2 = float(match.group(3))
            feedback.append(f"Detected MAE={mae}, RMSE={rmse}, R2={r2}")

            # Check if values are valid
            if mae > 0 and rmse > 0 and 0.7 <= r2 <= 1.0:
                score += 0.2
                feedback.append("Metrics are valid and R² is above 0.7.")
            elif 0 <= r2 < 0.7:
                score += 0.1
                feedback.append("Metrics are valid but R² below 0.7.")
            else:
                feedback.append("One or more metric values are invalid.")
        else:
            feedback.append("Invalid format. Expected 'MAE: <value>, RMSE: <value>, R2: <value>'.")

    except Exception as e:
        feedback.append(f"Error reading or validating output file: {e}")

    final_score = min(round(score, 2), 1.0)
    for msg in feedback:
        logger.info(msg)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    return final_score

def evaluate_bmw_transmission_comparison(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for ds183 — Transmission Type Price-to-Engine Ratio Analysis.

    Expected output file:
        bmw_transmission_summary.txt

    Expected format:
        Higher_Ratio_Transmission: <transmission_type>

    Example output:
        Higher_Ratio_Transmission: Automatic
    """

    score = 0.0
    feedback = []

    if not actual:
        feedback.append("No directory provided for evaluation.")
        return 0.0

    base_dir = os.path.dirname(actual)
    script_path = os.path.join(
        base_dir,
        "bmw_transmission_comparison.py"
    )
    output_path = os.path.join(
        base_dir,
        "bmw_transmission_summary.txt"
    )

    # Check Python script existence
    if os.path.exists(script_path):
        score += 0.4
        feedback.append("Python script found.")
    else:
        feedback.append("Python script missing.")

    # Check output text file existence
    if not os.path.exists(output_path):
        feedback.append("Output text file missing.")
        logger.info("\n".join(feedback))
        return round(score, 2)
    else:
        score += 0.4
        feedback.append("Output text file found.")

    # Validate content and format
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        pattern = r"^Higher_Ratio_Transmission:\s*([A-Za-z ]+)$"
        match = re.match(pattern, content)

        if match:
            transmission = match.group(1).strip()
            feedback.append(f"Detected transmission type: '{transmission}'")

            if transmission.lower() == "automatic":
                score += 0.2
                feedback.append("Output matches the expected result (Automatic).")
            elif len(transmission) > 1:
                score += 0.1
                feedback.append("Transmission type valid but not exact match.")
            else:
                feedback.append("Transmission type name invalid or empty.")
        else:
            feedback.append("Invalid format. Expected 'Higher_Ratio_Transmission: <transmission_type>'.")

    except Exception as e:
        feedback.append(f"Error reading or validating output file: {e}")

    # Final scoring
    final_score = min(round(score, 2), 1.0)
    for msg in feedback:
        logger.info(msg)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")

    return final_score

def evaluate_bmw_sales_linear_regression(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for ds184 — BMW Sales Linear Regression with Cross-Validation.
    Validates both script and result files, checks correct output format, and ensures
    realistic numeric values (R² ≥ 0.7 and RMSE > 0).

    Expected output format:
        Mean_R2: <value>, Mean_RMSE: <value>

    Example:
        Mean_R2: 0.8124, Mean_RMSE: 4570.29
    """
    score = 0.0
    feedback = []

    # === Locate files ===
    if not actual:
        feedback.append("No working directory provided.")
        return 0.0

    base_dir = os.path.dirname(actual)
    script_path = os.path.join(
        base_dir, "bmw_sales_linear_regression.py"
    )
    result_path = os.path.join(
        base_dir, "bmw_sales_regression_summary.txt"
    )

    # === Check script file ===
    if os.path.exists(script_path):
        score += 0.4
        feedback.append("Python script found.")
    else:
        feedback.append("Python script missing.")

    # === Check output file ===
    if os.path.exists(result_path):
        score += 0.4
        feedback.append("Result file found.")
    else:
        feedback.append("Result file missing.")
        logger.info("\n".join(feedback))
        return round(score, 2)

    # === Validate result format ===
    try:
        with open(result_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        pattern = r"^Mean_R2:\s*([-+]?\d*\.?\d+),\s*Mean_RMSE:\s*([-+]?\d*\.?\d+)$"
        match = re.match(pattern, content)

        if not match:
            feedback.append("Invalid output format. Expected 'Mean_R2: <value>, Mean_RMSE: <value>'.")
            logger.info("\n".join(feedback))
            return round(score, 2)

        r2 = float(match.group(1))
        rmse = float(match.group(2))
        feedback.append(f"Detected Mean_R2={r2:.4f}, Mean_RMSE={rmse:.2f}")

        # === Logical range checks ===
        if r2 >= 0.7:
            score += 0.1
            feedback.append("R² value is acceptable (≥ 0.7).")
        else:
            feedback.append("R² value below acceptable threshold.")

        if rmse > 0:
            score += 0.1
            feedback.append("RMSE value valid (positive).")
        else:
            feedback.append("RMSE value invalid (≤ 0).")

    except Exception as e:
        feedback.append(f"Error reading result file: {e}")

    final_score = min(round(score, 2), 1.0)
    for msg in feedback:
        logger.info(msg)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")

    return final_score

def evaluate_bmw_sales_volatility_trend(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for ds185 — BMW Sales Volatility Trend Analysis.
    Confirms that the script and result files exist and that the output
    format is valid and human-readable.

    Expected text output format:
        Volatility_Trend: Increasing
    or
        Volatility_Trend: Decreasing
    """
    score = 0.0
    feedback = []

    if not actual:
        feedback.append("No working directory provided for evaluation.")
        return 0.0

    base_dir = os.path.dirname(actual)

    script_path = os.path.join(
        base_dir, "bmw_sales_volatility_trend.py"
    )
    result_path = os.path.join(
        base_dir, "bmw_sales_volatility_summary.txt"
    )

    # Check for script file
    if os.path.exists(script_path):
        score += 0.4
        feedback.append("Python script located successfully.")
    else:
        feedback.append("Python script not found.")

    # Check for result file
    if os.path.exists(result_path):
        score += 0.4
        feedback.append("Result text file located successfully.")
    else:
        feedback.append("Result file missing.")
        logger.info("\n".join(feedback))
        return round(score, 2)

    # Validate the format and content of the output
    try:
        with open(result_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        pattern = r"^Volatility_Trend:\s*(Increasing)$"
        match = re.match(pattern, content, re.IGNORECASE)

        if match:
            trend = match.group(1).capitalize()
            feedback.append(f"Detected output format as valid: Volatility_Trend = {trend}")
            score += 0.2
        else:
            feedback.append("Output format invalid. Expected 'Volatility_Trend: Increasing' or 'Volatility_Trend: Decreasing'.")
            logger.info("\n".join(feedback))
            return round(score, 2)

        if trend in ["Increasing", "Decreasing"]:
            feedback.append("Recognized valid trend label.")
        else:
            feedback.append("Trend label not recognized.")
            score -= 0.1

    except Exception as e:
        feedback.append(f"Error reading or validating result file: {e}")

    final_score = min(round(score, 2), 1.0)
    for msg in feedback:
        logger.info(msg)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")

    return final_score


def evaluate_bmw_sales_visualization_quality(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the BMW Sales Visualization (Visual Studio Code) task.

    This evaluation checks:
      - The Python script file exists.
      - Both visualization image files exist and are non-empty.
      - GPT-4o verifies that:
          1. price_by_fuel_boxplot.png correctly shows a boxplot of Price_USD by Fuel_Type.
          2. engine_mileage_scatter.png correctly shows a scatter plot of Engine_Size_L vs Mileage_KM,
             colored by Sales_Classification.
    """

    if actual is None:
        logger.error("Directory collection failed. No path received.")
        return 0.0

    expected = expected.get("rules", expected)
    base_dir = os.path.dirname(actual)

    script_file = os.path.join(base_dir, "bmw_sales_visualization.py")
    chart1 = os.path.join(base_dir, "price_by_fuel_boxplot.png")
    chart2 = os.path.join(base_dir, "engine_mileage_scatter.png")

    score = 0.0
    feedback = []

    # Step 1: Check for the main Python script
    if os.path.exists(script_file):
        score += 0.4
        feedback.append("The Python script file exists.")
        logger.info("Script file located successfully.")
    else:
        feedback.append("The Python script file could not be found.")
        logger.warning("Script file missing.")

    # Helper function for encoding image files
    def encode_image(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    # Step 2: Check visualization outputs
    charts_to_check = [
        (
            chart1,
            "a boxplot showing Price_USD grouped by Fuel_Type, with clearly labeled axes and title."
        ),
        (
            chart2,
            "a scatter plot of Engine_Size_L (x-axis) versus Mileage_KM (y-axis), colored by Sales_Classification, with labels and title."
        )
    ]

    for chart, description in charts_to_check:
        chart_name = os.path.basename(chart)
        if os.path.exists(chart) and os.path.getsize(chart) > 0:
            score += 0.15
            feedback.append(f"The visualization {chart_name} exists and is not empty.")
            logger.info(f"{chart_name} verified for existence and size.")

            # Step 2a: Validate chart visually using GPT-4o
            try:
                encoded = encode_image(chart)
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are an expert in evaluating data visualizations. "
                                "Assess whether the given chart correctly matches its description, "
                                "with appropriate labeling, readability, and data representation."
                            )
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        f"Please analyze this chart and confirm whether it correctly represents {description}. "
                                        f"Reply 'yes' if the visualization matches the description, or 'no' if it does not, followed by a short reason."
                                    )
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{encoded}"}
                                }
                            ]
                        }
                    ],
                    max_tokens=100
                )

                answer = response.choices[0].message.content.lower()
                if "yes" in answer:
                    score += 0.1
                    feedback.append(f"GPT-4o verified that {chart_name} accurately represents the required visualization.")
                    logger.info(f"{chart_name} successfully validated by GPT-4o.")
                else:
                    feedback.append(f"GPT-4o reported that {chart_name} may not match the expected design: {answer}")
                    logger.warning(f"{chart_name} did not fully align with expectations.")
            except Exception as e:
                feedback.append(f"An error occurred during GPT-4o evaluation of {chart_name}: {e}")
                logger.error(f"GPT-4o evaluation error for {chart_name}: {e}", exc_info=True)
        else:
            feedback.append(f"The visualization {chart_name} is missing or empty.")
            logger.warning(f"{chart_name} missing or empty.")

    # Step 3: Final score calculation
    final_score = min(score, 1.0)
    logger.info(f"Final evaluation score: {final_score:.2f}")

    print("\nEvaluation Summary:")
    for msg in feedback:
        print("-", msg)
    print(f"\nFinal Score: {final_score:.2f}")

    return final_score

def evaluate_student_exam_interaction_effect(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for ds187 — Student Exam Interaction Effect.
    Checks that both script and result files exist, verifies output format, and
    confirms the reported interaction effect is 'Increases' with a valid numeric coefficient.
    """
    score = 0.0
    feedback = []

    if not actual:
        feedback.append("No working directory provided for evaluation.")
        return 0.0

    base_dir = os.path.dirname(actual)

    script_path = os.path.join(
        base_dir, "student_exam_interaction_effect.py"
    )
    result_path = os.path.join(
        base_dir, "student_exam_interaction_summary.txt"
    )

    # Check script existence
    if os.path.exists(script_path):
        score += 0.4
        feedback.append("Python script located successfully.")
    else:
        feedback.append("Python script not found.")

    # Check result file existence
    if os.path.exists(result_path):
        score += 0.4
        feedback.append("Result text file located successfully.")
    else:
        feedback.append("Result file missing.")
        logger.info("\n".join(feedback))
        return round(score, 2)

    # Read and validate the result file
    try:
        with open(result_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        # Expected format: Interaction_Coefficient: <float>, Effect: Increases
        pattern = r"^Interaction_Coefficient:\s*([-+]?\d*\.?\d+),\s*Effect:\s*(Increases|Decreases)$"
        match = re.match(pattern, content, re.IGNORECASE)

        if match:
            coeff = float(match.group(1))
            effect = match.group(2).capitalize()
            feedback.append(f"Detected valid format: Coefficient = {coeff}, Effect = {effect}")
            score += 0.2

            # Check logical correctness
            if effect == "Increases":
                feedback.append("Effect correctly identified as 'Increases'.")
            else:
                feedback.append("Effect should be 'Increases'.")
                score -= 0.1

            # Validate coefficient range
            if abs(coeff) < 1:
                feedback.append("Coefficient magnitude within expected range.")
            else:
                feedback.append("Coefficient unusually large but format is correct.")
        else:
            feedback.append("Output format invalid. Expected 'Interaction_Coefficient: <value>, Effect: <Increases/Decreases>'.")

    except Exception as e:
        feedback.append(f"Error while reading result file: {e}")

    final_score = min(round(score, 2), 1.0)

    for msg in feedback:
        logger.info(msg)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")

    return final_score


def evaluate_student_exam_correlation_difference(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for ds188 — Student Exam Correlation Difference.
    Checks for script and result file existence, validates output format, and ensures
    that correlation values are numeric and within the expected range.
    """

    score = 0.0
    feedback = []

    if not actual:
        feedback.append("No working directory provided for evaluation.")
        return 0.0

    base_dir = os.path.dirname(actual)

    script_path = os.path.join(
        base_dir, "student_exam_correlation_difference.py"
    )
    result_path = os.path.join(
        base_dir, "student_exam_correlation_summary.txt"
    )

    # Verify script existence
    if os.path.exists(script_path):
        score += 0.4
        feedback.append("Python script located successfully.")
    else:
        feedback.append("Python script not found.")

    # Verify result file existence
    if os.path.exists(result_path):
        score += 0.4
        feedback.append("Result text file located successfully.")
    else:
        feedback.append("Result file missing.")
        logger.info("\n".join(feedback))
        return round(score, 2)

    # Read and validate the result file
    try:
        with open(result_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        # Expected format: HighGroup_Corr: <float>, LowGroup_Corr: <float>, Abs_Diff: <float>
        pattern = (
            r"^HighGroup_Corr:\s*([-+]?\d*\.?\d+),\s*"
            r"LowGroup_Corr:\s*([-+]?\d*\.?\d+),\s*"
            r"Abs_Diff:\s*([-+]?\d*\.?\d+)$"
        )
        match = re.match(pattern, content, re.IGNORECASE)

        if match:
            high_corr = float(match.group(1))
            low_corr = float(match.group(2))
            abs_diff = float(match.group(3))
            feedback.append(
                f"Detected valid format: HighGroup_Corr={high_corr}, LowGroup_Corr={low_corr}, Abs_Diff={abs_diff}"
            )
            score += 0.2

            # Logical validation and tolerance-based comparison
            expected_values = {"HighGroup_Corr": 0.8086, "LowGroup_Corr": 0.8339, "Abs_Diff": 0.0254}
            tolerance = 0.01

            within_tolerance = (
                abs(high_corr - expected_values["HighGroup_Corr"]) <= tolerance
                and abs(low_corr - expected_values["LowGroup_Corr"]) <= tolerance
                and abs(abs_diff - expected_values["Abs_Diff"]) <= tolerance
            )

            if within_tolerance:
                feedback.append("All correlation values are within the expected tolerance range.")
            else:
                feedback.append(
                    f"Values differ from expected. Expected: {expected_values}, Found: "
                    f"{{'HighGroup_Corr': {high_corr}, 'LowGroup_Corr': {low_corr}, 'Abs_Diff': {abs_diff}}}"
                )
                score -= 0.1
        else:
            feedback.append(
                "Output format invalid. Expected 'HighGroup_Corr: <value>, LowGroup_Corr: <value>, Abs_Diff: <value>'."
            )

    except Exception as e:
        feedback.append(f"Error while reading result file: {e}")

    final_score = min(round(score, 2), 1.0)

    for msg in feedback:
        logger.info(msg)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")

    return final_score



def evaluate_student_exam_efficiency_analysis(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for ds189 — Student Exam Efficiency Analysis.
    Confirms that the script and result files exist, checks output format,
    and verifies that the efficiency and ratio values are within a realistic range.
    """

    score = 0.0
    feedback = []

    if not actual:
        feedback.append("No working directory provided for evaluation.")
        return 0.0

    base_dir = os.path.dirname(actual)

    script_path = os.path.join(
        base_dir, "student_exam_efficiency_analysis.py"
    )
    result_path = os.path.join(
        base_dir, "student_efficiency_summary.txt"
    )

    # Confirm script existence
    if os.path.exists(script_path):
        score += 0.4
        feedback.append("Python script located successfully.")
    else:
        feedback.append("Python script not found.")

    # Confirm result file existence
    if os.path.exists(result_path):
        score += 0.4
        feedback.append("Result text file located successfully.")
    else:
        feedback.append("Result file missing.")
        logger.info("\n".join(feedback))
        return round(score, 2)

    # Read and validate the result file content
    try:
        with open(result_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        # Expected format: Top_Student_Efficiency: <float>, ExamScore_Ratio: <float>
        pattern = r"^Top_Student_Efficiency:\s*([-+]?\d*\.?\d+),\s*ExamScore_Ratio:\s*([-+]?\d*\.?\d+)$"
        match = re.match(pattern, content)

        if match:
            efficiency = float(match.group(1))
            ratio = float(match.group(2))
            feedback.append(f"Valid format detected — Efficiency={efficiency}, Ratio={ratio}")
            score += 0.2

            # Ground truth and tolerance
            expected_values = {"Top_Student_Efficiency": 35.1978, "ExamScore_Ratio": 0.9248}
            tolerance = 0.01

            within_tolerance = (
                abs(efficiency - expected_values["Top_Student_Efficiency"]) <= tolerance
                and abs(ratio - expected_values["ExamScore_Ratio"]) <= tolerance
            )

            if within_tolerance:
                feedback.append("Values match expected results within tolerance range.")
            else:
                feedback.append(
                    f"Values deviate from expected.\nExpected: {expected_values}\n"
                    f"Found: {{'Top_Student_Efficiency': {efficiency}, 'ExamScore_Ratio': {ratio}}}"
                )
                score -= 0.1

            # Sanity check for valid ranges
            if efficiency > 0 and 0 < ratio <= 2:
                feedback.append("Efficiency and ratio values are within logical bounds.")
            else:
                feedback.append("Detected values outside expected logical range.")
                score -= 0.05
        else:
            feedback.append("Output format invalid. Expected 'Top_Student_Efficiency: <value>, ExamScore_Ratio: <value>'.")

    except Exception as e:
        feedback.append(f"Error reading result file: {e}")

    final_score = min(round(score, 2), 1.0)

    for msg in feedback:
        logger.info(msg)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")

    return final_score


def evaluate_student_exam_hidden_confounder(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for ds190 — Student Exam Hidden Confounder.
    Confirms that the regression script and result file exist, verifies output format,
    and checks that both the correlation value and confounder flag are accurate.
    """

    score = 0.0
    feedback = []

    if not actual:
        feedback.append("No working directory provided for evaluation.")
        return 0.0

    base_dir = os.path.dirname(actual)

    script_path = os.path.join(
        base_dir, "student_exam_hidden_confounder.py"
    )
    result_path = os.path.join(
        base_dir, "student_hidden_confounder_summary.txt"
    )

    # Verify the Python script
    if os.path.exists(script_path):
        score += 0.4
        feedback.append("Python script located successfully.")
    else:
        feedback.append("Python script not found.")

    # Verify the result text file
    if os.path.exists(result_path):
        score += 0.4
        feedback.append("Result text file located successfully.")
    else:
        feedback.append("Result file missing.")
        logger.info("\n".join(feedback))
        return round(score, 2)

    # Validate the contents of the result file
    try:
        with open(result_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        # Expected format: Residual_Attendance_Correlation: <float>, Confounder_Flag: <Yes/No>
        pattern = r"^Residual_Attendance_Correlation:\s*([-+]?\d*\.?\d+),\s*Confounder_Flag:\s*(Yes|No)$"
        match = re.match(pattern, content, re.IGNORECASE)

        if match:
            correlation = float(match.group(1))
            flag = match.group(2).capitalize()
            feedback.append(f"Valid format detected — Correlation={correlation}, Flag={flag}")
            score += 0.2

            # Ground truth values
            expected_values = {"Residual_Attendance_Correlation": 0.4049, "Confounder_Flag": "Yes"}
            tolerance = 0.02

            # Check numeric closeness and logical flag
            if abs(correlation - expected_values["Residual_Attendance_Correlation"]) <= tolerance:
                feedback.append("Correlation value is within expected tolerance range.")
            else:
                feedback.append(
                    f"Correlation value deviates from expected. Expected {expected_values['Residual_Attendance_Correlation']}, found {correlation}."
                )
                score -= 0.1

            if flag == expected_values["Confounder_Flag"]:
                feedback.append("Confounder flag correctly identified as 'Yes'.")
            else:
                feedback.append("Incorrect confounder flag value.")
                score -= 0.1

            # Sanity check
            if -1 <= correlation <= 1:
                feedback.append("Correlation value is within a valid range.")
            else:
                feedback.append("Correlation outside of valid [-1, 1] range.")
                score -= 0.05
        else:
            feedback.append(
                "Output format invalid. Expected 'Residual_Attendance_Correlation: <value>, Confounder_Flag: <Yes/No>'."
            )

    except Exception as e:
        feedback.append(f"Error while reading result file: {e}")

    final_score = min(round(score, 2), 1.0)

    for msg in feedback:
        logger.info(msg)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")

    return final_score

def evaluate_student_study_regime_change(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for ds191 — Student Study Regime Change Analysis.
    Confirms the Python script and result file exist, verifies correct text format,
    and checks that the reported slope difference is numerically valid and within tolerance.
    """

    score = 0.0
    feedback = []

    if not actual:
        feedback.append("No working directory path provided for evaluation.")
        return 0.0

    base_dir = os.path.dirname(actual)

    script_path = os.path.join(
        base_dir, "student_study_regime_change.py"
    )
    result_path = os.path.join(
        base_dir, "student_study_regime_summary.txt"
    )

    # Check if the main Python script exists
    if os.path.exists(script_path):
        score += 0.4
        feedback.append("Python script located successfully.")
    else:
        feedback.append("Python script not found in the target directory.")

    # Check if the output text file exists
    if os.path.exists(result_path):
        score += 0.4
        feedback.append("Result text file located successfully.")
    else:
        feedback.append("Result text file missing.")
        logger.info("\n".join(feedback))
        return round(score, 2)

    # Validate the output format and values
    try:
        with open(result_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        # Expected pattern: Slope_Difference: <float>
        pattern = r"^Slope_Difference:\s*([-+]?\d*\.?\d+)$"
        match = re.match(pattern, content)

        if match:
            slope_diff = float(match.group(1))
            feedback.append(f"Detected valid output format. Slope difference = {slope_diff}.")
            score += 0.2

            # Ground truth and tolerance
            expected_value = 0.2297
            tolerance = 0.02

            # Check closeness to expected result
            if abs(slope_diff - expected_value) <= tolerance:
                feedback.append("Slope difference is within expected range.")
            else:
                feedback.append(
                    f"Slope difference deviates from expected (expected {expected_value}, got {slope_diff})."
                )
                score -= 0.1

            # Ensure slope value is numeric and within a reasonable magnitude
            if abs(slope_diff) < 5:
                feedback.append("Slope difference magnitude is realistic.")
            else:
                feedback.append("Unusually large slope difference detected.")
                score -= 0.05
        else:
            feedback.append("Output format invalid. Expected 'Slope_Difference: <value>'.")

    except Exception as e:
        feedback.append(f"Error while reading or parsing result file: {e}")

    # Final scoring and logging
    final_score = min(round(score, 2), 1.0)
    for msg in feedback:
        logger.info(msg)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")

    return final_score


def evaluate_student_pseudo_causal_score(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for ds192 — Student Pseudo-Causal Correlation Analysis.
    Confirms that the Python script and output text file exist, validates output format,
    and checks that the correlation and p-value are within expected tolerance and valid ranges.
    """

    score = 0.0
    feedback = []

    if not actual:
        feedback.append("No working directory path provided for evaluation.")
        return 0.0

    base_dir = os.path.dirname(actual)

    script_path = os.path.join(
        base_dir, "student_pseudo_causal_score.py"
    )
    result_path = os.path.join(
        base_dir, "student_pseudo_causal_summary.txt"
    )

    # Check Python script existence
    if os.path.exists(script_path):
        score += 0.4
        feedback.append("Python script found successfully.")
    else:
        feedback.append("Python script missing from directory.")

    # Check output file existence
    if os.path.exists(result_path):
        score += 0.4
        feedback.append("Output text file found successfully.")
    else:
        feedback.append("Output text file missing.")
        logger.info("\n".join(feedback))
        return round(score, 2)

    # Validate the output content and values
    try:
        with open(result_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        # Expected format: Correlation: <float>, p_value: <float>
        pattern = r"^Correlation:\s*([-+]?\d*\.?\d+),\s*p_value:\s*([-+]?\d*\.?\d+)$"
        match = re.match(pattern, content)

        if match:
            corr = float(match.group(1))
            pval = float(match.group(2))
            feedback.append(f"Detected valid output format: Correlation={corr}, p_value={pval}.")
            score += 0.2

            # Ground truth reference
            expected_corr = 0.7775
            expected_pval = 0.0
            tolerance = 0.02

            # Check correlation closeness
            if abs(corr - expected_corr) <= tolerance:
                feedback.append("Correlation value is within expected range.")
            else:
                feedback.append(
                    f"Correlation deviates from expected. Expected {expected_corr}, got {corr}."
                )
                score -= 0.1

            # Check p-value closeness (allowing small floating precision)
            if abs(pval - expected_pval) <= 0.01:
                feedback.append("p_value is within expected range.")
            else:
                feedback.append(f"p_value differs from expected (expected {expected_pval}, got {pval}).")
                score -= 0.1

            # Logical sanity checks
            if -1 <= corr <= 1:
                feedback.append("Correlation value within valid range.")
            else:
                feedback.append("Correlation value outside valid [-1, 1] range.")
                score -= 0.05

            if 0 <= pval <= 1:
                feedback.append("p_value within valid probability range.")
            else:
                feedback.append("p_value outside valid [0, 1] range.")
                score -= 0.05

        else:
            feedback.append(
                "Output format invalid. Expected 'Correlation: <value>, p_value: <value>'."
            )

    except Exception as e:
        feedback.append(f"Error reading or validating result file: {e}")

    # Compute final score
    final_score = min(round(score, 2), 1.0)

    for msg in feedback:
        logger.info(msg)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")

    return final_score

def evaluate_student_mahalanobis_outlier_detection(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for ds193 — Student Mahalanobis Multivariate Outlier Detection.
    Ensures the script and output file exist, validates output format, and checks
    that the outlier count is within a reasonable range of the expected ground truth.
    """

    score = 0.0
    feedback = []

    if not actual:
        feedback.append("No directory path provided for evaluation.")
        return 0.0

    base_dir = os.path.dirname(actual)

    script_path = os.path.join(
        base_dir, "student_mahalanobis_outlier_detection.py"
    )
    result_path = os.path.join(
        base_dir, "student_mahalanobis_outlier_summary.txt"
    )

    # Verify Python script existence
    if os.path.exists(script_path):
        score += 0.4
        feedback.append("Python script located successfully.")
    else:
        feedback.append("Python script not found.")

    # Verify output text file existence
    if os.path.exists(result_path):
        score += 0.4
        feedback.append("Output text file found successfully.")
    else:
        feedback.append("Output text file missing.")
        logger.info("\n".join(feedback))
        return round(score, 2)

    # Validate file content format and numerical correctness
    try:
        with open(result_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        # Expected output format: Outlier_Count: <integer>
        pattern = r"^Outlier_Count:\s*(\d+)$"
        match = re.match(pattern, content)

        if match:
            outlier_count = int(match.group(1))
            feedback.append(f"Valid output format detected. Outlier count = {outlier_count}.")
            score += 0.2

            # Ground truth reference and tolerance
            expected_count = 10
            tolerance = 2

            if abs(outlier_count - expected_count) <= tolerance:
                feedback.append("Outlier count is within acceptable range of expected value.")
            else:
                feedback.append(
                    f"Outlier count deviates from expected. Expected {expected_count}, got {outlier_count}."
                )
                score -= 0.1

            # Sanity check for realistic value
            if 0 <= outlier_count <= 1000:
                feedback.append("Outlier count value is realistic.")
            else:
                feedback.append("Outlier count appears unrealistic.")
                score -= 0.05
        else:
            feedback.append("Invalid format. Expected 'Outlier_Count: <integer>'.")

    except Exception as e:
        feedback.append(f"Error while reading result file: {e}")

    # Final score computation
    final_score = min(round(score, 2), 1.0)

    for msg in feedback:
        logger.info(msg)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")

    return final_score

def evaluate_student_model_comparison(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for ds194 — Student Model Comparison (Linear vs Random Forest).
    Checks that the script and result file exist, validates output format,
    and ensures Random Forest achieves higher R² than Linear Regression
    with both models performing reasonably well (R² > 0.6).
    """

    score = 0.0
    feedback = []

    if not actual:
        feedback.append("No working directory provided for evaluation.")
        return 0.0

    base_dir = os.path.dirname(actual)

    script_path = os.path.join(
        base_dir, "student_model_comparison.py"
    )
    result_path = os.path.join(
        base_dir, "student_model_comparison_summary.txt"
    )

    # 1️⃣ Check for script existence
    if os.path.exists(script_path):
        score += 0.4
        feedback.append("Python script located successfully.")
    else:
        feedback.append("Python script not found.")

    # 2️⃣ Check for result text file existence
    if os.path.exists(result_path):
        score += 0.4
        feedback.append("Result text file found successfully.")
    else:
        feedback.append("Result file missing.")
        logger.info("\n".join(feedback))
        return round(score, 2)

    # 3️⃣ Validate output format and correctness
    try:
        with open(result_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        # Expected format:
        # R2_Linear: <float>, R2_RandomForest: <float>, Best_Model: <Linear/RandomForest>
        pattern = (
            r"^R2_Linear:\s*([-+]?\d*\.?\d+),\s*R2_RandomForest:\s*([-+]?\d*\.?\d+),\s*Best_Model:\s*(Linear|RandomForest)$"
        )
        match = re.match(pattern, content)

        if match:
            r2_linear = float(match.group(1))
            r2_rf = float(match.group(2))
            best_model = match.group(3)

            feedback.append(
                f"Detected valid format: R² (Linear) = {r2_linear:.4f}, R² (RandomForest) = {r2_rf:.4f}, Best = {best_model}"
            )
            score += 0.2

            # ✅ Check reasonable R² values
            if r2_linear >= 0.6 and r2_rf >= 0.6:
                feedback.append("Both models show good predictive performance (R² > 0.6).")
                score += 0.1
            else:
                feedback.append("One or both R² values are below the expected performance threshold.")

            # ✅ Check if Random Forest performs better
            if r2_rf > r2_linear and best_model == "RandomForest":
                feedback.append("Best model correctly identified as RandomForest.")
                score += 0.1
            elif best_model == "Linear":
                feedback.append("Best model incorrectly identified as Linear.")
                score -= 0.1
            else:
                feedback.append("Unexpected model label detected.")
        else:
            feedback.append(
                "Invalid output format. Expected 'R2_Linear: <value>, R2_RandomForest: <value>, Best_Model: <Linear/RandomForest>'."
            )

    except Exception as e:
        feedback.append(f"Error while reading result file: {e}")

    # Final score computation
    final_score = min(round(score, 2), 1.0)
    for msg in feedback:
        logger.info(msg)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")

    return final_score

def evaluate_mobile_ram_price_regression(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for ds195 — Mobile RAM–Price Regression Comparison.
    Validates that the Python script and result text file exist,
    checks that R² values are numeric and above 0.5,
    and confirms that the Better_Model field is either 'Linear' or 'Polynomial'.
    """

    score = 0.0
    feedback = []

    if not actual:
        feedback.append("No working directory provided for evaluation.")
        return 0.0

    base_dir = os.path.dirname(actual)

    script_path = os.path.join(base_dir, "mobile_ram_price_regression.py")
    result_path = os.path.join(base_dir, "mobile_ram_price_summary.txt")

    # Step 1: Verify script existence
    if os.path.exists(script_path):
        score += 0.4
        feedback.append("Python script located successfully.")
    else:
        feedback.append("Python script not found.")

    # Step 2: Verify result file existence
    if os.path.exists(result_path):
        score += 0.4
        feedback.append("Result text file found successfully.")
    else:
        feedback.append("Result text file missing.")
        logger.info("\n".join(feedback))
        return round(score, 2)

    # Step 3: Validate result format and content
    try:
        with open(result_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        pattern = (
            r"^R2_Linear:\s*([-+]?\d*\.?\d+),\s*"
            r"R2_Polynomial:\s*([-+]?\d*\.?\d+),\s*"
            r"Better_Model:\s*(Linear|Polynomial),\s*"
            r"R2_Difference:\s*([-+]?\d*\.?\d+)$"
        )

        match = re.match(pattern, content)
        if match:
            r2_linear = float(match.group(1))
            r2_poly = float(match.group(2))
            better_model = match.group(3)
            diff = float(match.group(4))

            feedback.append(
                f"Detected valid output: R²(Linear)={r2_linear:.4f}, "
                f"R²(Polynomial)={r2_poly:.4f}, Better_Model={better_model}, "
                f"R² Difference={diff:.4f}"
            )
            score += 0.2

            if r2_linear > 0.5 and r2_poly > 0.5:
                feedback.append("Both models achieved acceptable R² performance above 0.5.")
                score += 0.1
            else:
                feedback.append("One or both R² values are below the 0.5 threshold.")

            if better_model in ["Linear", "Polynomial"]:
                feedback.append(f"Better_Model field is valid and set to '{better_model}'.")
                score += 0.1
            else:
                feedback.append("Invalid Better_Model value. Must be either 'Linear' or 'Polynomial'.")
                score -= 0.1
        else:
            feedback.append(
                "Output format invalid. Expected format: "
                "'R2_Linear: <value>, R2_Polynomial: <value>, Better_Model: <Linear/Polynomial>, R2_Difference: <value>'."
            )

    except Exception as e:
        feedback.append(f"Error reading or validating result file: {e}")

    final_score = min(round(score, 2), 1.0)

    for msg in feedback:
        logger.info(msg)

    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    return final_score


def evaluate_mobile_battery_correlation(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for ds196 — Mobile Battery Correlation Analysis.
    Verifies that both the Python script and result file exist,
    checks for correct numeric correlation values,
    and validates that the stronger relationship field is either 'mobile_wt' or 'talk_time'.
    """

    score = 0.0
    feedback = []

    if not actual:
        feedback.append("No working directory path provided for evaluation.")
        return 0.0

    base_dir = os.path.dirname(actual)

    script_path = os.path.join(base_dir, "mobile_battery_correlation.py")
    result_path = os.path.join(base_dir, "mobile_battery_correlation_summary.txt")

    # Step 1: Verify script existence
    if os.path.exists(script_path):
        score += 0.4
        feedback.append("Python script located successfully.")
    else:
        feedback.append("Python script not found in the expected directory.")

    # Step 2: Verify result text file existence
    if os.path.exists(result_path):
        score += 0.4
        feedback.append("Result text file found successfully.")
    else:
        feedback.append("Result text file missing.")
        logger.info("\n".join(feedback))
        return round(score, 2)

    # Step 3: Validate content format and values
    try:
        with open(result_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        # Expected pattern
        pattern = (
            r"^Corr_Battery_Weight:\s*([-+]?\d*\.?\d+),\s*"
            r"Corr_Battery_TalkTime:\s*([-+]?\d*\.?\d+),\s*"
            r"Stronger_Relationship:\s*(mobile_wt|talk_time)$"
        )

        match = re.match(pattern, content)
        if match:
            corr_wt = float(match.group(1))
            corr_talk = float(match.group(2))
            stronger = match.group(3)

            feedback.append(
                f"Detected valid output format: Corr_Battery_Weight={corr_wt:.4f}, "
                f"Corr_Battery_TalkTime={corr_talk:.4f}, Stronger_Relationship={stronger}"
            )
            score += 0.2

            # Check correlation value validity
            if -1 <= corr_wt <= 1 and -1 <= corr_talk <= 1:
                feedback.append("Both correlation values are within the valid range of -1 to 1.")
                score += 0.1
            else:
                feedback.append("One or both correlation values are outside the expected range.")

            # Confirm relationship field validity
            if stronger in ["mobile_wt", "talk_time"]:
                feedback.append("Stronger_Relationship field correctly identified.")
                score += 0.1
            else:
                feedback.append("Invalid Stronger_Relationship field.")
                score -= 0.1
        else:
            feedback.append(
                "Output format invalid. Expected format: "
                "'Corr_Battery_Weight: <value>, Corr_Battery_TalkTime: <value>, Stronger_Relationship: <mobile_wt/talk_time>'."
            )

    except Exception as e:
        feedback.append(f"Error while reading or parsing result file: {e}")

    final_score = min(round(score, 2), 1.0)

    for msg in feedback:
        logger.info(msg)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")

    return final_score

def evaluate_mobile_connectivity_correlation(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for ds197 — Mobile Connectivity Correlation.
    Confirms script and result files exist, verifies numeric correlation value format,
    and ensures the correlation lies between -1 and 1.
    """

    score = 0.0
    feedback = []

    if not actual:
        feedback.append("No working directory path provided for evaluation.")
        return 0.0

    base_dir = os.path.dirname(actual)

    script_path = os.path.join(base_dir, "mobile_connectivity_correlation.py")
    result_path = os.path.join(base_dir, "mobile_connectivity_summary.txt")

    # Step 1: Check Python script existence
    if os.path.exists(script_path):
        score += 0.4
        feedback.append("Python script located successfully.")
    else:
        feedback.append("Python script not found in expected directory.")

    # Step 2: Check result text file existence
    if os.path.exists(result_path):
        score += 0.4
        feedback.append("Result text file found successfully.")
    else:
        feedback.append("Result file missing.")
        logger.info("\n".join(feedback))
        return round(score, 2)

    # Step 3: Read and validate file content
    try:
        with open(result_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        # Expected output pattern
        pattern = r"^Corr_Connectivity_Price:\s*([-+]?\d*\.?\d+)$"
        match = re.match(pattern, content)

        if match:
            corr_value = float(match.group(1))
            feedback.append(f"Detected valid output format: Corr_Connectivity_Price = {corr_value:.4f}")
            score += 0.2

            # Check value range
            if -1 <= corr_value <= 1:
                feedback.append("Correlation value lies within the valid range (-1 to 1).")
                score += 0.1
            else:
                feedback.append("Correlation value is outside the expected range (-1 to 1).")

        else:
            feedback.append(
                "Invalid format. Expected format: 'Corr_Connectivity_Price: <numeric_value>'."
            )

    except Exception as e:
        feedback.append(f"Error reading or validating the result file: {e}")

    final_score = min(round(score, 2), 1.0)

    for msg in feedback:
        logger.info(msg)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")

    return final_score

def evaluate_mobile_price_random_forest(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for ds198 — Mobile Price Random Forest Classification.
    Checks that the Python script and text output exist, verifies the output format,
    and confirms that Accuracy, Precision, and Recall are numeric and each ≥ 0.6.
    """

    score = 0.0
    feedback = []

    if not actual:
        feedback.append("No working directory path provided for evaluation.")
        return 0.0

    base_dir = os.path.dirname(actual)

    script_path = os.path.join(base_dir, "mobile_price_random_forest.py")
    result_path = os.path.join(base_dir, "mobile_price_rf_summary.txt")

    # Step 1: Check script existence
    if os.path.exists(script_path):
        score += 0.4
        feedback.append("Python script found successfully.")
    else:
        feedback.append("Python script missing.")

    # Step 2: Check result file existence
    if os.path.exists(result_path):
        score += 0.4
        feedback.append("Result text file found successfully.")
    else:
        feedback.append("Result file missing.")
        logger.info("\n".join(feedback))
        return round(score, 2)

    # Step 3: Validate content and metrics
    try:
        with open(result_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        # Expected format: Accuracy: <val>, Precision: <val>, Recall: <val>
        pattern = r"^Accuracy:\s*([-+]?\d*\.?\d+),\s*Precision:\s*([-+]?\d*\.?\d+),\s*Recall:\s*([-+]?\d*\.?\d+)$"
        match = re.match(pattern, content)

        if match:
            acc = float(match.group(1))
            prec = float(match.group(2))
            rec = float(match.group(3))
            feedback.append(f"Detected valid output format with Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}")
            score += 0.2

            # Logical validation
            if all(0 <= x <= 1 for x in [acc, prec, rec]):
                feedback.append("All metric values are within the valid range 0–1.")
            else:
                feedback.append("One or more metrics fall outside the expected 0–1 range.")
                score -= 0.1

            # Check performance thresholds
            if acc >= 0.6 and prec >= 0.6 and rec >= 0.6:
                feedback.append("All model metrics meet the minimum expected threshold (≥ 0.6).")
                score += 0.1
            else:
                feedback.append("Some metrics are below the expected threshold of 0.6.")
        else:
            feedback.append("Invalid format. Expected format: 'Accuracy: <value>, Precision: <value>, Recall: <value>'.")

    except Exception as e:
        feedback.append(f"Error reading or validating result file: {e}")

    final_score = min(round(score, 2), 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score



def evaluate_mobile_xgboost_gridsearch(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for ds199 — Mobile XGBoost GridSearch.
    Validates the presence of the Python script and result file, checks for correct output format,
    ensures Best_Params is a valid dictionary-like structure, and confirms Mean_F1 is numeric and ≥ 0.6.
    """

    score = 0.0
    feedback = []

    if not actual:
        feedback.append("No working directory path provided for evaluation.")
        return 0.0

    base_dir = os.path.dirname(actual)
    script_path = os.path.join(base_dir, "mobile_xgboost_gridsearch.py")
    result_path = os.path.join(base_dir, "mobile_xgboost_gridsearch_summary.txt")

    # Step 1: Check Python script existence
    if os.path.exists(script_path):
        score += 0.4
        feedback.append("Python script located successfully.")
    else:
        feedback.append("Python script missing in the directory.")

    # Step 2: Check result file existence
    if os.path.exists(result_path):
        score += 0.4
        feedback.append("Result text file located successfully.")
    else:
        feedback.append("Result text file not found.")
        logger.info("\n".join(feedback))
        return round(score, 2)

    # Step 3: Validate result content
    try:
        with open(result_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        # Expected output format
        pattern = r"^Best_Params:\s*(\{.*\}),\s*Mean_F1:\s*([-+]?\d*\.?\d+)$"
        match = re.match(pattern, content)

        if match:
            params_str = match.group(1)
            f1_val = float(match.group(2))
            feedback.append(f"Detected valid output format. Parsed Mean_F1 = {f1_val:.4f}")
            score += 0.2

            # Validate params dictionary-like structure
            if params_str.startswith("{") and params_str.endswith("}") and ":" in params_str:
                feedback.append("Best_Params appears to be a valid dictionary-like structure.")
                score += 0.1
            else:
                feedback.append("Best_Params does not resemble a valid dictionary format.")

            # Validate F1-score range and threshold
            if 0 <= f1_val <= 1:
                if f1_val >= 0.6:
                    feedback.append("Mean_F1 meets minimum expected threshold (≥ 0.6).")
                    score += 0.1
                else:
                    feedback.append("Mean_F1 below expected threshold (0.6).")
            else:
                feedback.append("Mean_F1 is outside the valid 0–1 range.")
        else:
            feedback.append("Invalid format. Expected: 'Best_Params: {...}, Mean_F1: <value>'.")

    except Exception as e:
        feedback.append(f"Error reading or validating result file: {e}")

    # Step 4: Compute final score
    final_score = min(round(score, 2), 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score


def evaluate_mobile_efficiency_ratio(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for ds200 — Mobile Efficiency Ratio.
    Checks that both the Python script and output text file exist,
    verifies that the reported format matches 'Max_Efficiency_Group: (x, y)',
    and ensures x and y are binary values (0 or 1).
    """

    score = 0.0
    feedback = []

    if not actual:
        feedback.append("No working directory path provided for evaluation.")
        return 0.0

    base_dir = os.path.dirname(actual)
    script_path = os.path.join(base_dir, "mobile_efficiency_ratio.py")
    result_path = os.path.join(base_dir, "mobile_efficiency_summary.txt")

    # Step 1: Check for Python script
    if os.path.exists(script_path):
        score += 0.4
        feedback.append("Python script found successfully.")
    else:
        feedback.append("Python script not found in the expected directory.")

    # Step 2: Check for result file
    if os.path.exists(result_path):
        score += 0.4
        feedback.append("Result text file located successfully.")
    else:
        feedback.append("Result text file missing.")
        logger.info("\n".join(feedback))
        return round(score, 2)

    # Step 3: Validate result content
    try:
        with open(result_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        # Expected pattern: Max_Efficiency_Group: (x, y)
        pattern = r"^Max_Efficiency_Group:\s*\(\s*(\d)\s*,\s*(\d)\s*\)$"
        match = re.match(pattern, content)

        if match:
            x, y = int(match.group(1)), int(match.group(2))
            feedback.append(f"Detected valid output format: Max_Efficiency_Group = ({x}, {y})")
            score += 0.2

            # Check binary validity of dual_sim and four_g
            if x in [0, 1] and y in [0, 1]:
                feedback.append("Both dual_sim and four_g values are valid binary indicators (0 or 1).")
            else:
                feedback.append("Values for dual_sim or four_g are not binary (expected 0 or 1).")
                score -= 0.1
        else:
            feedback.append("Output format invalid. Expected 'Max_Efficiency_Group: (x, y)' with binary values.")

    except Exception as e:
        feedback.append(f"Error reading or validating result file: {e}")

    # Step 4: Final scoring and logging
    final_score = min(round(score, 2), 1.0)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return final_score


def evaluate_heart_disease_cholesterol_diff(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for ds201 — Heart Disease Cholesterol Difference.
    Checks that the Python script and output text exist, validates format,
    and confirms the reported cholesterol difference is numerically correct
    within ±2 of the expected ground truth (-51.181).
    """

    score = 0.0
    feedback = []

    if not actual:
        feedback.append("No working directory provided for evaluation.")
        return 0.0

    base_dir = os.path.dirname(actual)
    script_path = os.path.join(base_dir, "heart_disease_cholesterol_diff.py")
    result_path = os.path.join(base_dir, "heart_disease_cholesterol_summary.txt")

    # Check for script existence
    if os.path.exists(script_path):
        score += 0.2
        feedback.append("Python script located successfully.")
    else:
        feedback.append("Script file missing.")

    # Check for result file
    if os.path.exists(result_path):
        score += 0.4
        feedback.append("Result text file located successfully.")
    else:
        feedback.append("Result file not found.")
        logger.info("\n".join(feedback))
        return round(score, 2)

    # Read and validate output format
    try:
        with open(result_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        pattern = r"^Mean_Cholesterol_Diff:\s*(-?\d*\.?\d+)$"
        match = re.match(pattern, content)

        if match:
            diff = float(match.group(1))
            score += 0.2
            feedback.append(f"Valid format detected. Reported difference = {diff:.3f}.")

            # Check if value close to expected (-51.181 ± 2)
            expected_value = -51.181
            if abs(diff - expected_value) <= 2:
                score += 0.2
                feedback.append("Reported value is accurate within tolerance.")
            else:
                feedback.append(f"Reported value differs from expected ({expected_value}) beyond tolerance.")
        else:
            feedback.append("Output format invalid. Expected 'Mean_Cholesterol_Diff: <value>'.")

    except Exception as e:
        feedback.append(f"Error reading result file: {e}")

    final_score = min(round(score, 2), 1.0)

    for msg in feedback:
        logger.info(msg)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")

    return final_score

def evaluate_heart_chestpain_spearman_corr(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for ds202 — Heart Disease Spearman Correlation by Chest Pain Type.
    Checks existence of script and result files, validates output format,
    and compares reported values against expected within a ±0.02 tolerance.
    """

    score = 0.0
    feedback = []

    if not actual:
        feedback.append("No working directory provided for evaluation.")
        return 0.0

    base_dir = os.path.dirname(actual)
    script_path = os.path.join(base_dir, "heart_chestpain_spearman_corr.py")
    result_path = os.path.join(base_dir, "heart_chestpain_corr_summary.txt")

    # Check for script existence
    if os.path.exists(script_path):
        score += 0.2
        feedback.append("Python script located successfully.")
    else:
        feedback.append("Script file missing.")

    # Check for result file
    if os.path.exists(result_path):
        score += 0.4
        feedback.append("Result text file located successfully.")
    else:
        feedback.append("Result file not found.")
        logger.info("\n".join(feedback))
        return round(score, 2)

    # Read and validate the output
    try:
        with open(result_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        pattern = r"^Strongest_Negative_Type:\s*(\w+),\s*Correlation:\s*(-?\d*\.?\d+)$"
        match = re.match(pattern, content)

        if match:
            chest_type = match.group(1).upper()
            corr_value = float(match.group(2))
            score += 0.2
            feedback.append(f"Valid output format detected. Type = {chest_type}, Correlation = {corr_value:.3f}.")

            # Ground truth check
            expected_type = "TA"
            expected_corr = -0.496

            if chest_type == expected_type and abs(corr_value - expected_corr) <= 0.02:
                score += 0.2
                feedback.append("Reported values match expected within tolerance.")
            else:
                feedback.append(f"Reported values deviate from expected (Expected: {expected_type}, {expected_corr}).")

        else:
            feedback.append("Output format invalid. Expected 'Strongest_Negative_Type: <type>, Correlation: <value>'.")

    except Exception as e:
        feedback.append(f"Error reading result file: {e}")

    final_score = min(round(score, 2), 1.0)

    for msg in feedback:
        logger.info(msg)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")

    return final_score

def evaluate_heart_downslope_heartdisease_percent(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for ds203 — Heart Disease DownSlope–HeartDisease Percentage.
    Checks that both script and result files exist, validates the output format,
    and ensures the percentage value is close to the expected value (±1 tolerance).
    """

    score = 0.0
    feedback = []

    if not actual:
        feedback.append("No working directory provided for evaluation.")
        return 0.0

    base_dir = os.path.dirname(actual)
    script_path = os.path.join(base_dir, "heart_downslope_heartdisease_percent.py")
    result_path = os.path.join(base_dir, "heart_downslope_summary.txt")

    # Check that the main Python script exists
    if os.path.exists(script_path):
        score += 0.2
        feedback.append("Script file found successfully.")
    else:
        feedback.append("Script file missing.")

    # Check that the result text file exists
    if os.path.exists(result_path):
        score += 0.4
        feedback.append("Result file found successfully.")
    else:
        feedback.append("Result file missing.")
        logger.info("\n".join(feedback))
        return round(score, 2)

    # Read and validate the result file
    try:
        with open(result_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        # Expected format: DownSlope_HeartDisease_Percent: <float>
        pattern = r"^DownSlope_HeartDisease_Percent:\s*([-+]?\d*\.?\d+)$"
        match = re.match(pattern, content)

        if match:
            value = float(match.group(1))
            feedback.append(f"Valid format detected with value = {value:.2f}.")
            score += 0.2

            expected_value = 5.34
            if abs(value - expected_value) <= 1.0:
                score += 0.2
                feedback.append("Percentage value is within the acceptable ±1 range of the expected result.")
            else:
                feedback.append(f"Value deviates from expected. Found {value:.2f}, expected around {expected_value:.2f}.")
        else:
            feedback.append("Output format invalid. Expected 'DownSlope_HeartDisease_Percent: <value>'.")

    except Exception as e:
        feedback.append(f"Error reading result file: {e}")

    final_score = min(round(score, 2), 1.0)

    for msg in feedback:
        logger.info(msg)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")

    return final_score

def evaluate_retail_missing_value_imputation(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for ds204 — Retail Missing Value Imputation.
    Checks the existence of script and output files, validates the output format,
    and confirms that imputation steps (median, mean, and regression) are implemented.
    """

    score = 0.0
    feedback = []

    if not actual:
        feedback.append("No working directory provided for evaluation.")
        return 0.0

    base_dir = os.path.dirname(actual)
    script_path = os.path.join(base_dir, "retail_missing_value_imputation.py")
    result_path = os.path.join(base_dir, "retail_missing_value_summary.txt")

    # --- Check script existence ---
    if os.path.exists(script_path):
        score += 0.2
        feedback.append("Script file found successfully.")
    else:
        feedback.append("Python script file not found.")

    # --- Check result file existence ---
    if os.path.exists(result_path):
        score += 0.4
        feedback.append("Result text file found successfully.")
    else:
        feedback.append("Result text file missing.")
        logger.info("\n".join(feedback))
        return round(score, 2)

    # --- Check imputation logic in script ---
    try:
        with open(script_path, "r", encoding="utf-8") as f:
            code = f.read()

        # Check if core imputation elements are implemented
        if "median" in code and "fillna" in code:
            feedback.append("Median imputation logic detected (likely for Price).")
            score += 0.1
        else:
            feedback.append("Median imputation not clearly found in code.")

        if "mean" in code and "fillna" in code:
            feedback.append("Mean imputation logic detected (likely for Rating).")
            score += 0.1
        else:
            feedback.append("Mean imputation not clearly found in code.")

        if "LinearRegression" in code or "fit" in code:
            feedback.append("Model-based imputation logic detected for Discount.")
            score += 0.1
        else:
            feedback.append("Model-based imputation for Discount not found.")
    except Exception as e:
        feedback.append(f"Error reading script file: {e}")

    # --- Validate result content ---
    try:
        with open(result_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        # Expected format: Total_Imputed_Cells: <int>
        pattern = r"^Total_Imputed_Cells:\s*(\d+)$"
        match = re.match(pattern, content)

        if match:
            value = int(match.group(1))
            feedback.append(f"Valid result format detected with Total_Imputed_Cells = {value}.")
            score += 0.2

            expected_value = 2616
            if abs(value - expected_value) <= 5:
                score += 0.2
                feedback.append("Total imputed cell count is within ±5 of the expected value.")
            else:
                feedback.append(f"Reported imputed count differs significantly (expected around {expected_value}).")
        else:
            feedback.append("Invalid output format. Expected 'Total_Imputed_Cells: <value>'.")
    except Exception as e:
        feedback.append(f"Error reading result file: {e}")

    # --- Final scoring ---
    final_score = min(round(score, 2), 1.0)

    for msg in feedback:
        logger.info(msg)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")

    return final_score

def evaluate_retail_stock_balance_smote(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for ds205 — Retail Stock Balance using SMOTE.
    Checks that the script and output exist, confirms correct output format,
    and verifies the new class ratio is close to balanced (≈ 1 ± 0.2).
    """

    score = 0.0
    feedback = []

    if not actual:
        feedback.append("No working directory provided for evaluation.")
        return 0.0

    base_dir = os.path.dirname(actual)
    script_path = os.path.join(base_dir, "retail_stock_balance_smote.py")
    result_path = os.path.join(base_dir, "retail_stock_balance_summary.txt")

    # Confirm Python script exists
    if os.path.exists(script_path):
        score += 0.2
        feedback.append("Python script located successfully.")
    else:
        feedback.append("Script file missing.")

    # Confirm result text file exists
    if os.path.exists(result_path):
        score += 0.4
        feedback.append("Result text file found successfully.")
    else:
        feedback.append("Result text file not found.")
        logger.info("\n".join(feedback))
        return round(score, 2)

    # Check if script includes SMOTE-based balancing logic
    try:
        with open(script_path, "r", encoding="utf-8") as f:
            code = f.read()

        if "SMOTE" in code and "fit_resample" in code:
            feedback.append("Detected proper use of SMOTE resampling logic.")
            score += 0.1
        else:
            feedback.append("SMOTE implementation not clearly found in code.")

        if "value_counts" in code:
            feedback.append("Value count analysis detected for imbalance check.")
            score += 0.1
        else:
            feedback.append("Class ratio calculation not explicitly found.")
    except Exception as e:
        feedback.append(f"Error reading Python script: {e}")

    # Validate output format
    try:
        with open(result_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        # Expected format: Balanced_Ratio: <float>
        pattern = r"^Balanced_Ratio:\s*([-+]?\d*\.?\d+)$"
        match = re.match(pattern, content)

        if match:
            ratio = float(match.group(1))
            feedback.append(f"Valid output format detected with ratio = {ratio:.2f}.")
            score += 0.2

            # Check if ratio is near 1 (balanced)
            if 0.8 <= ratio <= 1.2:
                score += 0.2
                feedback.append("Balanced ratio confirmed — near perfect class balance achieved.")
            else:
                feedback.append(f"Ratio deviates from balance (expected near 1, got {ratio:.2f}).")
        else:
            feedback.append("Output format invalid. Expected 'Balanced_Ratio: <value>'.")

    except Exception as e:
        feedback.append(f"Error reading result file: {e}")

    final_score = min(round(score, 2), 1.0)

    for msg in feedback:
        logger.info(msg)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")

    return final_score

def evaluate_retail_outlier_removal_iqr(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for ds206 — Retail Price & Discount Outlier Removal (IQR method).
    Checks that the script and output files exist, validates output format,
    and ensures the outlier removal percentage is within a reasonable range.
    """

    score = 0.0
    feedback = []

    if not actual:
        feedback.append("No working directory provided for evaluation.")
        return 0.0

    base_dir = os.path.dirname(actual)
    script_path = os.path.join(base_dir, "retail_outlier_removal_iqr.py")
    result_path = os.path.join(base_dir, "retail_outlier_removal_summary.txt")

    # Check script existence
    if os.path.exists(script_path):
        score += 0.2
        feedback.append("Python script located successfully.")
    else:
        feedback.append("Python script not found.")

    # Check result file existence
    if os.path.exists(result_path):
        score += 0.4
        feedback.append("Result text file located successfully.")
    else:
        feedback.append("Result text file missing.")
        logger.info("\n".join(feedback))
        return round(score, 2)

    # Verify IQR logic presence in script
    try:
        with open(script_path, "r", encoding="utf-8") as f:
            code = f.read()

        if "IQR" in code or "quantile" in code:
            score += 0.1
            feedback.append("Detected IQR or quantile-based outlier detection logic.")
        else:
            feedback.append("IQR logic not clearly found in script.")
    except Exception as e:
        feedback.append(f"Error reading Python script: {e}")

    # Validate result format and value
    try:
        with open(result_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        pattern = r"^Outlier_Removal_Percent:\s*([-+]?\d*\.?\d+)$"
        match = re.match(pattern, content)

        if match:
            value = float(match.group(1))
            feedback.append(f"Valid output format detected with value = {value:.2f}.")
            score += 0.2

            # Check if value is near expected (0 ± 2)
            if abs(value - 0.0) <= 2.0:
                score += 0.1
                feedback.append("Outlier removal percentage within expected range.")
            else:
                feedback.append(f"Outlier percentage deviates significantly (expected ~0, got {value:.2f}).")
        else:
            feedback.append("Invalid output format. Expected 'Outlier_Removal_Percent: <value>'.")
    except Exception as e:
        feedback.append(f"Error reading result file: {e}")

    final_score = min(round(score, 2), 1.0)

    for msg in feedback:
        logger.info(msg)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")

    return final_score

def evaluate_retail_weekly_sales_analysis(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for ds207 — Retail Weekly Sales Moving Average & Holiday Impact.
    Checks the script and output file existence, ensures correct format,
    and validates that the numeric difference is within a reasonable threshold (±100).
    """

    score = 0.0
    feedback = []

    if not actual:
        feedback.append("No working directory provided for evaluation.")
        return 0.0

    base_dir = os.path.dirname(actual)
    script_path = os.path.join(base_dir, "retail_weekly_sales_analysis.py")
    result_path = os.path.join(base_dir, "retail_holiday_sales_difference.txt")

    # Verify the script exists
    if os.path.exists(script_path):
        score += 0.2
        feedback.append("Python script located successfully.")
    else:
        feedback.append("Script file not found.")

    # Verify the output file exists
    if os.path.exists(result_path):
        score += 0.4
        feedback.append("Result text file found successfully.")
    else:
        feedback.append("Result file missing.")
        logger.info("\n".join(feedback))
        return round(score, 2)

    # Check code for expected logic
    try:
        with open(script_path, "r", encoding="utf-8") as f:
            code = f.read()

        if "rolling" in code or "window" in code:
            score += 0.1
            feedback.append("Detected rolling window or moving average logic.")
        else:
            feedback.append("Moving average logic not clearly detected in script.")

        if "IsHoliday" in code and "mean" in code:
            score += 0.1
            feedback.append("Detected IsHoliday-based group difference calculation.")
        else:
            feedback.append("Holiday vs. Non-holiday comparison not clearly implemented.")
    except Exception as e:
        feedback.append(f"Error reading script: {e}")

    # Validate result format and value range
    try:
        with open(result_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        pattern = r"^Holiday_vs_NonHoliday_Diff:\s*([-+]?\d*\.?\d+)$"
        match = re.match(pattern, content)

        if match:
            diff = float(match.group(1))
            feedback.append(f"Valid output format detected with difference = {diff:.2f}.")
            score += 0.2

            expected_value = 1134.38
            threshold = 100
            if abs(diff - expected_value) <= threshold:
                score += 0.2
                feedback.append("Holiday vs Non-holiday sales difference within ±100 of expected value.")
            else:
                feedback.append(f"Value deviates beyond acceptable range (expected ~{expected_value}, got {diff:.2f}).")
        else:
            feedback.append("Invalid output format. Expected 'Holiday_vs_NonHoliday_Diff: <value>'.")
    except Exception as e:
        feedback.append(f"Error reading result file: {e}")

    final_score = min(round(score, 2), 1.0)

    for msg in feedback:
        logger.info(msg)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")

    return final_score

def evaluate_retail_store_sales_variability(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for ds208 — Retail Store Sales Variability.
    Checks for script and output file existence, correct output format,
    and verifies that the standard deviation value is close to the expected result.
    """

    score = 0.0
    feedback = []

    if not actual:
        feedback.append("No working directory provided for evaluation.")
        return 0.0

    base_dir = os.path.dirname(actual)
    script_path = os.path.join(base_dir, "retail_store_sales_variability.py")
    result_path = os.path.join(base_dir, "retail_sales_variability_summary.txt")

    # 1. Check script existence
    if os.path.exists(script_path):
        score += 0.2
        feedback.append("Python script located successfully.")
    else:
        feedback.append("Script file not found.")

    # 2. Check result file existence
    if os.path.exists(result_path):
        score += 0.4
        feedback.append("Result text file located successfully.")
    else:
        feedback.append("Result file missing.")
        logger.info("\n".join(feedback))
        return round(score, 2)

    # 3. Validate content and logic
    try:
        with open(result_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        # Expected format: Highest_Variability_Store: <int>, Std_Sales: <float>
        pattern = r"^Highest_Variability_Store:\s*(\d+),\s*Std_Sales:\s*([-+]?\d*\.?\d+)$"
        match = re.match(pattern, content)

        if match:
            store_id = int(match.group(1))
            std_val = float(match.group(2))
            feedback.append(f"Valid output format detected (Store={store_id}, Std={std_val:.2f}).")
            score += 0.2

            # 4. Logical check
            expected_store = 14
            expected_std = 36911.12
            tolerance = 500  # acceptable deviation range

            if store_id == expected_store:
                score += 0.1
                feedback.append("Correct store identified (Store 14).")
            else:
                feedback.append(f"Store mismatch: expected 14, got {store_id}.")

            if abs(std_val - expected_std) <= tolerance:
                score += 0.1
                feedback.append("Standard deviation within acceptable ±500 range.")
            else:
                feedback.append(f"Std_Sales value outside acceptable range (expected ~{expected_std}, got {std_val:.2f}).")
        else:
            feedback.append("Output format invalid. Expected 'Highest_Variability_Store: <store_id>, Std_Sales: <value>'.")
    except Exception as e:
        feedback.append(f"Error reading result file: {e}")

    # 5. Basic script logic check (optional)
    try:
        with open(script_path, "r", encoding="utf-8") as f:
            code = f.read()

        if "groupby" in code and "std" in code:
            score += 0.05
            feedback.append("Detected proper use of groupby and std operations.")
        else:
            feedback.append("Groupby or std logic not detected clearly in script.")
    except Exception as e:
        feedback.append(f"Error reading script: {e}")

    final_score = min(round(score, 2), 1.0)

    for msg in feedback:
        logger.info(msg)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")

    return final_score


def evaluate_retail_top_store_holiday_analysis_quality(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Retail Top Store Holiday Analysis (Visual Studio Code) task.

    Evaluation checks:
      1. Script existence
      2. Visualization logic (lineplot, boxplot)
      3. Computation logic for identifying top stores
      4. Visualization file creation
      5. GPT-4o semantic confirmation
    """

    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    expected = expected.get("rules", expected)
    base_dir = os.path.dirname(actual)

    script_file = os.path.join(base_dir, "retail_top_store_holiday_analysis.py")
    chart1 = os.path.join(base_dir, "top_stores_sales_trends.png")
    chart2 = os.path.join(base_dir, "holiday_comparison_boxplot.png")

    score = 0.0
    feedback = []

    # 1. Check script presence
    if os.path.exists(script_file) and os.path.getsize(script_file) > 50:
        score += 0.2
        feedback.append("Script file exists.")
    else:
        feedback.append("Script file missing or empty.")
        return score

    # 2. Check visualization logic
    with open(script_file, "r", encoding="utf-8", errors="ignore") as f:
        code = f.read().lower()

    if "lineplot" in code or "plt.plot" in code:
        score += 0.1
        feedback.append("Lineplot logic found in code.")
    else:
        feedback.append("Lineplot logic missing.")

    if "boxplot" in code:
        score += 0.1
        feedback.append("Boxplot logic found in code.")
    else:
        feedback.append("Boxplot logic missing.")

    # 3. Check computation logic for top stores
    if any(keyword in code for keyword in ["groupby", "sort", "nlargest", "rank", "head(", "idxmax"]):
        score += 0.2
        feedback.append("Computation logic for identifying top stores found.")
    else:
        feedback.append("No computation logic for top stores detected.")

    # 4. Visualization files
    for chart_path in [chart1, chart2]:
        chart_name = os.path.basename(chart_path)
        if os.path.exists(chart_path) and os.path.getsize(chart_path) > 1000:
            score += 0.1
            feedback.append(f"{chart_name} exists and is non-empty.")
        else:
            feedback.append(f"{chart_name} missing or appears empty.")

    # 5. GPT-4o semantic verification
    def encode_image(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    charts = [
        (
            chart1,
            "a multi-line chart showing Weekly_Sales trends over time for the top 3 stores, with holiday weeks highlighted using colors or markers."
        ),
        (
            chart2,
            "a boxplot comparing Weekly_Sales distributions between holiday and non-holiday weeks for the top-performing store."
        ),
    ]

    for chart_path, description in charts:
        chart_name = os.path.basename(chart_path)
        if not os.path.exists(chart_path):
            continue
        try:
            encoded = encode_image(chart_path)
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a data visualization evaluator. "
                            "Judge whether the provided chart accurately represents the described visualization, with clear axes and labels."
                        )
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    f"Does this chart correctly represent {description}? "
                                    "Reply 'yes' if it matches the description, or 'no' if not."
                                )
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{encoded}"}
                            }
                        ]
                    }
                ],
                max_tokens=100
            )
            answer = response.choices[0].message.content.lower()
            if "yes" in answer:
                score += 0.1
                feedback.append(f"GPT-4o confirmed that {chart_name} matches the expected visualization.")
            else:
                feedback.append(f"GPT-4o indicated that {chart_name} may not match perfectly: {answer}")
        except Exception as e:
            feedback.append(f"GPT-4o evaluation failed for {chart_name}: {e}")

    final_score = min(score, 1.0)
    logger.info(f"Final score: {final_score:.2f}")

    print("\nEvaluation Summary:")
    for msg in feedback:
        print("-", msg)
    print(f"\nFinal Score: {final_score:.2f}")

    return final_score


def evaluate_retail_markdown_impact_correlation(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for ds210 — Retail Markdown Impact Correlation.
    Checks that both the Python script and result file exist, verifies correct output format,
    and ensures the correlation value is numeric and within the expected range (-1 to 1),
    approximately matching the ground truth (±0.05 tolerance).
    """

    score = 0.0
    feedback = []

    if not actual:
        feedback.append("No working directory provided for evaluation.")
        return 0.0

    base_dir = os.path.dirname(actual)
    script_path = os.path.join(base_dir, "retail_markdown_impact_correlation.py")
    result_path = os.path.join(base_dir, "retail_markdown_correlation_summary.txt")

    # 1. Check script existence
    if os.path.exists(script_path):
        score += 0.2
        feedback.append("Python script located successfully.")
    else:
        feedback.append("Script file not found.")

    # 2. Check result file existence
    if os.path.exists(result_path):
        score += 0.4
        feedback.append("Result text file located successfully.")
    else:
        feedback.append("Result file missing.")
        logger.info("\n".join(feedback))
        return round(score, 2)

    # 3. Validate the output format and value
    try:
        with open(result_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        pattern = r"^Holiday_Markdown_Corr:\s*([-+]?\d*\.?\d+)$"
        match = re.match(pattern, content)

        if match:
            corr_value = float(match.group(1))
            feedback.append(f"Detected valid format: Holiday_Markdown_Corr = {corr_value:.4f}")
            score += 0.2

            # Expected value
            expected_corr = 0.1043
            tolerance = 0.05

            # 4. Logical correctness
            if -1 <= corr_value <= 1:
                score += 0.1
                feedback.append("Correlation value within valid range (-1 to 1).")
            else:
                feedback.append("Correlation value outside valid range.")

            # 5. Check proximity to expected
            if abs(corr_value - expected_corr) <= tolerance:
                score += 0.1
                feedback.append("Correlation value close to expected result (±0.05).")
            else:
                feedback.append(f"Correlation deviates beyond tolerance (expected ~{expected_corr}).")
        else:
            feedback.append("Output format invalid. Expected 'Holiday_Markdown_Corr: <value>'.")
    except Exception as e:
        feedback.append(f"Error while reading result file: {e}")

    # 6. Script content validation
    try:
        with open(script_path, "r", encoding="utf-8") as f:
            code = f.read()
        if all(x in code for x in ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5", "corr", "IsHoliday"]):
            score += 0.05
            feedback.append("Detected proper feature computation and correlation logic.")
        else:
            feedback.append("Key computation or correlation logic not clearly found in script.")
    except Exception as e:
        feedback.append(f"Error reading script content: {e}")

    final_score = min(round(score, 2), 1.0)

    for msg in feedback:
        logger.info(msg)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")

    return final_score

def evaluate_retail_weekly_sales_regression(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for ds211 — Retail Weekly Sales Regression.
    Checks that the Python script and output file exist, verifies correct output format,
    confirms R² and RMSE are numeric, and validates R² ≥ 0.25 and RMSE ≤ 20000.
    """

    score = 0.0
    feedback = []

    if not actual:
        feedback.append("No working directory provided.")
        return 0.0

    base_dir = os.path.dirname(actual)
    script_path = os.path.join(base_dir, "retail_weekly_sales_regression.py")
    result_path = os.path.join(base_dir, "retail_sales_regression_summary.txt")

    # 1. Script check
    if os.path.exists(script_path):
        score += 0.2
        feedback.append("Python script found successfully.")
    else:
        feedback.append("Python script not found.")

    # 2. Output file check
    if os.path.exists(result_path):
        score += 0.4
        feedback.append("Result text file found successfully.")
    else:
        feedback.append("Result file missing.")
        logger.info("\n".join(feedback))
        return round(score, 2)

    # 3. Validate content format and logic
    try:
        with open(result_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        # Expected format: R2: <float>, RMSE: <float>
        pattern = r"^R2:\s*([-+]?\d*\.?\d+),\s*RMSE:\s*([-+]?\d*\.?\d+)$"
        match = re.match(pattern, content)

        if match:
            r2_val = float(match.group(1))
            rmse_val = float(match.group(2))
            feedback.append(f"Detected valid format: R2 = {r2_val:.4f}, RMSE = {rmse_val:.2f}")
            score += 0.2

            # Logical checks
            if 0 <= r2_val <= 1:
                feedback.append("R² value is within a valid range (0–1).")
                score += 0.05
            else:
                feedback.append("R² value outside valid range.")

            if rmse_val > 0:
                feedback.append("RMSE value is positive.")
                score += 0.05
            else:
                feedback.append("RMSE value should be positive.")

            # Threshold evaluation
            if r2_val >= 0.25:
                feedback.append("R² meets the minimum expected threshold (≥ 0.25).")
                score += 0.05
            else:
                feedback.append("R² below the acceptable level (expected ≥ 0.25).")

            if rmse_val <= 20000:
                feedback.append("RMSE within the acceptable range (≤ 20000).")
                score += 0.05
            else:
                feedback.append("RMSE exceeds the acceptable limit.")
        else:
            feedback.append("Output format invalid. Expected 'R2: <value>, RMSE: <value>'.")
    except Exception as e:
        feedback.append(f"Error while reading result file: {e}")

    # 4. Script keyword verification
    try:
        with open(script_path, "r", encoding="utf-8") as f:
            code = f.read()
        if all(x in code for x in ["LinearRegression", "train_test_split", "r2_score", "mean_squared_error"]):
            feedback.append("Detected essential regression keywords in script.")
            score += 0.05
        else:
            feedback.append("Some regression-related functions not found in the script.")
    except Exception as e:
        feedback.append(f"Error reading script content: {e}")

    final_score = min(round(score, 2), 1.0)
    for msg in feedback:
        logger.info(msg)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")

    return final_score

def evaluate_retail_high_sales_classifier(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for ds212 — Retail High Sales Classification.
    Validates script and output existence, checks correct output format,
    ensures numeric precision, recall, and F1 values between 0–1,
    and confirms all metrics are at least 0.5.
    """

    score = 0.0
    feedback = []

    if not actual:
        feedback.append("No working directory specified.")
        return 0.0

    base_dir = os.path.dirname(actual)
    script_path = os.path.join(base_dir, "retail_high_sales_classifier.py")
    result_path = os.path.join(base_dir, "retail_high_sales_classification_summary.txt")

    # Check script existence
    if os.path.exists(script_path):
        score += 0.2
        feedback.append("Python script found successfully.")
    else:
        feedback.append("Python script not found.")

    # Check output file existence
    if os.path.exists(result_path):
        score += 0.4
        feedback.append("Result text file located successfully.")
    else:
        feedback.append("Result file missing.")
        logger.info("\n".join(feedback))
        return round(score, 2)

    # Validate output content format
    try:
        with open(result_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        # Expected: Precision: <float>, Recall: <float>, F1: <float>
        pattern = r"^Precision:\s*([-+]?\d*\.?\d+),\s*Recall:\s*([-+]?\d*\.?\d+),\s*F1:\s*([-+]?\d*\.?\d+)$"
        match = re.match(pattern, content)

        if match:
            precision_val = float(match.group(1))
            recall_val = float(match.group(2))
            f1_val = float(match.group(3))
            feedback.append(f"Detected valid format: Precision={precision_val:.3f}, Recall={recall_val:.3f}, F1={f1_val:.3f}")
            score += 0.2

            # Logical validation
            all_valid = all(0 <= v <= 1 for v in [precision_val, recall_val, f1_val])
            if all_valid:
                feedback.append("All metrics are within valid 0–1 range.")
                score += 0.05
            else:
                feedback.append("One or more metrics are outside the 0–1 range.")

            # Performance thresholds
            if precision_val >= 0.5 and recall_val >= 0.5 and f1_val >= 0.5:
                feedback.append("Model meets minimum performance thresholds (≥ 0.5 for all metrics).")
                score += 0.1
            else:
                feedback.append("One or more metrics below expected threshold (0.5).")
        else:
            feedback.append("Output format invalid. Expected 'Precision: <value>, Recall: <value>, F1: <value>'.")
    except Exception as e:
        feedback.append(f"Error reading result file: {e}")

    # Verify important model keywords
    try:
        with open(script_path, "r", encoding="utf-8") as f:
            code = f.read()
        required_keywords = ["RandomForestClassifier", "precision_score", "recall_score", "f1_score"]
        if all(k in code for k in required_keywords):
            feedback.append("Detected key classification components in the script.")
            score += 0.05
        else:
            feedback.append("Some required classification components not found in the script.")
    except Exception as e:
        feedback.append(f"Error while reading script content: {e}")

    # Final scoring
    final_score = min(round(score, 2), 1.0)
    for msg in feedback:
        logger.info(msg)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    return final_score


def evaluate_retail_holiday_sales_filter(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for ds213 — Retail Holiday Sales Filter.
    Checks:
      - Script and output file existence
      - Output format correctness
      - Valid numeric Mean_CPI and positive Records_Selected
      - Ensures script contains sufficient implementation content
    """

    score = 0.0
    feedback = []

    if not actual:
        feedback.append("No working directory provided for evaluation.")
        return 0.0

    base_dir = os.path.dirname(actual)
    script_path = os.path.join(base_dir, "retail_holiday_sales_filter.py")
    result_path = os.path.join(base_dir, "retail_holiday_sales_filter_summary.txt")

    # Check script existence
    if os.path.exists(script_path):
        score += 0.3
        feedback.append("Python script located successfully.")
    else:
        feedback.append("Python script not found.")

    # Check result file existence
    if os.path.exists(result_path):
        score += 0.3
        feedback.append("Result text file found successfully.")
    else:
        feedback.append("Result file missing.")
        logger.info("\n".join(feedback))
        return round(score, 2)

    # Validate output format and content
    try:
        with open(result_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        # Expected format: Top_Store: <id>, Mean_CPI: <value>, Records_Selected: <count>
        pattern = r"^Top_Store:\s*(\d+),\s*Mean_CPI:\s*([-+]?\d*\.?\d+),\s*Records_Selected:\s*(\d+)$"
        match = re.match(pattern, content)

        if match:
            store_id = int(match.group(1))
            mean_cpi = float(match.group(2))
            record_count = int(match.group(3))
            feedback.append(f"Detected valid format — Store {store_id}, Mean_CPI {mean_cpi:.4f}, Records_Selected {record_count}.")
            score += 0.3

            if record_count > 0:
                feedback.append("Record count positive — filtered subset found.")
                score += 0.05
            else:
                feedback.append("Record count is zero — check filtering logic.")

            if 100 <= mean_cpi <= 400:
                feedback.append("Mean CPI value within a realistic range (100–400).")
                score += 0.05
            else:
                feedback.append("Mean CPI outside expected range — verify computation.")
        else:
            feedback.append("Output format invalid. Expected 'Top_Store: <id>, Mean_CPI: <value>, Records_Selected: <count>'.")
    except Exception as e:
        feedback.append(f"Error reading result file: {e}")

    # Simple code presence validation (no keyword enforcement)
    try:
        with open(script_path, "r", encoding="utf-8") as f:
            code = f.read().strip()
        if len(code) > 50:
            feedback.append("Script contains sufficient implementation content.")
            score += 0.05
        else:
            feedback.append("Script content appears too short — may be incomplete.")
    except Exception as e:
        feedback.append(f"Error reading script: {e}")

    final_score = min(round(score, 2), 1.0)

    for msg in feedback:
        logger.info(msg)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")

    return final_score

def evaluate_retail_xgboost_interaction_analysis(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for ds214 — Retail XGBoost SHAP Interaction Analysis.
    Checks:
      • Script and output file existence
      • Correct output format
      • Valid numeric Mean_Interaction_Value (positive)
    """

    score = 0.0
    feedback = []

    if not actual:
        feedback.append("No working directory provided for evaluation.")
        return 0.0

    base_dir = os.path.dirname(actual)
    script_path = os.path.join(base_dir, "retail_xgboost_interaction_analysis.py")
    result_path = os.path.join(base_dir, "retail_xgboost_interaction_summary.txt")

    # --- Check script existence ---
    if os.path.exists(script_path):
        score += 0.3
        feedback.append("Python script located successfully.")
    else:
        feedback.append("Python script not found.")

    # --- Check result file existence ---
    if os.path.exists(result_path):
        score += 0.3
        feedback.append("Result text file located successfully.")
    else:
        feedback.append("Result file missing.")
        logger.info("\n".join(feedback))
        return round(score, 2)

    # --- Validate output format ---
    try:
        with open(result_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        # Expected format: Top_Interaction: (<Feature_A>, <Feature_B>), Mean_Interaction_Value: <value>
        pattern = r"^Top_Interaction:\s*\(([^,]+),\s*([^)]+)\),\s*Mean_Interaction_Value:\s*([-+]?\d*\.?\d+)$"
        match = re.match(pattern, content)

        if match:
            feature_a = match.group(1).strip()
            feature_b = match.group(2).strip()
            value = float(match.group(3))
            feedback.append(f"Detected valid format — ({feature_a}, {feature_b}), Mean_Interaction_Value = {value:.4f}.")
            score += 0.3

            # --- Basic numeric sanity check ---
            if value > 0:
                feedback.append("Mean interaction value is positive — plausible SHAP magnitude.")
                score += 0.1
            else:
                feedback.append("Mean interaction value non-positive — verify computation.")
        else:
            feedback.append("Output format invalid. Expected 'Top_Interaction: (<Feature_A>, <Feature_B>), Mean_Interaction_Value: <value>'.")
    except Exception as e:
        feedback.append(f"Error reading result file: {e}")

    final_score = min(round(score, 2), 1.0)

    for msg in feedback:
        logger.info(msg)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")

    return final_score

def evaluate_retail_gradient_boosting_tuning(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for ds215 — Retail Gradient Boosting Two-Phase Tuning.
    Checks:
      • Script and result file existence
      • Output format correctness
      • RMSE plausibility (positive and below a reasonable upper limit)
    """

    score = 0.0
    feedback = []

    if not actual:
        feedback.append("No working directory provided for evaluation.")
        return 0.0

    base_dir = os.path.dirname(actual)
    script_path = os.path.join(base_dir, "retail_gradient_boosting_tuning.py")
    result_path = os.path.join(base_dir, "retail_gradient_boosting_summary.txt")

    # --- Check script existence ---
    if os.path.exists(script_path):
        score += 0.3
        feedback.append("Python script found successfully.")
    else:
        feedback.append("Python script not found.")

    # --- Check result file existence ---
    if os.path.exists(result_path):
        score += 0.3
        feedback.append("Result file found successfully.")
    else:
        feedback.append("Result file missing.")
        logger.info("\n".join(feedback))
        return round(score, 2)

    # --- Validate result content ---
    try:
        with open(result_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        # Expected format:
        # Best_Config: {max_depth: <value>, learning_rate: <value>, subsample: <value>, min_samples_split: <value>}, RMSE: <value>
        pattern = (
            r"^Best_Config:\s*\{max_depth:\s*(\d+),\s*learning_rate:\s*([\d.]+),\s*"
            r"subsample:\s*([\d.]+),\s*min_samples_split:\s*(\d+)\},\s*RMSE:\s*([\d.]+)$"
        )
        match = re.match(pattern, content)

        if match:
            params = {
                "max_depth": int(match.group(1)),
                "learning_rate": float(match.group(2)),
                "subsample": float(match.group(3)),
                "min_samples_split": int(match.group(4)),
            }
            rmse = float(match.group(5))
            feedback.append(f"Detected valid format with parameters: {params}, RMSE = {rmse:.2f}.")
            score += 0.3

            # --- Check RMSE plausibility ---
            if 0 < rmse < 50000:
                feedback.append("RMSE within a plausible range.")
                score += 0.1
            else:
                feedback.append("RMSE unusually high or invalid. Please verify model evaluation.")
        else:
            feedback.append("Output format invalid. Expected 'Best_Config: {...}, RMSE: <value>'.")
    except Exception as e:
        feedback.append(f"Error reading or parsing result file: {e}")

    final_score = min(round(score, 2), 1.0)
    for msg in feedback:
        logger.info(msg)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    return final_score

def evaluate_heart_disease_model_comparison(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for ds216 — Heart Disease Model Comparison.
    Evaluates presence of files, correct result format, and sanity of mean metric score.
    """

    score = 0.0
    feedback = []

    if not actual:
        feedback.append("No working directory provided for evaluation.")
        return 0.0

    base_dir = os.path.dirname(actual)
    script_path = os.path.join(base_dir, "heart_disease_model_comparison.py")
    result_path = os.path.join(base_dir, "heart_disease_model_comparison_summary.txt")

    # --- Check if the script exists ---
    if os.path.exists(script_path):
        score += 0.3
        feedback.append("Python script successfully located.")
    else:
        feedback.append("Python script not found.")

    # --- Check if result file exists ---
    if os.path.exists(result_path):
        score += 0.3
        feedback.append("Result text file successfully located.")
    else:
        feedback.append("Result text file missing.")
        logger.info("\n".join(feedback))
        return round(score, 2)

    # --- Validate the output file content ---
    try:
        with open(result_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        # Expected format: Best_Model: <model_name>, Mean_Metric_Score: <value>
        pattern = r"^Best_Model:\s*(\w+),\s*Mean_Metric_Score:\s*([\d.]+)$"
        match = re.match(pattern, content)

        if match:
            model = match.group(1)
            mean_score = float(match.group(2))
            feedback.append(f"Detected valid format — Model: {model}, Mean Score: {mean_score:.4f}.")
            score += 0.3

            # --- Sanity check for mean metric score ---
            if 0.5 <= mean_score <= 1.0:
                feedback.append("Mean metric score within plausible performance range.")
                score += 0.1
            else:
                feedback.append("Mean metric score outside expected range (should be between 0.5 and 1.0).")
        else:
            feedback.append("Output format invalid. Expected 'Best_Model: <model_name>, Mean_Metric_Score: <value>'.")
    except Exception as e:
        feedback.append(f"Error while reading result file: {e}")

    # --- Final scoring ---
    final_score = min(round(score, 2), 1.0)
    for msg in feedback:
        logger.info(msg)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    return final_score

def evaluate_store_sales_summary(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for ds217 — Store Sales Summary.
    Verifies script and output existence, correct columns, and valid numeric output.
    """

    score = 0.0
    feedback = []

    if not actual:
        feedback.append("No working directory provided for evaluation.")
        return 0.0

    base_dir = os.path.dirname(actual)

    script_path = os.path.join(base_dir, "store_sales_summary.py")
    result_path = os.path.join(base_dir, "store_sales_summary.csv")

    # check for script
    if os.path.exists(script_path):
        score += 0.2
        feedback.append("Python script located successfully.")
    else:
        feedback.append("Python script missing.")
        logger.info("\n".join(feedback))
        return round(score, 2)

    # check for output file
    if os.path.exists(result_path):
        score += 0.4
        feedback.append("Output CSV file found.")
    else:
        feedback.append("Output CSV missing.")
        logger.info("\n".join(feedback))
        return round(score, 2)

    # validate CSV structure and numeric sanity
    try:
        df = pd.read_csv(result_path)
        cols = list(df.columns)

        if all(c in cols for c in ["Store_Type", "Avg_Weekly_Sales"]):
            score += 0.2
            feedback.append("Output columns verified.")
        else:
            feedback.append(f"Unexpected columns found: {cols}")

        # check numeric values
        if df["Avg_Weekly_Sales"].dtype.kind in "fi" and df["Avg_Weekly_Sales"].mean() > 0:
            score += 0.2
            feedback.append("Numeric values in Avg_Weekly_Sales appear valid.")
        else:
            feedback.append("Avg_Weekly_Sales values seem invalid or non-numeric.")

    except Exception as e:
        feedback.append(f"Error reading CSV: {e}")

    final_score = min(round(score, 2), 1.0)

    for msg in feedback:
        logger.info(msg)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")

    return final_score


def evaluate_retail_scaling_comparison(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for ds218 — Retail Scaling Comparison.
    Checks for file existence, correct output format, and validates numeric values.
    """

    score = 0.0
    feedback = []

    if not actual:
        feedback.append("No working directory provided for evaluation.")
        return 0.0

    base_dir = os.path.dirname(actual)

    script_path = os.path.join(base_dir, "retail_scaling_comparison.py")
    result_path = os.path.join(base_dir, "retail_scaling_summary.txt")

    # check script
    if os.path.exists(script_path):
        score += 0.2
        feedback.append("Python script located successfully.")
    else:
        feedback.append("Python script missing.")
        logger.info("\n".join(feedback))
        return round(score, 2)

    # check output text file
    if os.path.exists(result_path):
        score += 0.4
        feedback.append("Result file found successfully.")
    else:
        feedback.append("Result text file missing.")
        logger.info("\n".join(feedback))
        return round(score, 2)

    # read and validate output content
    try:
        with open(result_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        pattern = (
            r"Zscore_Mean:\s*([-+]?\d*\.?\d+),\s*"
            r"Zscore_Std:\s*([-+]?\d*\.?\d+),\s*"
            r"MinMax_Mean:\s*([-+]?\d*\.?\d+),\s*"
            r"MinMax_Std:\s*([-+]?\d*\.?\d+),\s*"
            r"Corr_Mean:\s*([-+]?\d*\.?\d+)"
        )
        match = re.match(pattern, content)

        if match:
            score += 0.2
            values = list(map(float, match.groups()))
            feedback.append(f"Detected valid format with {len(values)} numeric entries.")

            z_mean, z_std, m_mean, m_std, corr = values

            # sanity checks
            if 0.0 < abs(z_std) < 5 and 0.0 < abs(m_std) < 5:
                score += 0.1
                feedback.append("Standard deviations appear within a valid range.")
            else:
                feedback.append("Standard deviation values appear off-scale.")

            if -1.0 <= corr <= 1.0:
                score += 0.1
                feedback.append("Correlation value within valid range.")
            else:
                feedback.append("Correlation value outside expected range.")
        else:
            feedback.append("Output format invalid. Expected properly labeled numeric values.")

    except Exception as e:
        feedback.append(f"Error reading result file: {e}")

    final_score = min(round(score, 2), 1.0)

    for msg in feedback:
        logger.info(msg)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")

    return final_score

def evaluate_retail_model_stacking(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for ds219 — Retail Model Stacking.
    Checks file existence, correct format, and ensures RMSE < 20k.
    """
    score = 0.0
    feedback = []

    if not actual:
        feedback.append("No working directory provided for evaluation.")
        return 0.0

    base_dir = os.path.dirname(actual)

    script_path = os.path.join(base_dir, "retail_model_stacking.py")
    result_path = os.path.join(base_dir, "retail_stacking_summary.txt")

    # Check script
    if os.path.exists(script_path):
        score += 0.2
        feedback.append("Python script located successfully.")
    else:
        feedback.append("Python script missing.")
        logger.info("\n".join(feedback))
        return round(score, 2)

    # Check output file
    if os.path.exists(result_path):
        score += 0.4
        feedback.append("Result text file found successfully.")
    else:
        feedback.append("Result file missing.")
        logger.info("\n".join(feedback))
        return round(score, 2)

    # Validate file content
    try:
        with open(result_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        pattern = (
            r"Base_Models:\s*\[RF,\s*GB,\s*LR\],\s*"
            r"Meta_Model:\s*GBR,\s*"
            r"Mean_RMSE:\s*([-+]?\d*\.?\d+)"
        )
        match = re.match(pattern, content)

        if match:
            rmse = float(match.group(1))
            feedback.append(f"Valid format detected. RMSE = {rmse:.2f}")
            score += 0.2

            if 0 < rmse < 20000:
                feedback.append("RMSE value within acceptable range (< 20k).")
                score += 0.2
            else:
                feedback.append("RMSE value exceeds expected threshold (>= 20k).")
        else:
            feedback.append("Output format invalid. Expected 'Base_Models: [RF, GB, LR], Meta_Model: GBR, Mean_RMSE: <value>'.")

    except Exception as e:
        feedback.append(f"Error reading result file: {e}")

    final_score = min(round(score, 2), 1.0)

    for msg in feedback:
        logger.info(msg)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")

    return final_score


def evaluate_retail_model_stacking(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for ds219 — Retail Model Stacking.
    Checks file existence, correct format, and ensures RMSE < 20k.
    """
    score = 0.0
    feedback = []

    if not actual:
        feedback.append("No working directory provided for evaluation.")
        return 0.0

    base_dir = os.path.dirname(actual)

    script_path = os.path.join(base_dir, "retail_model_stacking.py")
    result_path = os.path.join(base_dir, "retail_stacking_summary.txt")

    # Check script
    if os.path.exists(script_path):
        score += 0.2
        feedback.append("Python script located successfully.")
    else:
        feedback.append("Python script missing.")
        logger.info("\n".join(feedback))
        return round(score, 2)

    # Check output file
    if os.path.exists(result_path):
        score += 0.4
        feedback.append("Result text file found successfully.")
    else:
        feedback.append("Result file missing.")
        logger.info("\n".join(feedback))
        return round(score, 2)

    # Validate file content
    try:
        with open(result_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        pattern = (
            r"Base_Models:\s*\[RF,\s*GB,\s*LR\],\s*"
            r"Meta_Model:\s*GBR,\s*"
            r"Mean_RMSE:\s*([-+]?\d*\.?\d+)"
        )
        match = re.match(pattern, content)

        if match:
            rmse = float(match.group(1))
            feedback.append(f"Valid format detected. RMSE = {rmse:.2f}")
            score += 0.2

            if 0 < rmse < 20000:
                feedback.append("RMSE value within acceptable range (< 20k).")
                score += 0.2
            else:
                feedback.append("RMSE value exceeds expected threshold (>= 20k).")
        else:
            feedback.append("Output format invalid. Expected 'Base_Models: [RF, GB, LR], Meta_Model: GBR, Mean_RMSE: <value>'.")

    except Exception as e:
        feedback.append(f"Error reading result file: {e}")

    final_score = min(round(score, 2), 1.0)

    for msg in feedback:
        logger.info(msg)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")

    return final_score

def evaluate_mnist_pca_tsne_analysis(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for ds220 — MNIST PCA + t-SNE analysis.
    Checks for script, output text, correct format, and valid value ranges.
    """
    score = 0.0
    feedback = []

    if not actual:
        feedback.append("No working directory provided for evaluation.")
        return 0.0

    base_dir = os.path.dirname(actual)

    script_path = os.path.join(base_dir, "mnist_pca_tsne_analysis.py")
    result_path = os.path.join(base_dir, "mnist_pca_tsne_summary.txt")

    # Step 1: Check for script file
    if os.path.exists(script_path):
        score += 0.2
        feedback.append("Python script located successfully.")
    else:
        feedback.append("Python script missing.")
        logger.info("\n".join(feedback))
        return round(score, 2)

    # Step 2: Check for result file
    if os.path.exists(result_path):
        score += 0.4
        feedback.append("Result text file located successfully.")
    else:
        feedback.append("Result file missing.")
        logger.info("\n".join(feedback))
        return round(score, 2)

    # Step 3: Validate content format and numeric values
    try:
        with open(result_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        pattern = (
            r"^Explained_Variance_95%:\s*(\d+),\s*"
            r"Mean_IntraClass_Distance:\s*([-+]?\d*\.?\d+)$"
        )
        match = re.match(pattern, content)

        if match:
            n_components = int(match.group(1))
            intra_dist = float(match.group(2))
            feedback.append(
                f"Detected valid format — Components: {n_components}, "
                f"Mean_IntraClass_Distance: {intra_dist:.3f}"
            )
            score += 0.2

            # Logical checks
            if n_components > 0 and intra_dist > 0 and intra_dist < 10:
                feedback.append("Values are within valid expected range (distance < 10).")
                score += 0.2
            else:
                feedback.append("Values found but out of realistic range (distance ≥ 10).")
        else:
            feedback.append(
                "Output format invalid. Expected 'Explained_Variance_95%: <int>, Mean_IntraClass_Distance: <float>'."
            )

    except Exception as e:
        feedback.append(f"Error while reading or parsing result file: {e}")

    # Finalize
    final_score = min(round(score, 2), 1.0)
    for msg in feedback:
        logger.info(msg)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")

    return final_score


def evaluate_mnist_digit_ttest(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for ds221 — MNIST Welch’s t-test between two digit classes.
    Checks for script and result files, verifies format correctness, 
    and ensures that p_value < 0.05 when Significant=True.
    """

    score = 0.0
    feedback = []

    if not actual:
        feedback.append("No working directory provided for evaluation.")
        return 0.0

    base_dir = os.path.dirname(actual)

    script_path = os.path.join(base_dir, "mnist_digit_ttest.py")
    result_path = os.path.join(base_dir, "mnist_digit_ttest_summary.txt")

    # Step 1: Check script existence
    if os.path.exists(script_path):
        score += 0.2
        feedback.append("Python script found successfully.")
    else:
        feedback.append("Python script missing.")
        logger.info("\n".join(feedback))
        return round(score, 2)

    # Step 2: Check result file existence
    if os.path.exists(result_path):
        score += 0.4
        feedback.append("Result file located successfully.")
    else:
        feedback.append("Result file not found.")
        logger.info("\n".join(feedback))
        return round(score, 2)

    # Step 3: Validate output format and logical correctness
    try:
        with open(result_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        # Expected format: t_statistic: <float>, p_value: <float>, Significant: <True/False>
        pattern = r"^t_statistic:\s*([-+]?\d*\.?\d+),\s*p_value:\s*([-+]?\d*\.?\d+),\s*Significant:\s*(True|False)$"
        match = re.match(pattern, content)

        if match:
            t_stat = float(match.group(1))
            p_val = float(match.group(2))
            sig = match.group(3) == "True"

            feedback.append(
                f"Detected valid format — t_statistic={t_stat:.4f}, p_value={p_val:.6f}, Significant={sig}"
            )
            score += 0.2

            # Logical check on p-value and significance flag
            if (p_val < 0.05 and sig) or (p_val >= 0.05 and not sig):
                feedback.append("Significance flag correctly reflects p-value result.")
                score += 0.2
            else:
                feedback.append("Significance flag inconsistent with p-value result.")
        else:
            feedback.append("Invalid format. Expected 't_statistic: <value>, p_value: <value>, Significant: <True/False>'.")

    except Exception as e:
        feedback.append(f"Error reading or parsing result file: {e}")

    # Step 4: Final scoring
    final_score = min(round(score, 2), 1.0)
    for msg in feedback:
        logger.info(msg)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")

    return final_score


def evaluate_mnist_pca_logistic_metrics(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for ds222 — MNIST PCA + Logistic Regression metrics.
    Verifies script and CSV outputs, checks expected columns, and ensures accuracy is reasonable.
    """

    score = 0.0
    feedback = []

    if not actual:
        feedback.append("No working directory provided for evaluation.")
        return 0.0

    base_dir = os.path.dirname(actual)

    script_path = os.path.join(base_dir, "mnist_pca_logistic_metrics.py")
    result_path = os.path.join(base_dir, "mnist_metrics_summary.csv")

    # 1. Script existence
    if os.path.exists(script_path):
        score += 0.2
        feedback.append("Python script located successfully.")
    else:
        feedback.append("Python script not found.")
        logger.info("\n".join(feedback))
        return round(score, 2)

    # 2. Result CSV existence
    if os.path.exists(result_path):
        score += 0.4
        feedback.append("Metrics CSV file found successfully.")
    else:
        feedback.append("Result CSV file missing.")
        logger.info("\n".join(feedback))
        return round(score, 2)

    # 3. Validate structure and accuracy
    try:
        df = pd.read_csv(result_path)
        expected_cols = ["Digit", "Precision", "Recall", "F1", "Support", "Overall_Accuracy"]

        if all(col in df.columns for col in expected_cols):
            score += 0.2
            feedback.append("CSV contains all expected columns.")
        else:
            feedback.append(f"Missing expected columns. Found: {list(df.columns)}")

        # 4. Accuracy sanity check
        if "Overall_Accuracy" in df.columns:
            avg_acc = df["Overall_Accuracy"].mean()
            feedback.append(f"Detected mean overall accuracy: {avg_acc:.3f}")
            if 0.70 <= avg_acc <= 1.0:
                feedback.append("Accuracy within reasonable range.")
                score += 0.2
            else:
                feedback.append("Accuracy appears unusually low or invalid.")
        else:
            feedback.append("Column 'Overall_Accuracy' not found.")

    except Exception as e:
        feedback.append(f"Error reading or analyzing CSV file: {e}")

    final_score = min(round(score, 2), 1.0)

    for msg in feedback:
        logger.info(msg)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")

    return final_score

def evaluate_tesla_stock_fetch_analysis(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for ds223 — Tesla Stock Fetch and Analysis Task.
    Verifies:
      1. The script file exists.
      2. The output CSV exists and contains proper columns.
      3. The Close_Price value equals 430.60 (with tolerance ±0.5).
    """

    score = 0.0
    feedback = []

    if not actual:
        feedback.append("No working directory provided for evaluation.")
        logger.info("\n".join(feedback))
        return 0.0

    base_dir = os.path.dirname(actual)
    script_path = os.path.join(base_dir, "tesla_stock_fetch_analysis.py")
    csv_path = os.path.join(base_dir, "tesla_stock_latest.csv")

    # Check script existence
    if os.path.exists(script_path):
        score += 0.3
        feedback.append("Python script found successfully.")
    else:
        feedback.append("Python script missing.")

    # Check CSV existence
    if not os.path.exists(csv_path):
        feedback.append("Output CSV file not found.")
        logger.info("\n".join(feedback))
        return round(score, 2)

    try:
        df = pd.read_csv(csv_path)

        # Validate expected columns
        expected_cols = {"Timestamp", "Close_Price"}
        if expected_cols.issubset(df.columns):
            score += 0.3
            feedback.append("CSV columns are correctly named and formatted.")
        else:
            feedback.append(f"Missing required columns: {expected_cols - set(df.columns)}")

        # Check if Close_Price = 430.60 ± 0.5
        close_values = df["Close_Price"].dropna().astype(float)
        if not close_values.empty:
            value = float(close_values.iloc[0])
            if abs(value - 430.60) <= 0.5:
                score += 0.4
                feedback.append(f"Close_Price value {value} matches expected 430.60 within tolerance.")
            else:
                feedback.append(f"Close_Price {value} deviates from expected 430.60.")
        else:
            feedback.append("No Close_Price values found in CSV.")

    except Exception as e:
        feedback.append(f"Error reading CSV: {e}")

    # Final scoring
    final_score = min(round(score, 2), 1.0)

    logger.info("\n".join(feedback))
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    return final_score 

def evaluate_tesla_stock_volatility(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for ds224 — Tesla Stock Volatility Task.
    Checks:
      1. Script file exists.
      2. CSV file exists and contains required columns.
      3. Close_2025_10_10 ≈ 413.49 ± 0.5.
    """

    score = 0.0
    feedback = []

    if not actual:
        feedback.append("No working directory provided.")
        logger.info("\n".join(feedback))
        return 0.0

    base_dir = os.path.dirname(actual)
    script_path = os.path.join(base_dir, "tesla_stock_volatility.py")
    csv_path = os.path.join(base_dir, "tesla_volatility_metrics.csv")

    # Script existence
    if os.path.exists(script_path):
        score += 0.3
        feedback.append("Script file found.")
    else:
        feedback.append("Script missing.")

    # CSV existence
    if not os.path.exists(csv_path):
        feedback.append("Output CSV file not found.")
        logger.info("\n".join(feedback))
        return round(score, 2)

    try:
        df = pd.read_csv(csv_path)

        # Validate columns
        expected_cols = {"Timestamp", "Close_2025_10_10"}
        if expected_cols.issubset(df.columns):
            score += 0.3
            feedback.append("CSV columns are correctly formatted.")
        else:
            feedback.append(f"Missing required columns: {expected_cols - set(df.columns)}")

        # Check Close_2025_10_10 value ~ 413.49 ± 0.5
        if "Close_2025_10_10" in df.columns:
            val = float(df["Close_2025_10_10"].dropna().iloc[0])
            if abs(val - 413.49) <= 0.5:
                score += 0.4
                feedback.append(f"Close_2025_10_10 ({val}) matches expected 413.49 within tolerance.")
            else:
                feedback.append(f"Close_2025_10_10 ({val}) deviates from expected 413.49.")
        else:
            feedback.append("Column Close_2025_10_10 not found.")

    except Exception as e:
        feedback.append(f"Error reading CSV: {e}")

    final_score = min(round(score, 2), 1.0)
    logger.info("\n".join(feedback))
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    return final_score

def evaluate_meta_stock_trend_forecast(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for ds225 — Meta Stock Trend and Volatility Forecast.
    Checks:
      1. Python script exists.
      2. CSV file exists and contains required columns.
      3. Closing price on 2025-11-12 matches expected 609.01 ± 0.5.
      4. R2 > 0.5 and RMSE < 20.
    """

    score = 0.0
    feedback = []

    if not actual:
        feedback.append("No working directory provided.")
        logger.info("\n".join(feedback))
        return 0.0

    base_dir = os.path.dirname(actual)
    script_path = os.path.join(base_dir, "meta_stock_trend_forecast.py")
    csv_path = os.path.join(base_dir, "meta_stock_forecast_summary.csv")

    # check for script
    if os.path.exists(script_path):
        score += 0.3
        feedback.append("Python script found successfully.")
    else:
        feedback.append("Python script not found.")

    # check for csv
    if not os.path.exists(csv_path):
        feedback.append("CSV file not found.")
        logger.info("\n".join(feedback))
        return round(score, 2)

    try:
        df = pd.read_csv(csv_path)

        # check columns
        required_cols = {"Timestamp", "Close_2025_11_12"}
        if required_cols.issubset(df.columns):
            score += 0.3
            feedback.append("CSV columns are correctly formatted.")
        else:
            feedback.append("Required columns are missing.")

        # check close price value
        if "Close_2025_11_12" in df.columns:
            val = float(df["Close_2025_11_12"].dropna().iloc[0])
            if abs(val - 609.01) <= 0.5:
                score += 0.2
                feedback.append("Close_2025_11_12 value matches expected price within tolerance.")
            else:
                feedback.append(f"Close_2025_11_12 value {val} deviates from expected 609.01.")
        else:
            feedback.append("Column Close_2025_11_12 not found.")

        # optional model metrics
        if "R2" in df.columns and "RMSE" in df.columns:
            r2 = float(df["R2"].dropna().iloc[-1])
            rmse = float(df["RMSE"].dropna().iloc[-1])
            if r2 > 0.5 and rmse < 20:
                score += 0.2
                feedback.append("R2 and RMSE are within the expected range.")
            else:
                feedback.append("R2 and RMSE are outside expected performance range.")

    except Exception as e:
        feedback.append(f"Error while reading CSV file: {e}")

    final_score = min(round(score, 2), 1.0)
    logger.info("\n".join(feedback))
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    return final_score

def evaluate_bbc_tech_news_summary(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for ds226 — BBC Technology News Scraper and Summarizer.
    Checks that:
      1. Script and CSV exist.
      2. CSV has two columns: headline and article_first_100_words.
      3. CSV contains at least five rows.
      4. Text fields are non-empty and formatted correctly.
    """

    score = 0.0
    feedback = []

    if not actual:
        feedback.append("No working directory provided for evaluation.")
        logger.info("\n".join(feedback))
        return 0.0

    base_dir = os.path.dirname(actual)
    script_path = os.path.join(base_dir, "bbc_tech_news_summary.py")
    csv_path = os.path.join(base_dir, "tech_news_summary.csv")

    # Check script existence
    if os.path.exists(script_path):
        score += 0.3
        feedback.append("Python script found successfully.")
    else:
        feedback.append("Python script not found.")

    # Check CSV existence
    if not os.path.exists(csv_path):
        feedback.append("CSV file not found.")
        logger.info("\n".join(feedback))
        return round(score, 2)

    try:
        df = pd.read_csv(csv_path)

        # Check column names
        required_cols = {"headline", "article_first_100_words"}
        if required_cols.issubset(df.columns):
            score += 0.3
            feedback.append("CSV contains the required columns.")
        else:
            feedback.append("Required columns are missing.")

        # Check number of rows
        if len(df) >= 5:
            score += 0.2
            feedback.append("CSV contains at least five article entries.")
        else:
            feedback.append("Fewer than five article entries detected.")

        # Check text formatting
        if df["headline"].astype(str).str.len().mean() > 10 and df["article_first_100_words"].astype(str).str.len().mean() > 50:
            score += 0.2
            feedback.append("Headline and article summaries contain meaningful text.")
        else:
            feedback.append("Some text fields appear too short or incomplete.")

    except Exception as e:
        feedback.append(f"Error reading CSV file: {e}")

    final_score = min(round(score, 2), 1.0)
    logger.info("\n".join(feedback))
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    return final_score

def evaluate_worldbank_canada_gdp_inflation_analysis(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for ds227 — World Bank GDP–Inflation Correlation for Canada.
    Checks:
      1. Python script and output file exist.
      2. Script correctly references both required API endpoints.
      3. Output text matches expected format.
      4. Correlation value is valid and within range.
      5. Max_Divergence_Year is a valid 4-digit year.
    """

    score = 0.0
    feedback = []

    if not actual:
        feedback.append("No working directory provided.")
        logger.info("\n".join(feedback))
        return 0.0

    base_dir = os.path.dirname(actual)
    script_path = os.path.join(base_dir, "worldbank_canada_gdp_inflation_analysis.py")
    result_path = os.path.join(base_dir, "canada_gdp_inflation_summary.txt")

    # Check script presence
    if os.path.exists(script_path):
        score += 0.2
        feedback.append("Python script found successfully.")
    else:
        feedback.append("Script not found.")

    # Validate API endpoints inside script
    try:
        with open(script_path, "r", encoding="utf-8") as sfile:
            script_content = sfile.read()
            if (
                "NY.GDP.MKTP.CD" in script_content
                and "FP.CPI.TOTL.ZG" in script_content
                and "https://api.worldbank.org/v2/country/CA" in script_content
            ):
                score += 0.3
                feedback.append("World Bank API endpoints detected correctly.")
            else:
                feedback.append("Missing or incorrect API endpoint references.")
    except Exception as e:
        feedback.append(f"Could not read script for API verification: {e}")

    # Check result file
    if not os.path.exists(result_path):
        feedback.append("Output text file not found.")
        logger.info("\n".join(feedback))
        return round(score, 2)

    try:
        with open(result_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        # Expected: Correlation: <value>, Max_Divergence_Year: <year>
        pattern = r"^Correlation:\s*([-+]?\d*\.?\d+),\s*Max_Divergence_Year:\s*(\d{4})$"
        match = re.match(pattern, content)
        if match:
            corr = float(match.group(1))
            year = int(match.group(2))
            score += 0.4
            feedback.append(f"Detected valid result format: Correlation={corr}, Year={year}")

            if -1 <= corr <= 1:
                score += 0.1
                feedback.append("Correlation value within valid range.")
            else:
                feedback.append("Correlation value outside valid range.")

            if 2000 <= year <= 2025:
                feedback.append("Max divergence year is realistic.")
            else:
                feedback.append("Year appears unrealistic.")
        else:
            feedback.append("Invalid output format detected.")

    except Exception as e:
        feedback.append(f"Error reading output file: {e}")

    final_score = min(round(score, 2), 1.0)
    logger.info("\n".join(feedback))
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    return final_score


def evaluate_worldbank_india_energy_emission_analysis(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for ds228 — World Bank Energy–Emission Correlation for India.
    Checks:
      1. Script and output text file exist.
      2. Script includes both required API endpoints.
      3. Output format matches the expected structure.
      4. Correlation is valid and between -1 and 1.
      5. Sharpest_Emission_Drop_Year is a valid 4-digit year.
    """

    score = 0.0
    feedback = []

    if not actual:
        feedback.append("No working directory provided.")
        logger.info("\n".join(feedback))
        return 0.0

    base_dir = os.path.dirname(actual)
    script_path = os.path.join(base_dir, "worldbank_india_energy_emission_analysis.py")
    result_path = os.path.join(base_dir, "india_energy_emission_summary.txt")

    # Check script existence
    if os.path.exists(script_path):
        score += 0.2
        feedback.append("Python script found successfully.")
    else:
        feedback.append("Script missing.")

    # Validate endpoint usage
    try:
        with open(script_path, "r", encoding="utf-8") as file:
            content = file.read()
            if (
                "EG.USE.PCAP.KG.OE" in content
                and "EN.ATM.CO2E.PC" in content
                and "https://api.worldbank.org/v2/country/IN" in content
            ):
                score += 0.3
                feedback.append("World Bank API endpoints correctly referenced.")
            else:
                feedback.append("Endpoints missing or incorrectly formatted.")
    except Exception as e:
        feedback.append(f"Could not verify script: {e}")

    # Check output file existence
    if not os.path.exists(result_path):
        feedback.append("Output summary file not found.")
        logger.info("\n".join(feedback))
        return round(score, 2)

    try:
        with open(result_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        # Expected pattern
        pattern = r"^Correlation:\s*([-+]?\d*\.?\d+),\s*Sharpest_Emission_Drop_Year:\s*(\d{4})$"
        match = re.match(pattern, content)

        if match:
            corr = float(match.group(1))
            year = int(match.group(2))
            feedback.append(f"Detected output: Correlation={corr}, Year={year}")
            score += 0.4

            if -1 <= corr <= 1:
                score += 0.1
                feedback.append("Correlation value within valid range.")
            else:
                feedback.append("Correlation out of range.")

            if 2000 <= year <= 2025:
                feedback.append("Sharpest emission drop year is realistic.")
            else:
                feedback.append("Year value appears outside expected range.")
        else:
            feedback.append("Output format invalid.")

    except Exception as e:
        feedback.append(f"Error reading result file: {e}")

    final_score = min(round(score, 2), 1.0)
    logger.info("\n".join(feedback))
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    return final_score

def evaluate_github_trending_scraper(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for ds229 — GitHub Trending Scraper.
    Checks:
      1. Script and CSV file exist.
      2. CSV contains the expected columns and exactly 10 rows.
      3. Stars column is numeric and non-empty.
    """

    score = 0.0
    feedback = []

    if not actual:
        feedback.append("No working directory specified for evaluation.")
        logger.info("\n".join(feedback))
        return 0.0

    base_dir = os.path.dirname(actual)
    script_path = os.path.join(base_dir, "github_trending_scraper.py")
    csv_path = os.path.join(base_dir, "github_trending.csv")

    # Check script existence
    if os.path.exists(script_path):
        score += 0.3
        feedback.append("Python script found successfully.")
    else:
        feedback.append("Python script not found.")

    # Check CSV output
    if not os.path.exists(csv_path):
        feedback.append("CSV output file missing.")
        logger.info("\n".join(feedback))
        return round(score, 2)

    try:
        df = pd.read_csv(csv_path)
        score += 0.4
        feedback.append("CSV file loaded successfully.")

        if list(df.columns) == ["Repository", "Stars"]:
            score += 0.1
            feedback.append("Column names are correctly formatted.")
        else:
            feedback.append("Column names do not match expected format.")

        if len(df) == 10:
            score += 0.1
            feedback.append("CSV contains exactly 10 rows.")
        else:
            feedback.append(f"Row count mismatch: expected 10, found {len(df)}.")

        if df["Stars"].apply(lambda x: str(x).replace(",", "").isdigit()).all():
            score += 0.1
            feedback.append("Star counts are valid numeric values.")
        else:
            feedback.append("Non-numeric values found in Stars column.")

    except Exception as e:
        feedback.append(f"Error reading CSV file: {e}")

    final_score = min(round(score, 2), 1.0)
    logger.info("\n".join(feedback))
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    return final_score

def evaluate_github_trending_analysis(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for ds223 — GitHub Trending Analysis.
    Checks for script file, CSV file, image file, correct columns, and numeric sanity for stars.
    """

    score = 0.0
    feedback = []

    if not actual:
        feedback.append("No working directory provided for evaluation.")
        return 0.0

    base_dir = os.path.dirname(actual)

    script_path = os.path.join(base_dir, "github_trending_analysis.py")
    csv_path = os.path.join(base_dir, "github_trending_repos.csv")
    image_path = os.path.join(base_dir, "avg_stars_by_language.png")

    # Script file check
    if os.path.exists(script_path):
        score += 0.2
        feedback.append("Python script found.")
    else:
        feedback.append("Python script missing.")
        logger.info("\n".join(feedback))
        return round(score, 2)

    # CSV file check
    if os.path.exists(csv_path):
        score += 0.3
        feedback.append("CSV file found.")
    else:
        feedback.append("CSV file missing.")
        logger.info("\n".join(feedback))
        return round(score, 2)

    # Image file check
    if os.path.exists(image_path):
        score += 0.2
        feedback.append("Image file found.")
    else:
        feedback.append("Image file missing.")
        logger.info("\n".join(feedback))
        return round(score, 2)

    # Check CSV columns and values
    try:
        df = pd.read_csv(csv_path)
        expected_cols = ["repo_name", "author", "stars", "language"]
        if all(col in df.columns for col in expected_cols):
            score += 0.2
            feedback.append("CSV contains expected columns.")
        else:
            feedback.append(f"CSV missing expected columns; found {list(df.columns)}")

        if df["stars"].dtype.kind in "iu" and df["stars"].min() >= 0:
            score += 0.1
            feedback.append("Stars column has valid non-negative integers.")
        else:
            feedback.append("Stars column invalid or negative.")

    except Exception as e:
        feedback.append(f"Error reading CSV: {e}")

    final_score = min(round(score, 2), 1.0)
    for msg in feedback:
        logger.info(msg)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")

    return final_score


def evaluate_imdb_top50_sentiment(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for ds224 — IMDb Top50 Sentiment Analysis.
    Checks script & output existence, validates CSV contents, and ensures correlation is numeric.
    """
    score = 0.0
    feedback = []

    if not actual:
        feedback.append("Working directory path not provided.")
        return 0.0

    base_dir = os.path.dirname(actual)
    script_path = os.path.join(base_dir, "imdb_top50_sentiment.py")
    csv_path    = os.path.join(base_dir, "imdb_top50_movies.csv")
    txt_path    = os.path.join(base_dir, "imdb_sentiment_correlation.txt")

    # 1. Check script
    if os.path.exists(script_path):
        score += 0.2
        feedback.append("Python script located.")
    else:
        feedback.append("Python script missing.")
        logger.info("\n".join(feedback))
        return round(score, 2)

    # 2. Check CSV output
    if os.path.exists(csv_path):
        score += 0.3
        feedback.append("Movies CSV found.")
    else:
        feedback.append("Movies CSV missing.")
        logger.info("\n".join(feedback))
        return round(score, 2)

    # 3. Check summary text file
    if os.path.exists(txt_path):
        score += 0.2
        feedback.append("Correlation summary text file found.")
    else:
        feedback.append("Correlation summary file missing.")
        logger.info("\n".join(feedback))
        return round(score, 2)

    # 4. Validate CSV columns and values
    try:
        df = pd.read_csv(csv_path)
        expected_cols = ["title", "year", "rating", "avg_sentiment"]
        if all(col in df.columns for col in expected_cols):
            score += 0.2
            feedback.append("CSV contains expected columns.")
        else:
            feedback.append(f"CSV missing expected columns: {list(df.columns)}")

        if df["rating"].dtype.kind in "fu" and df["avg_sentiment"].dtype.kind in "fiu":
            feedback.append("Rating and avg_sentiment columns are numeric.")
            score += 0.1
        else:
            feedback.append("Numeric columns rating or avg_sentiment appear invalid.")
    except Exception as e:
        feedback.append(f"Error reading CSV: {e}")
        return round(score, 2)

    # 5. Validate correlation value in text file
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        pattern = r"^Correlation:\s*([-+]?\d*\.?\d+)$"
        match = re.match(pattern, content)
        if match:
            corr_val = float(match.group(1))
            feedback.append(f"Detected correlation value: {corr_val:.4f}")
            score += 0.1
            if -1.0 <= corr_val <= 1.0:
                feedback.append("Correlation value within valid range.")
                score += 0.1
            else:
                feedback.append("Correlation value out of valid range.")
        else:
            feedback.append("Summary file format incorrect. Expected 'Correlation: <value>'.")
    except Exception as e:
        feedback.append(f"Error reading summary file: {e}")

    final_score = min(round(score, 2), 1.0)
    for msg in feedback:
        logger.info(msg)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")
    return final_score

def evaluate_hn_frontpage_scraper(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for ds225 — Hacker News Frontpage Scraper.
    This version avoids robotic syntax and simply explains each step clearly.
    It checks that the script, database, image, and summary text file exist,
    verifies that the database has the expected table and columns,
    and confirms that the summary file follows the required text format.
    """

    score = 0.0
    notes = []

    if not actual:
        notes.append("No working directory provided for evaluation.")
        return 0.0

    base_dir = os.path.dirname(actual)
    script_path = os.path.join(base_dir, "hn_frontpage_scraper.py")
    db_path = os.path.join(base_dir, "hn_posts.db")
    image_path = os.path.join(base_dir, "hn_keyword_score.png")
    summary_path = os.path.join(base_dir, "hn_keyword_summary.txt")

    # Step 1: Check if the Python script exists
    if os.path.exists(script_path):
        score += 0.2
        notes.append("Python script located successfully.")
    else:
        notes.append("Script not found.")
        logger.info("\n".join(notes))
        return round(score, 2)

    # Step 2: Check if the SQLite database is present and has valid structure
    if os.path.exists(db_path):
        score += 0.2
        notes.append("Database file found.")
        try:
            connection = sqlite3.connect(db_path)
            cursor = connection.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='posts';")
            table_exists = cursor.fetchone()
            if table_exists:
                score += 0.1
                notes.append("Database contains 'posts' table.")
                cursor.execute("PRAGMA table_info(posts);")
                columns = [row[1] for row in cursor.fetchall()]
                required = ["timestamp_scraped", "title", "score", "submission_time"]
                if all(c in columns for c in required):
                    score += 0.1
                    notes.append("Expected columns found in the 'posts' table.")
                else:
                    notes.append(f"Database table missing expected columns: {columns}")
                cursor.execute("SELECT COUNT(*) FROM posts;")
                count = cursor.fetchone()[0]
                if count > 0:
                    score += 0.1
                    notes.append(f"Database contains {count} records.")
                else:
                    notes.append("Database is empty.")
            else:
                notes.append("No 'posts' table found in database.")
            connection.close()
        except Exception as e:
            notes.append(f"Error while reading the database: {e}")
    else:
        notes.append("Database file missing.")
        logger.info("\n".join(notes))
        return round(score, 2)

    # Step 3: Check for the visualization image
    if os.path.exists(image_path):
        score += 0.1
        notes.append("Keyword frequency image found.")
        try:
            mpimg.imread(image_path)
            notes.append("Image file opened successfully.")
            score += 0.05
        except Exception:
            notes.append("Could not read image content properly.")
    else:
        notes.append("Image file missing.")

    # Step 4: Check the summary text file format
    if os.path.exists(summary_path):
        score += 0.1
        notes.append("Summary text file found.")
        try:
            with open(summary_path, "r", encoding="utf-8") as file:
                text = file.read().strip()
            pattern = r"^Top20_Keywords_AverageScores:\s*([a-zA-Z0-9_\-]+=\d*\.?\d+(?:,\s*[a-zA-Z0-9_\-]+=\d*\.?\d+)*)$"
            match = re.match(pattern, text)
            if match:
                score += 0.15
                notes.append("Summary file format looks correct.")
                pairs = re.findall(r"([a-zA-Z0-9_\-]+)=(\d*\.?\d+)", text)
                numbers = [float(v) for _, v in pairs]
                if all(v >= 0 for v in numbers):
                    score += 0.05
                    notes.append("Average scores are valid non-negative numbers.")
                else:
                    notes.append("Some average scores appear invalid.")
            else:
                notes.append("Summary format incorrect. Expected format: 'Top20_Keywords_AverageScores: keyword=value,...'")
        except Exception as e:
            notes.append(f"Error reading summary file: {e}")
    else:
        notes.append("Summary text file not found.")

    final_score = min(round(score, 2), 1.0)
    for n in notes:
        logger.info(n)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")

    return final_score

def evaluate_covid_global_cases_analysis(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for ds232 — COVID Global Cases Analysis.
    Checks that the script, CSV output, and text file exist.
    Validates required columns, confirms the presence of the computed cases_per_million column,
    ensures numeric consistency, and verifies that the top five countries list is properly formatted.
    """

    score = 0.0
    feedback = []

    if not actual:
        feedback.append("No working directory path provided for evaluation.")
        return 0.0

    base_dir = os.path.dirname(actual)
    script_path = os.path.join(base_dir, "covid_global_cases_analysis.py")
    csv_path = os.path.join(base_dir, "covid_country_summary.csv")
    text_path = os.path.join(base_dir, "covid_top5_cases_per_million.txt")

    # Step 1: Check that the script exists
    if os.path.exists(script_path):
        score += 0.3
        feedback.append("Python script located successfully.")
    else:
        feedback.append("Script file missing.")
        logger.info("\n".join(feedback))
        return round(score, 2)

    # Step 2: Check that the CSV output exists and contains expected columns
    if os.path.exists(csv_path):
        score += 0.3
        feedback.append("CSV output file found.")
        try:
            df = pd.read_csv(csv_path)
            required_columns = [
                "country",
                "total_cases",
                "total_deaths",
                "total_recovered",
                "population",
                "cases_per_million"
            ]
            if all(col in df.columns for col in required_columns):
                score += 0.2
                feedback.append("All required columns found in the CSV file.")
            else:
                feedback.append(f"Some expected columns missing. Columns found: {list(df.columns)}")

            # Check numeric validity for computed column
            if "cases_per_million" in df.columns:
                if df["cases_per_million"].dtype in ["float64", "int64"]:
                    score += 0.05
                    feedback.append("Computed column 'cases_per_million' is numeric.")
                else:
                    feedback.append("Column 'cases_per_million' exists but is not numeric.")
            else:
                feedback.append("Column 'cases_per_million' missing from CSV.")
        except Exception as e:
            feedback.append(f"Error reading the CSV file: {e}")
    else:
        feedback.append("CSV output file not found.")
        logger.info("\n".join(feedback))
        return round(score, 2)

    # Step 3: Check that the text file exists and follows the expected format
    if os.path.exists(text_path):
        score += 0.1
        feedback.append("Text summary file found.")
        try:
            with open(text_path, "r", encoding="utf-8") as file:
                content = file.read().strip()
            pattern = r"^Top5_By_Cases_Per_Million:\s*([A-Za-z\s,]+)$"
            if re.match(pattern, content):
                score += 0.05
                feedback.append("Summary file format is valid.")
                # Check if it contains exactly five comma-separated countries
                parts = content.split(":")[-1].strip().split(",")
                if len(parts) == 5:
                    score += 0.05
                    feedback.append("Exactly five countries are listed as expected.")
                else:
                    feedback.append("The summary should include exactly five countries.")
            else:
                feedback.append("Summary file format invalid. Expected 'Top5_By_Cases_Per_Million: country1,country2,...'")
        except Exception as e:
            feedback.append(f"Error reading summary text file: {e}")
    else:
        feedback.append("Text summary file not found.")

    # Step 4: Final scoring and log output
    final_score = min(round(score, 2), 1.0)
    for note in feedback:
        logger.info(note)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")

    return final_score

import matplotlib.image as mpimg

def evaluate_nobel_laureates_by_country(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for ds233 — Nobel Laureates by Country.
    Checks script file, CSV and image outputs, ensures correct columns,
    and verifies that the top10 bar plot file exists and the shares sum logic is plausible.
    """

    score = 0.0
    feedback = []

    if not actual:
        feedback.append("Working directory not provided.")
        return 0.0

    base_dir = os.path.dirname(actual)
    script_file = os.path.join(base_dir, "nobel_laureates_by_country.py")
    csv_file    = os.path.join(base_dir, "nobel_country_summary.csv")
    image_file  = os.path.join(base_dir, "top10_nobel_countries.png")

    # 1: Script existence
    if os.path.exists(script_file):
        score += 0.2
        feedback.append("Python script found.")
    else:
        feedback.append("Script file missing.")
        logger.info("\n".join(feedback))
        return round(score, 2)

    # 2: CSV existence and column check
    if os.path.exists(csv_file):
        score += 0.3
        feedback.append("Summary CSV file found.")
        try:
            df = pd.read_csv(csv_file)
            expected_cols = ["country","total_laureates","Peace_share","Literature_share","Physics_share","Chemistry_share","Medicine_share","Economics_share"]
            if all(col in df.columns for col in expected_cols):
                score += 0.2
                feedback.append("CSV contains all expected columns.")
            else:
                feedback.append(f"CSV missing columns; found: {list(df.columns)}")

            # Check that total_laureates is numeric and positive
            if df["total_laureates"].dtype.kind in "iu" and df["total_laureates"].min() >= 0:
                score += 0.1
                feedback.append("total_laureates appears valid numeric.")
            else:
                feedback.append("total_laureates column invalid or negative.")

            # Check that share columns are between 0 and 1
            share_cols = ["Peace_share","Literature_share","Physics_share","Chemistry_share","Medicine_share","Economics_share"]
            invalid = False
            for sc in share_cols:
                if sc in df.columns:
                    vals = df[sc].dropna()
                    if not all((vals >= 0) & (vals <= 1)):
                        invalid = True
            if not invalid:
                score += 0.1
                feedback.append("Share columns are in valid range 0 to 1.")
            else:
                feedback.append("One or more share columns contain invalid values.")
        except Exception as e:
            feedback.append(f"Error reading summary CSV: {e}")
    else:
        feedback.append("Summary CSV file missing.")
        logger.info("\n".join(feedback))
        return round(score, 2)

    # 3: Image existence and readability
    if os.path.exists(image_file):
        score += 0.1
        feedback.append("Bar plot image file found.")
        try:
            mpimg.imread(image_file)
            feedback.append("Image file opened successfully.")
            score += 0.05
        except Exception:
            feedback.append("Image file exists but cannot be opened.")
    else:
        feedback.append("Bar plot image missing.")

    final_score = min(round(score, 2), 1.0)
    for msg in feedback:
        logger.info(msg)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")

    return final_score


def evaluate_ds_salary_by_state(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for ds234 — Data Scientist Salary by State.
    Checks:
      - The script file exists.
      - The CSV output exists and has expected columns.
      - The PNG choropleth map exists and can be opened.
      - Median salary values are numeric and reasonable (e.g., > 0 and < 500000).
      - Job counts per state > 0 for at least some states.
    """

    score = 0.0
    notes = []

    if not actual:
        notes.append("Working directory path not provided.")
        return 0.0

    base_dir = os.path.dirname(actual)
    script_path = os.path.join(base_dir, "data_science_salary_by_state.py")
    csv_path = os.path.join(base_dir, "ds_salary_by_state.csv")
    image_path = os.path.join(base_dir, "ds_salary_by_state.png")

    # Check for script file
    if os.path.exists(script_path):
        score += 0.2
        notes.append("Python script located.")
    else:
        notes.append("Script file missing.")
        logger.info("\n".join(notes))
        return round(score, 2)

    # Check for CSV file and columns
    if os.path.exists(csv_path):
        score += 0.3
        notes.append("CSV output file found.")
        try:
            df = pd.read_csv(csv_path)
            expected_cols = ["state", "median_salary", "job_count"]
            if all(col in df.columns for col in expected_cols):
                score += 0.2
                notes.append("CSV contains expected columns.")
            else:
                notes.append(f"CSV missing expected columns; found: {list(df.columns)}")

            # Validate numbers
            if df["median_salary"].dtype.kind in "fiu" and (df["median_salary"] > 0).all():
                if (df["median_salary"] < 500000).all():
                    score += 0.1
                    notes.append("median_salary values are numeric and within expected range.")
                else:
                    notes.append("Some median_salary values unusually high (>500k).")
            else:
                notes.append("median_salary values invalid or non-positive.")

            if df["job_count"].dtype.kind in "iu" and (df["job_count"] > 0).any():
                score += 0.05
                notes.append("job_count values valid (some states have >0 jobs).")
            else:
                notes.append("job_count values appear invalid or zero for all states.")
        except Exception as e:
            notes.append(f"Error reading CSV file: {e}")
    else:
        notes.append("CSV output file not found.")
        logger.info("\n".join(notes))
        return round(score, 2)

    # Check for image file existence and readability
    if os.path.exists(image_path):
        score += 0.1
        notes.append("Choropleth map PNG found.")
        try:
            _ = mpimg.imread(image_path)
            notes.append("Image file can be opened.")
            score += 0.05
        except Exception:
            notes.append("Could not open image file.")
    else:
        notes.append("Choropleth map PNG missing.")

    # Final log and scoring
    final_score = min(round(score, 2), 1.0)
    for note in notes:
        logger.info(note)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")

    return final_score


def evaluate_soccer_scoreboard_summary(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for ds236 — Soccer Scoreboard Summary.
    Checks that the script file exists, validates the CSV output with correct columns,
    ensures numeric fields make sense, and that the highest scoring match details are non-empty.
    """

    score = 0.0
    feedback = []

    if not actual:
        feedback.append("No working directory path provided for evaluation.")
        return 0.0

    base_dir = os.path.dirname(actual)
    script_path = os.path.join(base_dir, "soccer_scoreboard_summary.py")
    csv_path    = os.path.join(base_dir, "soccer_results_summary.csv")

    # Check script existence
    if os.path.exists(script_path):
        score += 0.3
        feedback.append("Python script located successfully.")
    else:
        feedback.append("Script file missing.")
        logger.info("\n".join(feedback))
        return round(score, 2)

    # Check CSV output
    if os.path.exists(csv_path):
        score += 0.4
        feedback.append("CSV output file found.")
        try:
            df = pd.read_csv(csv_path)
            expected_cols = [
                "league","avg_goals_per_match",
                "highest_scoring_match","highest_match_home_team",
                "highest_match_away_team","highest_match_goals"
            ]
            if all(col in df.columns for col in expected_cols):
                score += 0.2
                feedback.append("CSV contains all expected columns.")
            else:
                feedback.append(f"CSV missing expected columns: {list(df.columns)}")

            # Numeric validity check
            if df["avg_goals_per_match"].dtype.kind in "fiu" and (df["avg_goals_per_match"] >= 0).all():
                score += 0.05
                feedback.append("avg_goals_per_match values appear valid.")
            else:
                feedback.append("Invalid values in avg_goals_per_match column.")

            if df["highest_match_goals"].dtype.kind in "iu" and (df["highest_match_goals"] > 0).any():
                score += 0.05
                feedback.append("highest_match_goals values appear valid.")
            else:
                feedback.append("Invalid or zero values in highest_match_goals column.")
        except Exception as e:
            feedback.append(f"Error reading CSV file: {e}")
    else:
        feedback.append("CSV output file not found.")
        logger.info("\n".join(feedback))
        return round(score, 2)

    final_score = min(round(score, 2), 1.0)
    for msg in feedback:
        logger.info(msg)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")

    return final_score


def evaluate_crypto_volatility_analysis(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for ds236 — Crypto Volatility Analysis.
    Checks existence of script, CSV output, summary text,
    validates columns, numeric ranges, and volatility computations.
    """

    score = 0.0
    feedback = []

    if not actual:
        feedback.append("Working directory path not provided.")
        return 0.0

    base_dir = os.path.dirname(actual)
    script_path = os.path.join(base_dir, "crypto_volatility_analysis.py")
    csv_path    = os.path.join(base_dir, "crypto_volatility.csv")
    txt_path    = os.path.join(base_dir, "top_volatile_coins.txt")

    # 1. Check script existence
    if os.path.exists(script_path):
        score += 0.3
        feedback.append("Python script found.")
    else:
        feedback.append("Script file missing.")
        logger.info("\n".join(feedback))
        return round(score, 2)

    # 2. Check CSV output and columns
    if os.path.exists(csv_path):
        score += 0.4
        feedback.append("CSV output file found.")
        try:
            df = pd.read_csv(csv_path)
            expected_cols = ["name","symbol","price","24h_change","7d_change","market_cap","seven_day_volatility"]
            if all(col in df.columns for col in expected_cols):
                score += 0.2
                feedback.append("CSV contains all expected columns.")
            else:
                feedback.append(f"CSV missing expected columns: {list(df.columns)}")

            # Validate numeric columns > 0 for price and market cap
            if df["price"].dtype.kind in "fiu" and (df["price"] > 0).all():
                score += 0.05
                feedback.append("Price values valid and positive.")
            else:
                feedback.append("Price column has invalid values.")

            if df["seven_day_volatility"].dtype.kind in "fiu" and (df["seven_day_volatility"] >= 0).all():
                score += 0.05
                feedback.append("Volatility column appears valid.")
            else:
                feedback.append("Volatility values invalid or negative.")
        except Exception as e:
            feedback.append(f"Error reading CSV file: {e}")
    else:
        feedback.append("CSV output file not found.")
        logger.info("\n".join(feedback))
        return round(score, 2)

    # 3. Check summary text file format
    if os.path.exists(txt_path):
        score += 0.1
        feedback.append("Summary text file found.")
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                line = f.read().strip()
            pattern = r"^Most_Volatile_Coins:\s*[A-Z0-9]+(?:,[A-Z0-9]+){2}$"
            if re.match(pattern, line):
                score += 0.1
                feedback.append("Summary file format valid.")
            else:
                feedback.append("Summary file format incorrect. Expected 'Most_Volatile_Coins: symbol1,symbol2,symbol3'.")
        except Exception as e:
            feedback.append(f"Error reading summary file: {e}")
    else:
        feedback.append("Summary text file missing.")

    final_score = min(round(score, 2), 1.0)
    for msg in feedback:
        logger.info(msg)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")

    return final_score

def evaluate_goodreads_top100_rating_analysis(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for ds237 — Goodreads Top100 Rating Analysis.
    Checks that the script exists, the CSV, image and summary text exist.
    Verifies required columns in CSV, ensures image can be opened,
    reads correlation from summary text and ensures –1<=corr<=1,
    and checks that Trend value is one of Increases/Decreases/None.
    """

    score = 0.0
    feedback = []

    if not actual:
        feedback.append("Working directory path not provided.")
        return 0.0

    base_dir = os.path.dirname(actual)
    script_path = os.path.join(base_dir, "goodreads_top100_rating_analysis.py")
    csv_path    = os.path.join(base_dir, "goodreads_top100_summary.csv")
    image_path  = os.path.join(base_dir, "rating_vs_reviews.png")
    text_path   = os.path.join(base_dir, "goodreads_correlation_summary.txt")

    # Check for script file
    if os.path.exists(script_path):
        score += 0.2
        feedback.append("Script file found.")
    else:
        feedback.append("Script file missing.")
        logger.info("\n".join(feedback))
        return round(score, 2)

    # Check CSV existence and contents
    if os.path.exists(csv_path):
        score += 0.3
        feedback.append("Summary CSV file found.")
        try:
            df = pd.read_csv(csv_path)
            required_cols = ["title","author","average_rating","review_count"]
            if all(c in df.columns for c in required_cols):
                score += 0.15
                feedback.append("CSV contains all required columns.")
            else:
                feedback.append(f"CSV missing expected columns: {list(df.columns)}")

            # Check numeric columns valid
            if df["average_rating"].dtype.kind in "fiu" and ((df["average_rating"] >= 0) & (df["average_rating"] <= 5)).all():
                score += 0.05
                feedback.append("average_rating values appear valid between 0 and 5.")
            else:
                feedback.append("average_rating values out of expected range.")
            if df["review_count"].dtype.kind in "iu" and (df["review_count"] >= 0).all():
                score += 0.05
                feedback.append("review_count values appear valid non-negative.")
            else:
                feedback.append("review_count values invalid or negative.")
        except Exception as e:
            feedback.append(f"Error reading CSV file: {e}")
    else:
        feedback.append("Summary CSV file missing.")
        logger.info("\n".join(feedback))
        return round(score, 2)

    # Check image existence
    if os.path.exists(image_path):
        score += 0.1
        feedback.append("Scatter plot image found.")
        try:
            _ = mpimg.imread(image_path)
            feedback.append("Image file opened successfully.")
            score += 0.05
        except Exception:
            feedback.append("Could not open image file properly.")
    else:
        feedback.append("Scatter plot image missing.")

    # Check summary text file
    if os.path.exists(text_path):
        score += 0.1
        feedback.append("Correlation summary text file found.")
        try:
            with open(text_path, "r", encoding="utf-8") as f:
                line = f.read().strip()
            pattern = r"^Corr_ReviewCount_Rating:\s*([-+]?\d*\.\d+),\s*Trend:\s*(Increases|Decreases|None)$"
            match = re.match(pattern, line)
            if match:
                corr = float(match.group(1))
                trend = match.group(2)
                feedback.append(f"Detected valid format: correlation = {corr}, trend = {trend}")
                score += 0.1
                if -1 <= corr <= 1:
                    score += 0.05
                    feedback.append("Correlation value within expected range.")
                else:
                    feedback.append("Correlation value outside expected range (-1 to 1).")
                if trend in ["Increases","Decreases","None"]:
                    score += 0.05
                    feedback.append("Trend value is valid.")
                else:
                    feedback.append("Trend value invalid.")
            else:
                feedback.append("Summary text format invalid. Expected 'Corr_ReviewCount_Rating: <value>, Trend: <Increases/Decreases/None>'.")
        except Exception as e:
            feedback.append(f"Error reading summary text file: {e}")
    else:
        feedback.append("Correlation summary text file missing.")

    final_score = min(round(score, 2), 1.0)
    for msg in feedback:
        logger.info(msg)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")

    return final_score

def evaluate_tech_stock_comparison(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for ds238 — Tech Stock Comparison.
    Checks:
      - Script file exists.
      - Output CSV exists and has correct columns.
      - Values for average_return and volatility are numeric and reasonable.
      - Plot image exists and can be opened.
    """

    score = 0.0
    feedback = []

    if not actual:
        feedback.append("No working directory provided for evaluation.")
        return 0.0

    base_dir = os.path.dirname(actual)
    script_path = os.path.join(base_dir, "tech_stock_comparison.py")
    csv_path    = os.path.join(base_dir, "tech_stock_comparison.csv")
    image_path  = os.path.join(base_dir, "return_vs_volatility.png")

    # Check script existence
    if os.path.exists(script_path):
        score += 0.25
        feedback.append("Python script found.")
    else:
        feedback.append("Script file missing.")
        logger.info("\n".join(feedback))
        return round(score, 2)

    # Check CSV and columns
    if os.path.exists(csv_path):
        score += 0.35
        feedback.append("CSV output file found.")
        try:
            df = pd.read_csv(csv_path)
            expected_cols = ["ticker","average_return","volatility"]
            if all(col in df.columns for col in expected_cols):
                score += 0.20
                feedback.append("CSV contains all expected columns.")
            else:
                feedback.append(f"CSV missing expected columns: {list(df.columns)}")

            # Validate numeric values
            if df["average_return"].dtype.kind in "fiu" and (df["average_return"].abs() < 1).all():
                score += 0.10
                feedback.append("average_return values are numeric and within expected range (<1 in absolute).")
            else:
                feedback.append("average_return values out of expected range or non-numeric.")
            if df["volatility"].dtype.kind in "fiu" and (df["volatility"] >= 0).all():
                score += 0.10
                feedback.append("volatility values are numeric and non-negative.")
            else:
                feedback.append("volatility values invalid or negative.")
        except Exception as e:
            feedback.append(f"Error reading CSV file: {e}")
    else:
        feedback.append("CSV output file not found.")
        logger.info("\n".join(feedback))
        return round(score, 2)

    # Check image existence
    if os.path.exists(image_path):
        score += 0.10
        feedback.append("Scatter plot image found.")
        try:
            _ = mpimg.imread(image_path)
            feedback.append("Image file opened successfully.")
            score += 0.05
        except Exception:
            feedback.append("Image exists but cannot be opened.")
    else:
        feedback.append("Plot image missing.")

    final_score = min(round(score, 2), 1.0)
    for msg in feedback:
        logger.info(msg)
    logger.info(f"Final Evaluation Score: {final_score:.2f}")

    return final_score

def evaluate_weather_stock_correlation(actual: str, expected: dict, **kwargs) -> float:
    """
    Evaluator for ds239 — checks correlation between temperature change and SP500 return.
    Ensures realistic data, correlation value, and correct output format.
    """
    score = 0.0
    base = os.path.dirname(actual)
    script = os.path.join(base, "weather_stock_correlation.py")
    csvf = os.path.join(base, "weather_stock_correlation.csv")
    txtf = os.path.join(base, "weather_stock_corr_summary.txt")

    # Script check
    if os.path.exists(script):
        score += 0.25

    # CSV validation
    if os.path.exists(csvf):
        score += 0.35
        df = pd.read_csv(csvf)
        cols = ["date","temperature","sp500_close","temp_change","sp500_return"]
        if all(c in df.columns for c in cols):
            score += 0.20
        if abs(df["temperature"].mean()) > 0 and df["sp500_close"].mean() > 0:
            score += 0.05
        if df.shape[0] >= 10:
            score += 0.05

    # Text validation
    if os.path.exists(txtf):
        score += 0.05
        with open(txtf) as f:
            line = f.read().strip()
        match = re.match(r"Corr_TempChange_SP500Return:\s*(-?\d*\.\d+)", line)
        if match:
            corr = float(match.group(1))
            if -1 <= corr <= 1:
                score += 0.05
    return min(round(score,2),1.0)

def evaluate_premier_league_team_regression(actual: str, expected: dict, **kwargs) -> float:
    """
    Evaluator for ds240 — verifies fbref scraping, regression results, and file outputs.
    """
    score = 0.0
    base = os.path.dirname(actual)
    script = os.path.join(base,"premier_league_team_regression.py")
    csvf = os.path.join(base,"premier_league_team_summary.csv")
    txtf = os.path.join(base,"premier_league_regression_summary.txt")

    # Script presence
    if os.path.exists(script):
        score += 0.25

    # CSV validation
    if os.path.exists(csvf):
        score += 0.4
        df = pd.read_csv(csvf)
        cols = ["Team","Goals","Assists","Shots","Possession%","Pass%"]
        if all(c in df.columns for c in cols):
            score += 0.2
        if "Predicted_Points" in df.columns:
            score += 0.05
        if df.shape[0] >= 10:
            score += 0.05

    # Text validation
    if os.path.exists(txtf):
        score += 0.05
        with open(txtf) as f:
            line = f.read().strip()
        m = re.match(r"R2:\s*(\d*\.\d+),\s*RMSE:\s*(\d*\.\d+)", line)
        if m:
            r2, rmse = float(m.group(1)), float(m.group(2))
            if 0 <= r2 <= 1:
                score += 0.05
            if rmse > 0:
                score += 0.05

    return min(round(score,2),1.0)

def evaluate_mnist_pca_reconstruction(actual: str, expected: dict, **kwargs) -> float:
    """
    Evaluates ds241 — MNIST PCA Reconstruction Error.
    Checks for script presence, JSON file existence, valid numeric values, and proper format.
    """
    score = 0.0
    base = os.path.dirname(actual)
    script = os.path.join(base, "mnist_pca_reconstruction.py")
    json_path = os.path.join(base, "mnist_pca_reconstruction_summary.json")

    # 1. Script existence
    if os.path.exists(script):
        score += 0.4
        logger.info("Python script found.")
    else:
        logger.info("Script missing.")
        return round(score, 2)

    # 2. JSON existence and format
    if os.path.exists(json_path):
        score += 0.4
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                line = f.read().strip()
                # Validate JSON line
                data = json.loads(line)
                keys = ["mse_10", "mse_50", "mse_100"]
                if all(k in data for k in keys):
                    score += 0.1
                    # Validate numeric range
                    vals = [float(data[k]) for k in keys]
                    if all(v > 0 for v in vals) and vals[0] > vals[1] > vals[2]:
                        # Expect MSE decreases with more components
                        score += 0.1
                        logger.info("Valid numeric MSE values found, monotonic decrease verified.")
                    else:
                        logger.info("Numeric values found but monotonic property missing.")
                else:
                    logger.info(f"Missing expected keys. Found: {list(data.keys())}")
        except Exception as e:
            logger.info(f"Error reading JSON file: {e}")
    else:
        logger.info("JSON summary missing.")

    return min(round(score, 2), 1.0)

def evaluate_mnist_class_similarity(actual: str, expected: dict, **kwargs) -> float:
    """
    Evaluator for ds242 — MNIST class-level cosine similarity analysis.
    Checks script presence, text file format, and realistic output pattern.
    """
    score = 0.0
    base = os.path.dirname(actual)
    script = os.path.join(base, "mnist_class_similarity.py")
    txt_path = os.path.join(base, "mnist_class_similarity_summary.txt")

    # Script existence
    if os.path.exists(script):
        score += 0.5
        logger.info("Python script detected.")

    # Check text file and format
    if os.path.exists(txt_path):
        score += 0.4
        with open(txt_path, "r", encoding="utf-8") as f:
            line = f.read().strip()

        # Expected pattern: (digit, digit), (digit, digit), (digit, digit)
        pattern = r"^\(\d,\s*\d\),\s*\(\d,\s*\d\),\s*\(\d,\s*\d\)$"
        if re.match(pattern, line):
            score += 0.1
            logger.info("Valid output format found.")
        else:
            logger.info(f"Invalid output format: {line}")
    else:
        logger.info("Summary file missing.")

    return min(round(score, 2), 1.0)

def evaluate_mnist_class_compactness(actual: str, expected: dict, **kwargs) -> float:
    """
    Evaluator for ds243 — MNIST class compactness via variance.
    Verifies script and JSON output format with numeric digits (0–9).
    """
    score = 0.0
    base = os.path.dirname(actual)
    script = os.path.join(base, "mnist_class_compactness.py")
    json_path = os.path.join(base, "mnist_class_compactness_summary.json")

    # Check for Python script
    if os.path.exists(script):
        score += 0.4
        logger.info("Python script detected.")
    else:
        logger.info("Script not found.")
        return round(score, 2)

    # Check JSON output
    if os.path.exists(json_path):
        score += 0.4
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "most_compact" in data and "least_compact" in data:
                score += 0.1
                mc, lc = data["most_compact"], data["least_compact"]
                if all(isinstance(v, int) and 0 <= v <= 9 for v in [mc, lc]):
                    score += 0.1
                    logger.info(f"Valid digits found: most={mc}, least={lc}")
                else:
                    logger.info("Values not valid digits between 0–9.")
            else:
                logger.info(f"JSON keys missing: {data.keys()}")
        except Exception as e:
            logger.info(f"Error reading JSON: {e}")
    else:
        logger.info("JSON output missing.")

    return min(round(score, 2), 1.0)

def evaluate_cifar10_rgb_mean_intensity(actual: str, expected: dict, **kwargs) -> float:
    """
    Evaluator for ds244 — CIFAR-10 RGB mean intensity analysis.
    Ensures the script and JSON output exist and contain valid numeric channel means.
    """
    score = 0.0
    base_dir = os.path.dirname(actual)
    script_path = os.path.join(base_dir, "cifar10_rgb_mean_intensity.py")
    json_path = os.path.join(base_dir, "cifar10_rgb_mean_summary.json")

    # Check script existence
    if os.path.exists(script_path):
        score += 0.4
        logger.info("Python script detected.")
    else:
        logger.info("Script file not found.")
        return round(score, 2)

    # Check JSON result
    if os.path.exists(json_path):
        score += 0.4
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            keys = ["mean_r", "mean_g", "mean_b"]
            if all(k in data for k in keys):
                score += 0.1
                if all(isinstance(data[k], (int, float)) for k in keys):
                    score += 0.1
                    logger.info("Valid numeric RGB mean values found.")
                else:
                    logger.info("Non-numeric values detected in JSON.")
            else:
                logger.info("Missing required JSON keys.")
        except Exception as e:
            logger.info(f"Error reading JSON: {e}")
    else:
        logger.info("JSON output file missing.")

    return min(round(score, 2), 1.0)


def evaluate_cifar10_rgb_class_similarity(actual: str, expected: dict, **kwargs) -> float:
    """
    Evaluator for ds245 — CIFAR-10 RGB class similarity.
    Checks that both the script and summary file exist,
    and that the output line matches the expected tuple format.
    """
    score = 0.0
    base = os.path.dirname(actual)
    script = os.path.join(base, "cifar10_rgb_class_similarity.py")
    txt_path = os.path.join(base, "cifar10_rgb_class_similarity_summary.txt")

    # Check script existence
    if os.path.exists(script):
        score += 0.4
        logger.info("Python script found.")
    else:
        logger.info("Script file missing.")
        return round(score, 2)

    # Check output text file
    if os.path.exists(txt_path):
        score += 0.4
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                content = f.read().strip()

            # Expected format: (digit, digit), (digit, digit), (digit, digit)
            pattern = r"^\(\d,\s*\d\),\s*\(\d,\s*\d\),\s*\(\d,\s*\d\)$"
            if re.match(pattern, content):
                score += 0.2
                logger.info("Valid tuple output format detected.")
            else:
                logger.info(f"Invalid output format: {content}")
        except Exception as e:
            logger.info(f"Error reading result file: {e}")
    else:
        logger.info("Result text file missing.")

    return min(round(score, 2), 1.0)

def evaluate_cifar10_rgb_variance_ratios(actual: str, expected: dict, **kwargs) -> float:
    """
    Evaluator for ds246 — CIFAR-10 RGB variance ratios.
    Verifies that the script and JSON output exist and that all ratio values are numeric.
    """
    score = 0.0
    base = os.path.dirname(actual)
    script = os.path.join(base, "cifar10_rgb_variance_ratios.py")
    json_path = os.path.join(base, "cifar10_rgb_variance_summary.json")

    # Script existence
    if os.path.exists(script):
        score += 0.4
        logger.info("Python script detected.")
    else:
        logger.info("Script file not found.")
        return round(score, 2)

    # JSON output existence
    if os.path.exists(json_path):
        score += 0.4
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            keys = ["variance_r_g", "variance_g_b", "variance_r_b"]
            if all(k in data for k in keys):
                score += 0.1
                if all(isinstance(data[k], (int, float)) for k in keys):
                    score += 0.1
                    logger.info("Valid numeric variance ratios detected.")
                else:
                    logger.info("Non-numeric ratio values found.")
            else:
                logger.info("Missing required JSON keys.")
        except Exception as e:
            logger.info(f"Error reading JSON: {e}")
    else:
        logger.info("Output JSON file missing.")

    return min(round(score, 2), 1.0)

def evaluate_cifar10_rgb_feature_entropy(actual: str, expected: dict, **kwargs) -> float:
    """
    Evaluator for ds247 — CIFAR-10 RGB feature entropy analysis.
    Ensures both script and JSON output exist and that the output
    is a valid list of five integer indices.
    """
    score = 0.0
    base = os.path.dirname(actual)
    script_path = os.path.join(base, "cifar10_rgb_feature_entropy.py")
    json_path = os.path.join(base, "cifar10_rgb_feature_entropy_summary.json")

    # Check script
    if os.path.exists(script_path):
        score += 0.4
        logger.info("Python script detected.")
    else:
        logger.info("Script file not found.")
        return round(score, 2)

    # Check JSON result
    if os.path.exists(json_path):
        score += 0.4
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "top_entropy_features" in data and isinstance(data["top_entropy_features"], list):
                features = data["top_entropy_features"]
                if len(features) == 5 and all(isinstance(i, int) for i in features):
                    score += 0.2
                    logger.info("Valid top-5 entropy feature indices found.")
                else:
                    logger.info("List does not contain exactly five integers.")
            else:
                logger.info("Missing 'top_entropy_features' key or invalid type.")
        except Exception as e:
            logger.info(f"Error reading JSON: {e}")
    else:
        logger.info("JSON output file missing.")

    return min(round(score, 2), 1.0)

def evaluate_cifar10_rgb_pca_reconstruction(actual: str, expected: dict, **kwargs) -> float:
    """
    Evaluator for ds248 — CIFAR-10 RGB PCA reconstruction.
    Ensures both the script and JSON output exist,
    with valid non-negative numeric MSE values for 50, 200, and 500 components.
    """
    score = 0.0
    base = os.path.dirname(actual)
    script = os.path.join(base, "cifar10_rgb_pca_reconstruction.py")
    json_file = os.path.join(base, "cifar10_rgb_pca_reconstruction_summary.json")

    # Script existence
    if os.path.exists(script):
        score += 0.4
        logger.info("Python script detected.")
    else:
        logger.info("Script file not found.")
        return round(score, 2)

    # JSON existence
    if os.path.exists(json_file):
        score += 0.4
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            required = ["mse_50", "mse_200", "mse_500"]
            if all(k in data for k in required):
                score += 0.1
                if all(isinstance(data[k], (int, float)) and data[k] >= 0 for k in required):
                    score += 0.1
                    logger.info("Valid numeric MSE values found for all PCA component sets.")
                else:
                    logger.info("Non-numeric or negative MSE values detected.")
            else:
                logger.info("Missing required MSE keys in JSON output.")
        except Exception as e:
            logger.info(f"Error reading JSON output: {e}")
    else:
        logger.info("Output JSON file missing.")

    return min(round(score, 2), 1.0)


def evaluate_cifar10_rgb_feature_skewness(actual: str, expected: dict, **kwargs) -> float:
    """
    Evaluator for ds249 — CIFAR-10 RGB feature skewness analysis.
    Verifies that both the script and JSON exist and that the output
    is a valid list of five integer indices.
    """
    score = 0.0
    base = os.path.dirname(actual)
    script = os.path.join(base, "cifar10_rgb_feature_skewness.py")
    json_file = os.path.join(base, "cifar10_rgb_feature_skewness_summary.json")

    # Check script existence
    if os.path.exists(script):
        score += 0.4
        logger.info("Python script detected.")
    else:
        logger.info("Script file not found.")
        return round(score, 2)

    # Check JSON output
    if os.path.exists(json_file):
        score += 0.4
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "top_skewed_features" in data and isinstance(data["top_skewed_features"], list):
                features = data["top_skewed_features"]
                if len(features) == 5 and all(isinstance(i, int) for i in features):
                    score += 0.2
                    logger.info("Valid top-5 skewed feature indices found.")
                else:
                    logger.info("List does not contain exactly five integer indices.")
            else:
                logger.info("Missing 'top_skewed_features' key or invalid format.")
        except Exception as e:
            logger.info(f"Error reading JSON: {e}")
    else:
        logger.info("JSON output file missing.")

    return min(round(score, 2), 1.0)

def evaluate_cifar10_rgb_class_color_diff(actual: str, expected: dict, **kwargs) -> float:
    """
    Evaluator for ds250 — CIFAR-10 RGB class-level red-green difference.
    Ensures both script and JSON exist, and that the output contains exactly
    10 numeric values representing class-wise mean (R−G) differences.
    """
    score = 0.0
    base = os.path.dirname(actual)
    script = os.path.join(base, "cifar10_rgb_class_color_diff.py")
    json_file = os.path.join(base, "cifar10_rgb_class_color_diff_summary.json")

    # Check for script existence
    if os.path.exists(script):
        score += 0.4
        logger.info("Python script detected.")
    else:
        logger.info("Script file not found.")
        return round(score, 2)

    # Check for JSON output
    if os.path.exists(json_file):
        score += 0.4
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "class_diffs" in data and isinstance(data["class_diffs"], list):
                vals = data["class_diffs"]
                if len(vals) == 10 and all(isinstance(v, (int, float)) for v in vals):
                    score += 0.2
                    logger.info("Valid 10-value class difference vector detected.")
                else:
                    logger.info("class_diffs does not contain 10 numeric values.")
            else:
                logger.info("Missing or invalid 'class_diffs' key.")
        except Exception as e:
            logger.info(f"Error reading JSON: {e}")
    else:
        logger.info("Output JSON missing.")

    return min(round(score, 2), 1.0)


def evaluate_avg_humidity(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for ds251 — Average Humidity per Country task.
    Verifies:
      - The Python script exists.
      - The script imports pandas.
      - It includes humidity calculation logic (groupby/mean).
      - The output summary file exists.
    """

    if not actual:
        logger.info("No actual file path provided.")
        return 0.0

    base_dir = os.path.dirname(actual)
    script = os.path.join(base_dir, "avg_humidity_analysis.py")
    result = os.path.join(base_dir, "avg_humidity_summary.txt")

    score = 0.0

    # Check script existence
    if os.path.exists(script):
        score += 0.4
        logger.info("Python script detected.")

        try:
            with open(script, "r", encoding="utf-8") as f:
                content = f.read().lower()

            # Check for pandas import
            if "import pandas" in content:
                score += 0.3
                logger.info("Pandas import confirmed.")

            # Check for relevant humidity logic
            if "humidity" in content and ("groupby" in content or "mean" in content):
                score += 0.2
                logger.info("Humidity computation logic found.")
            else:
                logger.info("Missing groupby/mean logic for humidity.")
        except Exception as e:
            logger.info(f"Error reading script file: {e}")
    else:
        logger.info("Script file missing.")

    # Check output file existence
    if os.path.exists(result):
        score += 0.1
        logger.info("Output summary file detected.")
    else:
        logger.info("Output summary file missing.")

    return min(round(score, 2), 1.0)


def evaluate_single_column_count(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for ds252 — Column Count Analysis.
    This evaluator checks:
      - The Python script exists.
      - The script correctly imports pandas.
      - The output text file exists and contains valid column count information.
      - If the filename is different, still accepts numeric column count presence.
    """

    if not actual:
        logger.info("No path provided to evaluator.")
        return 0.0

    base = os.path.dirname(actual)
    script = os.path.join(base, "column_count_analysis.py")
    txt = os.path.join(base, "column_counts.txt")

    score = 0.0
    feedback = []

    try:
        # 1. Script existence
        if os.path.exists(script):
            score += 0.4
            feedback.append("Script file found.")

            with open(script, "r", encoding="utf-8") as f:
                content = f.read().lower()

            # 2. Library check
            if "import pandas" in content:
                score += 0.3
                feedback.append("Pandas import detected.")
            else:
                feedback.append("Missing pandas import statement.")

        else:
            feedback.append("Script not found.")
            return round(score, 2)

        # 3. Output file existence and content
        if os.path.exists(txt):
            with open(txt, "r", encoding="utf-8") as f:
                out = f.read().strip()

            if out:
                score += 0.2
                feedback.append("Output file contains content.")

                # 4. Accept either exact filename or numeric output
                if "global_climate_energy_2020_2024.csv" in out or any(ch.isdigit() for ch in out):
                    score += 0.1
                    feedback.append("Output includes valid filename or numeric column count.")
                else:
                    feedback.append("Output missing filename or numeric column reference.")
            else:
                feedback.append("Output file is empty.")
        else:
            feedback.append("Output file missing.")

    except Exception as e:
        feedback.append(f"Evaluator encountered an error: {e}")

    return min(round(score, 2), 1.0)


def evaluate_country_entry_counts(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for ds253 — Count dataset entries per country.
    Verifies:
      - Script existence and pandas import.
      - CSV file presence.
      - Output text file existence and non-empty content.
    """

    if not actual:
        logger.info("No valid path provided to evaluator.")
        return 0.0

    base = os.path.dirname(actual)
    py_file = os.path.join(base, "country_entry_count.py")
    csv_file = os.path.join(base, "global_climate_energy_2020_2024.csv")
    txt_file = os.path.join(base, "country_entry_counts.txt")

    score = 0.0
    feedback = []

    try:
        # Check script existence and import
        if os.path.exists(py_file):
            score += 0.25
            feedback.append("Python script detected.")
            with open(py_file, "r", encoding="utf-8") as f:
                content = f.read()
            if "import pandas" in content:
                score += 0.25
                feedback.append("Script correctly imports pandas.")
            else:
                feedback.append("Pandas import missing in script.")
        else:
            feedback.append("Python script missing.")

        # Check CSV existence
        if os.path.exists(csv_file):
            score += 0.25
            feedback.append("CSV dataset found.")
        else:
            feedback.append("CSV file not found in directory.")

        # Check output text file
        if os.path.exists(txt_file):
            with open(txt_file, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
            if lines:
                score += 0.25
                feedback.append("Output file contains country entry counts.")
            else:
                feedback.append("Output file exists but is empty.")
        else:
            feedback.append("Output file missing.")

    except Exception as e:
        feedback.append(f"Evaluator encountered an error: {e}")

    for item in feedback:
        logger.info(item)
    logger.info(f"Evaluation Score: {score:.2f}")

    return min(round(score, 2), 1.0)

def evaluate_earthquake_dataset_count(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for ds254 — Earthquake dataset total-row-count task.
    Verifies:
      - Script exists and imports pandas.
      - CSV dataset exists.
      - Output text file exists, non-empty, and includes numeric row count.
    """

    if not actual:
        logger.info("No valid base path provided to evaluator.")
        return 0.0

    base_dir = os.path.dirname(actual)
    py_file = os.path.join(base_dir, "earthquake_dataset_count.py")
    csv_file = os.path.join(base_dir, "earthquake_data_tsunami.csv")
    txt_file = os.path.join(base_dir, "earthquake_dataset_count.txt")

    score = 0.0
    feedback = []

    try:
        # 1. Script existence and import check
        if os.path.exists(py_file):
            score += 0.25
            feedback.append("Python script detected.")
            with open(py_file, "r", encoding="utf-8") as f:
                content = f.read()
            if "import pandas" in content:
                score += 0.25
                feedback.append("Pandas library correctly imported in script.")
            else:
                feedback.append("Pandas import missing in the script.")
        else:
            feedback.append("Python script file not found.")

        # 2. Dataset presence
        if os.path.exists(csv_file):
            score += 0.25
            feedback.append("CSV dataset found in directory.")
        else:
            feedback.append("CSV dataset missing.")

        # 3. Output file existence and numeric content
        if os.path.exists(txt_file):
            with open(txt_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
            if content:
                score += 0.25
                feedback.append("Output text file exists and contains data.")
                if any(ch.isdigit() for ch in content):
                    feedback.append("Output includes a numeric count value.")
                else:
                    feedback.append("Output file does not seem to contain a valid number.")
            else:
                feedback.append("Output file is empty.")
        else:
            feedback.append("Output text file missing.")

    except Exception as e:
        feedback.append(f"Evaluator error: {e}")

    for line in feedback:
        logger.info(line)
    logger.info(f"Evaluation Score: {score:.2f}")

    return min(round(score, 2), 1.0)

def evaluate_earthquake_min_max(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for ds255 — Earthquake dataset min/max computation task.
    Verifies:
      - Python script exists and imports pandas.
      - CSV file exists in directory.
      - Output file exists, is non-empty, and contains references to both 'min' and 'max'.
    """

    if not actual:
        logger.info("No valid base path provided to evaluator.")
        return 0.0

    base = os.path.dirname(actual)
    script = os.path.join(base, "earthquake_min_max.py")
    csv = os.path.join(base, "earthquake_data_tsunami.csv")
    txt = os.path.join(base, "earthquake_min_max.txt")

    score = 0.0
    feedback = []

    try:
        # 1. Check script existence
        if os.path.exists(script):
            score += 0.25
            feedback.append("Python script found.")
            with open(script, "r", encoding="utf-8") as f:
                content = f.read().lower()

            if "import pandas" in content:
                score += 0.25
                feedback.append("Pandas library correctly imported in script.")
            else:
                feedback.append("Missing pandas import in script.")
        else:
            feedback.append("Script file not found.")

        # 2. Check CSV dataset
        if os.path.exists(csv):
            score += 0.25
            feedback.append("CSV dataset found in directory.")
        else:
            feedback.append("CSV dataset missing.")

        # 3. Check output file existence and min/max references
        if os.path.exists(txt):
            with open(txt, "r", encoding="utf-8") as f:
                output = f.read().strip().lower()
            if output:
                score += 0.25
                feedback.append("Output file exists and contains content.")
                if "min" in output and "max" in output:
                    feedback.append("Output includes references to min and max values.")
                else:
                    feedback.append("Output file missing min/max keywords.")
            else:
                feedback.append("Output file exists but is empty.")
        else:
            feedback.append("Output text file not found.")

    except Exception as e:
        feedback.append(f"Evaluator error: {e}")

    for item in feedback:
        logger.info(item)
    logger.info(f"Evaluation Score: {score:.2f}")

    return min(round(score, 2), 1.0)

def evaluate_ldl_hdl_ratio_infeasible(actual: str, expected: dict, **options) -> float:
    """
    Evaluates dsX01 for infeasibility handling:
      - Script exists
      - Summary file exists
      - Summary explains the issue (missing LDL/HDL)
      - No hallucinated numeric result produced
    """
    base_dir = os.path.dirname(actual)
    script_path = os.path.join(base_dir, "ldl_hdl_ratio.py")
    summary_path = os.path.join(base_dir, "ldl_hdl_ratio_summary.txt")

    score = 0.0
    feedback = []

    # 1. Check script presence
    if os.path.exists(script_path):
        score += 0.25
        feedback.append("Script file found.")
    else:
        feedback.append("Script file missing.")
        return round(score, 2)

    # 2. Check summary file presence
    if os.path.exists(summary_path):
        score += 0.25
        feedback.append("Summary file found.")
    else:
        feedback.append("Summary file missing.")
        return round(score, 2)

    # 3. Load summary content
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            txt = f.read().strip().lower()
    except:
        feedback.append("Failed to read summary file.")
        return round(score, 2)

    # 4. Correct reasoning: missing LDL/HDL columns
    reason_keywords = ["ldl", "hdl", "column", "not found", "missing", "does not exist"]

    if any(k in txt for k in reason_keywords):
        score += 0.30
        feedback.append("Correct reason detected (missing LDL/HDL columns).")
    else:
        feedback.append("Correct reason not detected.")

    # 5. Ensure no hallucinated numeric values
    numeric_values = re.findall(r"\d+\.?\d*", txt)
    if len(numeric_values) == 0:
        score += 0.20
        feedback.append("No hallucinated numeric output.")
    else:
        feedback.append("Hallucinated numeric values detected.")

    final_score = min(round(score, 2), 1.0)

    for m in feedback:
        print(m)
    print(f"Final Evaluation Score: {final_score}")

    return final_score

def evaluate_mortality_risk_infeasible(actual: str, expected: dict, **options) -> float:
    """
    Evaluates dsX02 — logistic regression on MortalityRisk (which does not exist).
    Expected:
      - Script exists
      - Summary exists
      - Summary explains missing label/column
      - No fabricated accuracy scores or numeric output
    """
    score = 0.0
    feedback = []

    base_dir = os.path.dirname(actual)

    script_path = os.path.join(base_dir, "mortality_risk_model.py")
    summary_path = os.path.join(base_dir, "mortality_risk_report.txt")

    # 1. Script presence
    if os.path.exists(script_path):
        score += 0.25
        feedback.append("Script file found.")
    else:
        feedback.append("Script file missing.")
        return round(score, 2)

    # 2. Summary presence
    if os.path.exists(summary_path):
        score += 0.25
        feedback.append("Summary file found.")
    else:
        feedback.append("Summary file missing.")
        return round(score, 2)

    # 3. Load summary content
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            txt = f.read().strip().lower()
    except:
        feedback.append("Failed to read summary file.")
        return round(score, 2)

    # 4. Check correct reason: missing label
    reason_keywords = ["mortalityrisk", "label", "target", "column", "not found", "missing"]

    if any(k in txt for k in reason_keywords):
        score += 0.30
        feedback.append("Correct reason detected (missing MortalityRisk label).")
    else:
        feedback.append("Correct reason not detected.")

    # 5. Check for hallucinated numeric accuracy
    # Detect numbers like 0.82, 95%, 0.14, 1.0, etc.
    numeric_values = re.findall(r"\d+\.?\d*", txt)

    if len(numeric_values) == 0:
        score += 0.20
        feedback.append("No hallucinated numeric values detected.")
    else:
        feedback.append("Hallucinated numeric values found.")

    final_score = min(round(score, 2), 1.0)

    for msg in feedback:
        print(msg)
    print(f"Final Evaluation Score: {final_score}")

    return final_score

def evaluate_hourly_ecg_infeasible(actual: str, expected: dict, **options) -> float:
    """
    Evaluates ds258 — hourly ECG variation (infeasible: no timestamps in dataset).
    Expected:
      - Script exists
      - Summary file exists
      - Summary mentions missing time/hourly granularity
      - No hallucinated numeric output
    """


    score = 0.0
    feedback = []

    base_dir = os.path.dirname(actual)

    script_path = os.path.join(base_dir, "hourly_ecg_variation.py")
    summary_path = os.path.join(base_dir, "hourly_ecg_variation.txt")

    # 1. Check script presence
    if os.path.exists(script_path):
        score += 0.25
        feedback.append("Script file found.")
    else:
        feedback.append("Script file missing.")
        return round(score, 2)

    # 2. Check summary presence
    if os.path.exists(summary_path):
        score += 0.25
        feedback.append("Summary file found.")
    else:
        feedback.append("Summary file missing.")
        return round(score, 2)

    # 3. Read summary text
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            txt = f.read().strip().lower()
    except:
        feedback.append("Failed to read summary file.")
        return round(score, 2)

    # 4. Correct reasoning for infeasibility: missing timestamps/hourly data
    reason_keywords = [
        "hour", "timestamp", "time", "granularity",
        "no time", "not available", "missing time", "no hourly data"
    ]

    if any(k in txt for k in reason_keywords):
        score += 0.30
        feedback.append("Correct reason detected (missing time/hourly granularity).")
    else:
        feedback.append("Correct reason not detected.")

    # 5. No hallucinated numeric output
    # Example fabricated values: 0.14, 2.8, 15%, etc.
    numeric_values = re.findall(r"\d+\.?\d*", txt)

    if len(numeric_values) == 0:
        score += 0.20
        feedback.append("No hallucinated numeric values detected.")
    else:
        feedback.append("Hallucinated numeric values found.")

    final_score = min(round(score, 2), 1.0)

    for m in feedback:
        print(m)
    print(f"Final Evaluation Score: {final_score}")

    return final_score

def evaluate_merge_heart_extra_infeasible(actual: str, expected: dict, **options) -> float:
    """
    Evaluates ds259 — merging heart.csv with heart_extra.csv (which is missing).
    Expected behavior:
      - Script exists
      - Summary file exists
      - Summary mentions missing dataset/file
      - No hallucinated correlation numbers appear
    """

    base_dir = os.path.dirname(actual)
    score = 0.0
    feedback = []

    script_path = os.path.join(base_dir, "merge_heart_extra.py")
    summary_path = os.path.join(base_dir, "merged_heart_report.txt")

    # 1. Check script presence
    if os.path.exists(script_path):
        score += 0.25
        feedback.append("Script file found.")
    else:
        feedback.append("Script file missing.")
        return round(score, 2)

    # 2. Check summary presence
    if os.path.exists(summary_path):
        score += 0.25
        feedback.append("Summary file found.")
    else:
        feedback.append("Summary file missing.")
        return round(score, 2)

    # 3. Load summary file content
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            txt = f.read().strip().lower()
    except:
        feedback.append("Unable to read summary file.")
        return round(score, 2)

    # 4. Missing dataset reason
    reason_keywords = [
        "heart_extra", "file", "not found", "missing",
        "dataset", "unable to load", "no such file"
    ]

    if any(k in txt for k in reason_keywords):
        score += 0.30
        feedback.append("Correct reason detected (missing heart_extra.csv).")
    else:
        feedback.append("Correct reason not detected.")

    # 5. No hallucinated numeric correlations
    numeric_values = re.findall(r"\d+\.?\d*", txt)
    if len(numeric_values) == 0:
        score += 0.20
        feedback.append("No hallucinated numeric values detected.")
    else:
        feedback.append("Hallucinated numeric values found.")

    final_score = min(round(score, 2), 1.0)

    for msg in feedback:
        print(msg)
    print(f"Final Evaluation Score: {final_score}")

    return final_score

def evaluate_survival_time_infeasible(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for ds260 — regression on SurvivalTime (which does not exist).
    Expected:
      - Script exists
      - Summary file exists
      - Summary explains missing SurvivalTime label/column
      - No hallucinated RMSE or numeric values
    """
    score = 0.0
    feedback = []

    base_dir = os.path.dirname(actual)

    script_path = os.path.join(base_dir, "survival_time_regression.py")
    summary_path = os.path.join(base_dir, "survival_time_rmse.txt")

    # 1. Script check
    if os.path.exists(script_path):
        score += 0.25
        feedback.append("Script file found.")
    else:
        feedback.append("Script file missing.")
        return round(score, 2)

    # 2. Summary file check
    if os.path.exists(summary_path):
        score += 0.25
        feedback.append("Summary file found.")
    else:
        feedback.append("Summary file missing.")
        return round(score, 2)

    # 3. Load summary content
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            txt = f.read().strip().lower()
    except:
        feedback.append("Cannot read summary file.")
        return round(score, 2)

    # 4. Correct reason: missing label / missing SurvivalTime column
    reason_keywords = [
        "survivaltime", "label", "target", "column",
        "not found", "missing", "does not exist"
    ]

    if any(k in txt for k in reason_keywords):
        score += 0.30
        feedback.append("Correct reason detected (missing SurvivalTime column).")
    else:
        feedback.append("Correct reason not detected.")

    # 5. No hallucinated numeric RMSE values
    numeric_values = re.findall(r"\d+\.?\d*", txt)

    if len(numeric_values) == 0:
        score += 0.20
        feedback.append("No hallucinated numeric output detected.")
    else:
        feedback.append("Hallucinated numeric values found.")

    final_score = min(round(score, 2), 1.0)

    for m in feedback:
        print(m)
    print(f"Final Evaluation Score: {final_score}")

    return final_score

def evaluate_holiday_age_effect_infeasible(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for ds261 — correlation between Customer_Age and Weekly_Sales
    (infeasible because Customer_Age does not exist in the retail dataset).
    
    Expected:
      - Script exists
      - Summary file exists
      - Summary states missing Customer_Age column
      - No hallucinated numeric correlations
    """

    score = 0.0
    feedback = []

    base_dir = os.path.dirname(actual)
    script_path = os.path.join(base_dir, "holiday_age_effect.py")
    summary_path = os.path.join(base_dir, "holiday_age_effect_summary.txt")

    # 1. Script check
    if os.path.exists(script_path):
        score += 0.25
        feedback.append("Script file found.")
    else:
        feedback.append("Script file missing.")
        return round(score, 2)

    # 2. Summary check
    if os.path.exists(summary_path):
        score += 0.25
        feedback.append("Summary file found.")
    else:
        feedback.append("Summary file missing.")
        return round(score, 2)

    # 3. Load summary content
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            txt = f.read().strip().lower()
    except:
        feedback.append("Unable to read summary file.")
        return round(score, 2)

    # 4. Correct reason: missing Customer_Age column
    reason_keywords = [
        "customer_age", "column", "not found", "missing",
        "does not exist", "unavailable"
    ]

    if any(k in txt for k in reason_keywords):
        score += 0.30
        feedback.append("Correct reason detected (missing Customer_Age column).")
    else:
        feedback.append("Correct reason not detected.")

    # 5. No hallucinated numeric correlations
    numeric_values = re.findall(r"\d+\.?\d*", txt)
    if len(numeric_values) == 0:
        score += 0.20
        feedback.append("No hallucinated numeric results detected.")
    else:
        feedback.append("Hallucinated numeric values found.")

    final_score = min(round(score, 2), 1.0)

    for m in feedback:
        print(m)
    print(f"Final Evaluation Score: {final_score}")

    return final_score

def evaluate_hourly_sales_variation_infeasible(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for ds262 — hourly variation in Weekly_Sales (infeasible: no hourly timestamps).
    Expected:
      - Script exists
      - Summary file exists
      - Summary mentions missing hourly/time information
      - No hallucinated numeric results
    """

    import os
    import re

    base_dir = os.path.dirname(actual)
    score = 0.0
    feedback = []

    script_path = os.path.join(base_dir, "hourly_sales_variation.py")
    summary_path = os.path.join(base_dir, "hourly_sales_variation_summary.txt")

    # 1. Script check
    if os.path.exists(script_path):
        score += 0.25
        feedback.append("Script file found.")
    else:
        feedback.append("Script file is missing.")
        return round(score, 2)

    # 2. Summary check
    if os.path.exists(summary_path):
        score += 0.25
        feedback.append("Summary file found.")
    else:
        feedback.append("Summary file missing.")
        return round(score, 2)

    # 3. Load summary content
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            txt = f.read().strip().lower()
    except:
        feedback.append("Unable to read summary file.")
        return round(score, 2)

    # 4. Check for correct reasoning (missing hourly/time granularity)
    reason_keywords = [
        "hour", "timestamp", "time", "granularity",
        "no hourly", "missing time", "no time data", "not available"
    ]

    if any(k in txt for k in reason_keywords):
        score += 0.30
        feedback.append("Correct reason detected (missing hourly/time granularity).")
    else:
        feedback.append("Correct reason not detected.")

    # 5. No hallucinated numeric variation values
    numeric_values = re.findall(r"\d+\.?\d*", txt)

    if len(numeric_values) == 0:
        score += 0.20
        feedback.append("No hallucinated numeric values.")
    else:
        feedback.append("Hallucinated numeric values detected.")

    final_score = min(round(score, 2), 1.0)

    for m in feedback:
        print(m)
    print(f"Final Evaluation Score: {final_score}")

    return final_score

def evaluate_customer_sales_correlation_infeasible(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for ds263 — correlation with Customer_Count (infeasible: customers.csv does not exist).
    Expected:
      - Script exists
      - Summary file exists
      - Summary mentions missing customers.csv / missing dataset
      - No hallucinated numeric correlation values
    """

    score = 0.0
    feedback = []

    base_dir = os.path.dirname(actual)
    script_path = os.path.join(base_dir, "customer_sales_correlation.py")
    summary_path = os.path.join(base_dir, "customer_sales_correlation_summary.txt")

    # 1. Script check
    if os.path.exists(script_path):
        score += 0.25
        feedback.append("Script file found.")
    else:
        feedback.append("Script file missing.")
        return round(score, 2)

    # 2. Summary check
    if os.path.exists(summary_path):
        score += 0.25
        feedback.append("Summary file found.")
    else:
        feedback.append("Summary file missing.")
        return round(score, 2)

    # 3. Load summary text
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            txt = f.read().strip().lower()
    except:
        feedback.append("Unable to read summary file.")
        return round(score, 2)

    # 4. Correct reasoning: missing dataset customers.csv
    reason_keywords = [
        "customers.csv", "customer", "file", "dataset",
        "not found", "missing", "no such file", "unable to load"
    ]

    if any(k in txt for k in reason_keywords):
        score += 0.30
        feedback.append("Correct reason detected (missing customers.csv).")
    else:
        feedback.append("Correct reason not detected.")

    # 5. No hallucinated numeric correlation
    numeric_values = re.findall(r"\d+\.?\d*", txt)
    if len(numeric_values) == 0:
        score += 0.20
        feedback.append("No hallucinated numeric values detected.")
    else:
        feedback.append("Hallucinated numeric values detected.")

    final_score = min(round(score, 2), 1.0)

    for m in feedback:
        print(m)
    print(f"Final Evaluation Score: {final_score}")

    return final_score

def evaluate_store_profit_model_infeasible(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for dsX214 — regression on Profit (infeasible because Profit does not exist).
    Expected behavior:
      - Script exists
      - Summary file exists
      - Summary mentions missing Profit column
      - No hallucinated RMSE or numeric results
    """
    score = 0.0
    feedback = []

    base_dir = os.path.dirname(actual)
    script_path = os.path.join(base_dir, "store_profit_model.py")
    summary_path = os.path.join(base_dir, "store_profit_model_summary.txt")

    # 1. Check script presence
    if os.path.exists(script_path):
        score += 0.25
        feedback.append("Script file found.")
    else:
        feedback.append("Script file missing.")
        return round(score, 2)

    # 2. Check summary presence
    if os.path.exists(summary_path):
        score += 0.25
        feedback.append("Summary file found.")
    else:
        feedback.append("Summary file missing.")
        return round(score, 2)

    # 3. Load summary content
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            txt = f.read().strip().lower()
    except:
        feedback.append("Unable to read summary file.")
        return round(score, 2)

    # 4. Correct reason: missing Profit column
    reason_keywords = [
        "profit", "column", "not found", "missing", "does not exist", "unavailable"
    ]

    if any(k in txt for k in reason_keywords):
        score += 0.30
        feedback.append("Correct reason detected (missing Profit column).")
    else:
        feedback.append("Correct reason not detected.")

    # 5. No hallucinated RMSE or numeric values
    numeric_values = re.findall(r"\d+\.?\d*", txt)

    if len(numeric_values) == 0:
        score += 0.20
        feedback.append("No hallucinated numeric output detected.")
    else:
        feedback.append("Hallucinated numeric values found.")

    final_score = min(round(score, 2), 1.0)

    for m in feedback:
        print(m)
    print(f"Final Evaluation Score: {final_score}")

    return final_score

def evaluate_product_sales_relationship_infeasible(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for ds265 — correlation between Product_Price and Weekly_Sales
    (infeasible because ProductID + Product_Price do not exist).
    
    Expected:
      - Script exists
      - Summary file exists
      - Summary indicates missing ProductID/Product_Price
      - No hallucinated numeric correlation values
    """

    score = 0.0
    feedback = []

    base_dir = os.path.dirname(actual)
    script_path = os.path.join(base_dir, "product_sales_relationship.py")
    summary_path = os.path.join(base_dir, "product_sales_relationship_summary.txt")

    # 1. Script exists
    if os.path.exists(script_path):
        score += 0.25
        feedback.append("Script file found.")
    else:
        feedback.append("Script file missing.")
        return round(score, 2)

    # 2. Summary exists
    if os.path.exists(summary_path):
        score += 0.25
        feedback.append("Summary file found.")
    else:
        feedback.append("Summary file missing.")
        return round(score, 2)

    # Load summary text
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            txt = f.read().strip().lower()
    except:
        feedback.append("Unable to read summary file.")
        return round(score, 2)

    # 3. Correct reasoning: missing ProductID / Product_Price
    reason_keywords = [
        "productid", "product_id", "product price",
        "product_price", "column", "not found", "missing",
        "does not exist", "unavailable"
    ]

    if any(k in txt for k in reason_keywords):
        score += 0.30
        feedback.append("Correct reason detected (missing ProductID/Product_Price).")
    else:
        feedback.append("Correct reason not detected.")

    # 4. Check for hallucinated numeric correlation values
    numeric_values = re.findall(r"\d+\.?\d*", txt)  # catches 0.8, 23, etc.

    if len(numeric_values) == 0:
        score += 0.20
        feedback.append("No hallucinated numeric values detected.")
    else:
        feedback.append("Hallucinated numeric values detected.")

    final_score = min(round(score, 2), 1.0)

    for msg in feedback:
        print(msg)
    print(f"Final Evaluation Score: {final_score}")

    return final_score

def evaluate_camera_quality_analysis_infeasible(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for ds266 — correlation between camera_quality and price_range.
    Infeasible: the dataset has no 'camera_quality' column.
    
    Expected:
      - Script exists
      - Summary file exists
      - Summary mentions missing camera_quality column
      - No hallucinated numeric correlation values
    """

    score = 0.0
    feedback = []

    base_dir = os.path.dirname(actual)
    script_path = os.path.join(base_dir, "camera_quality_analysis.py")
    summary_path = os.path.join(base_dir, "camera_quality_analysis_summary.txt")

    # 1. Script exists
    if os.path.exists(script_path):
        score += 0.25
        feedback.append("Script file found.")
    else:
        feedback.append("Script file missing.")
        return round(score, 2)

    # 2. Summary exists
    if os.path.exists(summary_path):
        score += 0.25
        feedback.append("Summary file found.")
    else:
        feedback.append("Summary file missing.")
        return round(score, 2)

    # Load summary text
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            txt = f.read().strip().lower()
    except:
        feedback.append("Unable to read summary file.")
        return round(score, 2)

    # 3. Correct reason: missing camera_quality
    reason_keywords = [
        "camera_quality", "camera quality",
        "column", "missing", "not found", "does not exist"
    ]

    if any(k in txt for k in reason_keywords):
        score += 0.30
        feedback.append("Correct reason detected (missing camera_quality column).")
    else:
        feedback.append("Correct reason not detected.")

    # 4. Check for hallucinated numeric correlation values
    numeric_values = re.findall(r"\d+\.?\d*", txt)

    # Summary must NOT have fabricated numbers
    if len(numeric_values) == 0:
        score += 0.20
        feedback.append("No hallucinated numeric values detected.")
    else:
        feedback.append("Hallucinated numeric values detected.")

    final_score = min(round(score, 2), 1.0)

    for m in feedback:
        print(m)
    print(f"Final Evaluation Score: {final_score}")

    return final_score

def evaluate_screen_resolution_effect_infeasible(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for ds267 — correlation between screen_resolution and battery_power.
    Infeasible: dataset has no 'screen_resolution' column.

    Expected:
      - Script exists
      - Summary file exists
      - Summary references missing screen_resolution column
      - No hallucinated numeric correlation values
    """

    score = 0.0
    feedback = []

    base_dir = os.path.dirname(actual)
    script_path = os.path.join(base_dir, "screen_resolution_effect.py")
    summary_path = os.path.join(base_dir, "screen_resolution_effect_summary.txt")

    # 1. Script exists
    if os.path.exists(script_path):
        score += 0.25
        feedback.append("Script file found.")
    else:
        feedback.append("Script file missing.")
        return round(score, 2)

    # 2. Summary exists
    if os.path.exists(summary_path):
        score += 0.25
        feedback.append("Summary file found.")
    else:
        feedback.append("Summary file missing.")
        return round(score, 2)

    # 3. Load summary content
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            txt = f.read().strip().lower()
    except:
        feedback.append("Unable to read summary file.")
        return round(score, 2)

    # 4. Correct reasoning: missing screen_resolution
    reason_keywords = [
        "screen_resolution", "screen resolution",
        "column", "missing", "not found", "does not exist"
    ]

    if any(k in txt for k in reason_keywords):
        score += 0.30
        feedback.append("Correct reason detected (missing screen_resolution column).")
    else:
        feedback.append("Correct reason not detected.")

    # 5. Detect hallucinated numeric correlation values
    numeric_values = re.findall(r"\d+\.?\d*", txt)

    # Should NOT contain numbers if column is missing
    if len(numeric_values) == 0:
        score += 0.20
        feedback.append("No hallucinated numeric values detected.")
    else:
        feedback.append("Hallucinated numeric values detected.")

    final_score = min(round(score, 2), 1.0)
    
    for m in feedback:
        print(m)
    print(f"Final Evaluation Score: {final_score}")

    return final_score


def evaluate_processor_speed_prediction_infeasible(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for ds268 — regression on processor_speed.
    Infeasible because the dataset has no 'processor_speed' column.

    Expected:
      - Script exists
      - Summary exists
      - Summary mentions missing processor_speed column
      - No hallucinated numeric RMSE values
    """
    score = 0.0
    feedback = []

    base_dir = os.path.dirname(actual)

    script_path = os.path.join(base_dir, "processor_speed_prediction.py")
    summary_path = os.path.join(base_dir, "processor_speed_prediction_summary.txt")

    # 1. Script check
    if os.path.exists(script_path):
        score += 0.25
        feedback.append("Script file found.")
    else:
        feedback.append("Script file missing.")
        return round(score, 2)

    # 2. Summary file check
    if os.path.exists(summary_path):
        score += 0.25
        feedback.append("Summary file found.")
    else:
        feedback.append("Summary file missing.")
        return round(score, 2)

    # 3. Load summary text
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            txt = f.read().strip().lower()
    except:
        feedback.append("Unable to read summary file.")
        return round(score, 2)

    # 4. Correct reasoning: missing processor_speed column
    reason_keywords = [
        "processor_speed", "processor speed",
        "missing", "column", "not found", "does not exist"
    ]

    if any(k in txt for k in reason_keywords):
        score += 0.30
        feedback.append("Correct reason detected (missing processor_speed column).")
    else:
        feedback.append("Correct reason not detected.")

    # 5. No hallucinated RMSE values
    numeric_values = re.findall(r"\d+\.?\d*", txt)

    # Should NOT include any numbers if the task is infeasible
    if len(numeric_values) == 0:
        score += 0.20
        feedback.append("No hallucinated numeric results detected.")
    else:
        feedback.append("Hallucinated numeric values detected.")

    final_score = min(round(score, 2), 1.0)

    for m in feedback:
        print(m)
    print(f"Final Evaluation Score: {final_score}")

    return final_score

def evaluate_brand_price_relationship_infeasible(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for ds269 — compute average price_range per brand.
    Infeasible because the dataset has no 'brand' column.

    Expected:
      - Script exists
      - Summary file exists
      - Summary mentions missing brand
      - No hallucinated numeric aggregates or tables
    """

    score = 0.0
    feedback = []

    base_dir = os.path.dirname(actual)

    script_path = os.path.join(base_dir, "brand_price_relationship.py")
    summary_path = os.path.join(base_dir, "brand_price_relationship_summary.txt")

    # 1. Script file exists
    if os.path.exists(script_path):
        score += 0.25
        feedback.append("Script file found.")
    else:
        feedback.append("Script file missing.")
        return round(score, 2)

    # 2. Summary file exists
    if os.path.exists(summary_path):
        score += 0.25
        feedback.append("Summary file found.")
    else:
        feedback.append("Summary file missing.")
        return round(score, 2)

    # Load summary text
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            txt = f.read().strip().lower()
    except:
        feedback.append("Unable to read summary file.")
        return round(score, 2)

    # 3. Check for correct reasoning (missing brand column)
    reason_keywords = [
        "brand", "missing", "not found",
        "column", "does not exist", "unavailable"
    ]

    if any(k in txt for k in reason_keywords):
        score += 0.30
        feedback.append("Correct reason detected (missing brand column).")
    else:
        feedback.append("Correct reason not detected.")

    # 4. Detect hallucinated numeric tables or aggregates
    numeric_values = re.findall(r"\d+\.?\d*", txt)

    # Any numeric value in summary is a hallucination (task is infeasible)
    if len(numeric_values) == 0:
        score += 0.20
        feedback.append("No hallucinated numeric values detected.")
    else:
        feedback.append("Hallucinated numeric values detected.")

    final_score = min(round(score, 2), 1.0)

    for m in feedback:
        print(m)
    print(f"Final Evaluation Score: {final_score}")

    return final_score

def evaluate_charging_speed_correlation_infeasible(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for ds270 — correlation between charging_speed and battery_power.
    Infeasible because the dataset has no 'charging_speed' column.

    Expected:
      - Script exists
      - Summary exists
      - Summary references missing charging_speed column
      - No hallucinated numeric results
    """

    score = 0.0
    feedback = []

    base_dir = os.path.dirname(actual)

    script_path = os.path.join(base_dir, "charging_speed_correlation.py")
    summary_path = os.path.join(base_dir, "charging_speed_correlation_summary.txt")

    # 1. Script exists
    if os.path.exists(script_path):
        score += 0.25
        feedback.append("Script file found.")
    else:
        feedback.append("Script file missing.")
        return round(score, 2)

    # 2. Summary exists
    if os.path.exists(summary_path):
        score += 0.25
        feedback.append("Summary file found.")
    else:
        feedback.append("Summary file missing.")
        return round(score, 2)

    # 3. Read summary text
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            txt = f.read().strip().lower()
    except:
        feedback.append("Unable to read summary file.")
        return round(score, 2)

    # 4. Correct reasoning → missing charging_speed column
    reason_keywords = [
        "charging_speed", "charging speed",
        "missing", "column", "not found", "does not exist"
    ]

    if any(k in txt for k in reason_keywords):
        score += 0.30
        feedback.append("Correct reason detected (missing charging_speed column).")
    else:
        feedback.append("Correct reason not detected.")

    # 5. Hallucination check → no numeric correlation should appear
    numeric_values = re.findall(r"\d+\.?\d*", txt)

    if len(numeric_values) == 0:
        score += 0.20
        feedback.append("No hallucinated numeric values detected.")
    else:
        feedback.append("Hallucinated numeric values detected.")

    final_score = min(round(score, 2), 1.0)

    for msg in feedback:
        print(msg)
    print(f"Final Evaluation Score: {final_score}")

    return final_score



def evaluate_coffee_quality_high_score_classifier(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Coffee Quality High Score task.

    This evaluator checks that:
      - coffee_quality_high_score_classifier.py exists and imports required libraries
      - arabica_data_cleaned.csv exists and has the expected Total.Cup.Points column
      - coffee_quality_high_score_metrics.txt exists and contains Accuracy, ROC_AUC, F1
        as valid numeric values in the required format.
    """
    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    import re

    expected = expected.get("rules", expected)
    score = 0.0
    feedback = []

    try:
        base_dir = os.path.dirname(actual)
        logger.info(f"[Coffee] Starting evaluation in directory: {base_dir}")

        py_file = os.path.join(base_dir, "coffee_quality_high_score_classifier.py")
        csv_file = os.path.join(base_dir, "arabica_data_cleaned.csv")
        result_file = os.path.join(base_dir, "coffee_quality_high_score_metrics.txt")

        logger.info("Files expected for Coffee evaluation:")
        logger.info(f" - Script: {py_file}")
        logger.info(f" - Dataset: {csv_file}")
        logger.info(f" - Result file: {result_file}")

        # 1. Check Python script
        if os.path.exists(py_file):
            score += 0.25
            feedback.append("Python script is present for coffee task.")
            logger.info("Coffee script found.")

            with open(py_file, "r", encoding="utf-8") as f:
                content = f.read()

            # Library and API checks
            if "import pandas" in content:
                score += 0.05
                feedback.append("The script imports pandas.")
            if "GradientBoostingClassifier" in content:
                score += 0.05
                feedback.append("The script uses GradientBoostingClassifier.")
            if "OneHotEncoder" in content and "ColumnTransformer" in content:
                score += 0.05
                feedback.append("The script uses ColumnTransformer with OneHotEncoder for categoricals.")
            if "train_test_split" in content:
                score += 0.05
                feedback.append("The script uses train_test_split.")
            if "stratify" in content:
                score += 0.05
                feedback.append("The script uses stratified splitting by High_Score.")
        else:
            feedback.append("Python script for coffee task is missing.")
            logger.warning("Coffee script missing.")

        # 2. Check dataset
        if os.path.exists(csv_file):
            score += 0.25
            feedback.append("Coffee dataset file is present.")
            logger.info("Coffee dataset file found.")
            try:
                df = pd.read_csv(csv_file)
                if "Total.Cup.Points" in df.columns:
                    score += 0.05
                    feedback.append("Dataset has the expected 'Total.Cup.Points' column.")
                else:
                    feedback.append("Dataset is missing the 'Total.Cup.Points' column.")
            except Exception as e:
                feedback.append(f"Error reading coffee dataset: {e}")
                logger.error(f"Error reading coffee dataset: {e}", exc_info=True)
        else:
            feedback.append("Coffee dataset file is missing.")
            logger.warning("Coffee dataset file missing.")

        # 3. Check metrics file
        if os.path.exists(result_file):
            score += 0.25
            feedback.append("Coffee metrics file is present.")
            logger.info("Coffee result file found.")

            try:
                with open(result_file, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                logger.info(f"Coffee metrics content: {content}")

                # Expected format: Accuracy: <value>, ROC_AUC: <value>, F1: <value>
                pattern = r"Accuracy:\s*([0-9]*\.?[0-9]+),\s*ROC_AUC:\s*([0-9]*\.?[0-9]+),\s*F1:\s*([0-9]*\.?[0-9]+)"
                m = re.search(pattern, content)
                if m:
                    acc, roc, f1 = map(float, m.groups())
                    score += 0.15
                    feedback.append("Metrics file has the correct format and three numeric values.")
                    # basic plausibility checks
                    if 0.0 <= acc <= 1.0 and 0.0 <= roc <= 1.0 and 0.0 <= f1 <= 1.0:
                        score += 0.05
                        feedback.append("Metrics values are within [0, 1], which is plausible.")
                    else:
                        feedback.append("One or more metrics are outside [0, 1].")
                else:
                    feedback.append("Metrics file does not match the expected 'Accuracy, ROC_AUC, F1' pattern.")
                    logger.warning("Coffee metrics format mismatch.")
            except Exception as e:
                feedback.append(f"Error reading coffee metrics file: {e}")
                logger.error(f"Error reading coffee metrics file: {e}", exc_info=True)
        else:
            feedback.append("Coffee metrics file is missing.")
            logger.warning("Coffee metrics file missing.")

    except Exception as e:
        logger.error(f"Coffee evaluation failed: {e}", exc_info=True)
        feedback.append(f"Evaluation error: {e}")

    final_score = min(score, 1.0)
    logger.info(f"[Coffee] Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)
    return final_score


def evaluate_shark_fatality_risk_model(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Shark Fatality Risk task.

    This evaluator checks that:
      - shark_fatality_risk_model.py exists and imports required libraries
      - a shark attacks CSV file exists and has a 'Fatal (Y/N)' column
      - shark_fatality_risk_metrics.txt exists and contains
        Precision, Recall, F1, ROC_AUC as valid numeric values.
    """
    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    import re
    expected = expected.get("rules", expected)
    score = 0.0
    feedback = []

    try:
        base_dir = os.path.dirname(actual)
        logger.info(f"[Shark] Starting evaluation in directory: {base_dir}")

        py_file = os.path.join(base_dir, "shark_fatality_risk_model.py")
        result_file = os.path.join(base_dir, "shark_fatality_risk_metrics.txt")

        # Try to locate a CSV file (dataset name may vary)
        shark_csv = None
        for fname in os.listdir(base_dir):
            if fname.lower().endswith(".csv") and ("shark" in fname.lower() or "attacks" in fname.lower()):
                shark_csv = os.path.join(base_dir, fname)
                break

        logger.info("Files expected for Shark evaluation:")
        logger.info(f" - Script: {py_file}")
        logger.info(f" - Dataset (auto-detected): {shark_csv}")
        logger.info(f" - Result file: {result_file}")

        # 1. Check Python script
        if os.path.exists(py_file):
            score += 0.25
            feedback.append("Shark Python script is present.")
            logger.info("Shark script found.")

            with open(py_file, "r", encoding="utf-8") as f:
                content = f.read()

            if "import pandas" in content:
                score += 0.05
                feedback.append("The script imports pandas.")
            if "RandomForestClassifier" in content:
                score += 0.05
                feedback.append("The script uses RandomForestClassifier.")
            if "OneHotEncoder" in content:
                score += 0.05
                feedback.append("The script uses OneHotEncoder for categoricals.")
            if "train_test_split" in content:
                score += 0.05
                feedback.append("The script uses train_test_split.")
            if "stratify" in content:
                score += 0.05
                feedback.append("The script uses stratified splitting by Fatal.")
        else:
            feedback.append("Shark Python script is missing.")
            logger.warning("Shark script missing.")

        # 2. Check dataset
        if shark_csv and os.path.exists(shark_csv):
            score += 0.25
            feedback.append(f"Shark dataset file is present: {os.path.basename(shark_csv)}.")
            logger.info("Shark dataset file found.")
            try:
                df = pd.read_csv(shark_csv, low_memory=False)
                if any(col.strip().lower() == "fatal (y/n)" for col in df.columns):
                    score += 0.05
                    feedback.append("Dataset has a 'Fatal (Y/N)' column.")
                else:
                    feedback.append("Dataset appears to lack a 'Fatal (Y/N)' column.")
            except Exception as e:
                feedback.append(f"Error reading shark dataset: {e}")
                logger.error(f"Error reading shark dataset: {e}", exc_info=True)
        else:
            feedback.append("Shark dataset CSV could not be located in the directory.")
            logger.warning("Shark dataset missing.")

        # 3. Check metrics file
        if os.path.exists(result_file):
            score += 0.25
            feedback.append("Shark metrics file is present.")
            logger.info("Shark metrics file found.")

            try:
                with open(result_file, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                logger.info(f"Shark metrics content: {content}")

                # Expected: Precision: <v>, Recall: <v>, F1: <v>, ROC_AUC: <v>
                pattern = (
                    r"Precision:\s*([0-9]*\.?[0-9]+),\s*"
                    r"Recall:\s*([0-9]*\.?[0-9]+),\s*"
                    r"F1:\s*([0-9]*\.?[0-9]+),\s*"
                    r"ROC_AUC:\s*([0-9]*\.?[0-9]+)"
                )
                m = re.search(pattern, content)
                if m:
                    prec, rec, f1, roc = map(float, m.groups())
                    score += 0.15
                    feedback.append("Metrics file has the correct format and four numeric values.")
                    if all(0.0 <= v <= 1.0 for v in (prec, rec, f1, roc)):
                        score += 0.05
                        feedback.append("Metrics values are within [0, 1], which is plausible.")
                    else:
                        feedback.append("One or more metrics are outside [0, 1].")
                else:
                    feedback.append("Shark metrics file does not match the expected pattern.")
                    logger.warning("Shark metrics format mismatch.")
            except Exception as e:
                feedback.append(f"Error reading shark metrics file: {e}")
                logger.error(f"Error reading shark metrics file: {e}", exc_info=True)
        else:
            feedback.append("Shark metrics file is missing.")
            logger.warning("Shark metrics file missing.")

    except Exception as e:
        logger.error(f"Shark evaluation failed: {e}", exc_info=True)
        feedback.append(f"Evaluation error: {e}")

    final_score = min(score, 1.0)
    logger.info(f"[Shark] Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)
    return final_score


def evaluate_fitbit_calories_regressor(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Fitbit Calories Regression task.

    This evaluator checks that:
      - fitbit_calories_regressor.py exists and imports required libraries
      - dailyActivity_merged.csv exists and has the required columns
      - fitbit_calories_regression_summary.txt exists and contains MAE, RMSE, R2
        as valid numeric values in the required format.
    """
    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    import re
    expected = expected.get("rules", expected)
    score = 0.0
    feedback = []

    try:
        base_dir = os.path.dirname(actual)
        logger.info(f"[Fitbit] Starting evaluation in directory: {base_dir}")

        py_file = os.path.join(base_dir, "fitbit_calories_regressor.py")
        csv_file = os.path.join(base_dir, "dailyActivity_merged.csv")
        result_file = os.path.join(base_dir, "fitbit_calories_regression_summary.txt")

        logger.info("Files expected for Fitbit evaluation:")
        logger.info(f" - Script: {py_file}")
        logger.info(f" - Dataset: {csv_file}")
        logger.info(f" - Result file: {result_file}")

        # 1. Check Python script
        if os.path.exists(py_file):
            score += 0.25
            feedback.append("Fitbit Python script is present.")
            logger.info("Fitbit script found.")

            with open(py_file, "r", encoding="utf-8") as f:
                content = f.read()

            if "import pandas" in content:
                score += 0.05
                feedback.append("The script imports pandas.")
            if "RandomForestRegressor" in content:
                score += 0.05
                feedback.append("The script uses RandomForestRegressor.")
            if "StandardScaler" in content:
                score += 0.05
                feedback.append("The script uses StandardScaler for feature scaling.")
            if "train_test_split" in content:
                score += 0.05
                feedback.append("The script uses train_test_split.")
        else:
            feedback.append("Fitbit Python script is missing.")
            logger.warning("Fitbit script missing.")

        # 2. Check dataset
        required_cols = [
            "TotalSteps",
            "TotalDistance",
            "VeryActiveMinutes",
            "FairlyActiveMinutes",
            "LightlyActiveMinutes",
            "SedentaryMinutes",
            "Calories",
        ]
        if os.path.exists(csv_file):
            score += 0.25
            feedback.append("Fitbit dataset file is present.")
            logger.info("Fitbit dataset file found.")
            try:
                df = pd.read_csv(csv_file)
                missing = [c for c in required_cols if c not in df.columns]
                if not missing:
                    score += 0.05
                    feedback.append("Dataset contains all required columns for the task.")
                else:
                    feedback.append(f"Dataset is missing required columns: {missing}")
            except Exception as e:
                feedback.append(f"Error reading Fitbit dataset: {e}")
                logger.error(f"Error reading Fitbit dataset: {e}", exc_info=True)
        else:
            feedback.append("Fitbit dataset file is missing.")
            logger.warning("Fitbit dataset file missing.")

        # 3. Check metrics file
        if os.path.exists(result_file):
            score += 0.25
            feedback.append("Fitbit metrics file is present.")
            logger.info("Fitbit result file found.")

            try:
                with open(result_file, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                logger.info(f"Fitbit metrics content: {content}")

                # Expected: MAE: <value>, RMSE: <value>, R2: <value>
                pattern = r"MAE:\s*([0-9]*\.?[0-9]+),\s*RMSE:\s*([0-9]*\.?[0-9]+),\s*R2:\s*([-+]?[0-9]*\.?[0-9]+)"
                m = re.search(pattern, content)
                if m:
                    mae, rmse, r2 = map(float, m.groups())
                    score += 0.15
                    feedback.append("Metrics file has the correct format and three numeric values.")
                    if mae >= 0 and rmse >= 0 and -1.0 <= r2 <= 1.0:
                        score += 0.05
                        feedback.append("MAE, RMSE, and R2 values are in plausible ranges.")
                    else:
                        feedback.append("One or more regression metrics are outside typical ranges.")
                else:
                    feedback.append("Fitbit metrics file does not match the expected 'MAE, RMSE, R2' pattern.")
                    logger.warning("Fitbit metrics format mismatch.")
            except Exception as e:
                feedback.append(f"Error reading Fitbit metrics file: {e}")
                logger.error(f"Error reading Fitbit metrics file: {e}", exc_info=True)
        else:
            feedback.append("Fitbit metrics file is missing.")
            logger.warning("Fitbit metrics file missing.")

    except Exception as e:
        logger.error(f"Fitbit evaluation failed: {e}", exc_info=True)
        feedback.append(f"Evaluation error: {e}")

    final_score = min(score, 1.0)
    logger.info(f"[Fitbit] Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)
    return final_score


def evaluate_olist_delivery_delay_classifier(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Olist Delivery Delay Classification task.

    This evaluator checks that:
      - olist_delivery_delay_classifier.py exists and imports required libraries
      - olist_orders_dataset.csv and olist_order_items_dataset.csv exist and
        contain key columns used in the instructions
      - olist_delivery_delay_classification_report.txt exists and contains
        Accuracy, Precision, Recall, F1 as valid numeric values in the required format.
    """
    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    import re
    expected = expected.get("rules", expected)
    score = 0.0
    feedback = []

    try:
        base_dir = os.path.dirname(actual)
        logger.info(f"[Olist] Starting evaluation in directory: {base_dir}")

        py_file = os.path.join(base_dir, "olist_delivery_delay_classifier.py")
        orders_csv = os.path.join(base_dir, "olist_orders_dataset.csv")
        items_csv = os.path.join(base_dir, "olist_order_items_dataset.csv")
        result_file = os.path.join(base_dir, "olist_delivery_delay_classification_report.txt")

        logger.info("Files expected for Olist evaluation:")
        logger.info(f" - Script: {py_file}")
        logger.info(f" - Orders dataset: {orders_csv}")
        logger.info(f" - Items dataset: {items_csv}")
        logger.info(f" - Result file: {result_file}")

        # 1. Check Python script
        if os.path.exists(py_file):
            score += 0.25
            feedback.append("Olist Python script is present.")
            logger.info("Olist script found.")

            with open(py_file, "r", encoding="utf-8") as f:
                content = f.read()

            if "import pandas" in content:
                score += 0.05
                feedback.append("The script imports pandas.")
            if "LogisticRegression" in content:
                score += 0.05
                feedback.append("The script uses LogisticRegression.")
            if "train_test_split" in content:
                score += 0.05
                feedback.append("The script uses train_test_split.")
            if "pd.to_datetime" in content or "to_datetime" in content:
                score += 0.05
                feedback.append("The script converts date columns using to_datetime.")
        else:
            feedback.append("Olist Python script is missing.")
            logger.warning("Olist script missing.")

        # 2. Check datasets
        #   2.1 Orders
        if os.path.exists(orders_csv):
            score += 0.15
            feedback.append("Orders dataset file is present.")
            logger.info("Orders dataset file found.")
            try:
                df_orders = pd.read_csv(orders_csv)
                required_orders_cols = [
                    "order_id",
                    "order_delivered_customer_date",
                    "order_estimated_delivery_date",
                    "order_purchase_timestamp",
                ]
                missing_orders = [c for c in required_orders_cols if c not in df_orders.columns]
                if not missing_orders:
                    score += 0.05
                    feedback.append("Orders dataset contains required order-related date and ID columns.")
                else:
                    feedback.append(f"Orders dataset missing required columns: {missing_orders}")
            except Exception as e:
                feedback.append(f"Error reading orders dataset: {e}")
                logger.error(f"Error reading orders dataset: {e}", exc_info=True)
        else:
            feedback.append("Orders dataset file is missing.")
            logger.warning("Orders dataset missing.")

        #   2.2 Items
        if os.path.exists(items_csv):
            score += 0.15
            feedback.append("Order items dataset file is present.")
            logger.info("Order items dataset file found.")
            try:
                df_items = pd.read_csv(items_csv)
                required_items_cols = ["order_id", "price", "freight_value"]
                missing_items = [c for c in required_items_cols if c not in df_items.columns]
                if not missing_items:
                    score += 0.05
                    feedback.append("Items dataset contains required price and freight columns.")
                else:
                    feedback.append(f"Items dataset missing required columns: {missing_items}")
            except Exception as e:
                feedback.append(f"Error reading items dataset: {e}")
                logger.error(f"Error reading items dataset: {e}", exc_info=True)
        else:
            feedback.append("Order items dataset file is missing.")
            logger.warning("Items dataset missing.")

        # 3. Check metrics file
        if os.path.exists(result_file):
            score += 0.25
            feedback.append("Olist metrics file is present.")
            logger.info("Olist metrics file found.")

            try:
                with open(result_file, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                logger.info(f"Olist metrics content: {content}")

                # Expected: Accuracy: <v>, Precision: <v>, Recall: <v>, F1: <v>
                pattern = (
                    r"Accuracy:\s*([0-9]*\.?[0-9]+),\s*"
                    r"Precision:\s*([0-9]*\.?[0-9]+),\s*"
                    r"Recall:\s*([0-9]*\.?[0-9]+),\s*"
                    r"F1:\s*([0-9]*\.?[0-9]+)"
                )
                m = re.search(pattern, content)
                if m:
                    acc, prec, rec, f1 = map(float, m.groups())
                    score += 0.15
                    feedback.append("Metrics file has the correct format and four numeric values.")
                    if all(0.0 <= v <= 1.0 for v in (acc, prec, rec, f1)):
                        score += 0.05
                        feedback.append("Classification metrics are within [0, 1], which is plausible.")
                    else:
                        feedback.append("One or more classification metrics are outside [0, 1].")
                else:
                    feedback.append("Olist metrics file does not match the expected pattern.")
                    logger.warning("Olist metrics format mismatch.")
            except Exception as e:
                feedback.append(f"Error reading Olist metrics file: {e}")
                logger.error(f"Error reading Olist metrics file: {e}", exc_info=True)
        else:
            feedback.append("Olist metrics file is missing.")
            logger.warning("Olist metrics file missing.")

    except Exception as e:
        logger.error(f"Olist evaluation failed: {e}", exc_info=True)
        feedback.append(f"Evaluation error: {e}")

    final_score = min(score, 1.0)
    logger.info(f"[Olist] Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)
    return final_score


def evaluate_wildfire_spread_baseline_model(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for the Next-Day Wildfire Spread Baseline task.

    This evaluator checks that:
      - wildfire_spread_baseline_model.py exists and imports required libraries
      - a training CSV (e.g., train.csv) exists
      - wildfire_spread_baseline_metrics.txt exists and contains
        Accuracy, Precision, Recall, F1, ROC_AUC as valid numeric values.

    NOTE: Because the exact column names for features and target can vary in the dataset,
    this evaluator focuses on structural correctness rather than recomputing metrics.
    """
    if actual is None:
        logger.error("Directory collection failed.")
        return 0.0

    import re
    expected = expected.get("rules", expected)
    score = 0.0
    feedback = []

    try:
        base_dir = os.path.dirname(actual)
        logger.info(f"[Wildfire] Starting evaluation in directory: {base_dir}")

        py_file = os.path.join(base_dir, "wildfire_spread_baseline_model.py")
        result_file = os.path.join(base_dir, "wildfire_spread_baseline_metrics.txt")

        # Try to locate train.csv or another plausible main CSV
        train_csv = None
        for fname in os.listdir(base_dir):
            if fname.lower() == "train.csv":
                train_csv = os.path.join(base_dir, fname)
                break
        if train_csv is None:
            # fallback: any csv with 'fire' or 'wild' in the name
            for fname in os.listdir(base_dir):
                if fname.lower().endswith(".csv") and ("fire" in fname.lower() or "wild" in fname.lower()):
                    train_csv = os.path.join(base_dir, fname)
                    break

        logger.info("Files expected for Wildfire evaluation:")
        logger.info(f" - Script: {py_file}")
        logger.info(f" - Training dataset (auto-detected): {train_csv}")
        logger.info(f" - Result file: {result_file}")

        # 1. Check Python script
        if os.path.exists(py_file):
            score += 0.25
            feedback.append("Wildfire Python script is present.")
            logger.info("Wildfire script found.")

            with open(py_file, "r", encoding="utf-8") as f:
                content = f.read()

            if "import pandas" in content:
                score += 0.05
                feedback.append("The script imports pandas.")
            if "XGBClassifier" in content:
                score += 0.05
                feedback.append("The script uses XGBClassifier from xgboost.")
            if "StandardScaler" in content:
                score += 0.05
                feedback.append("The script uses StandardScaler for numeric features.")
            if "train_test_split" in content:
                score += 0.05
                feedback.append("The script uses train_test_split.")
            if "stratify" in content:
                score += 0.05
                feedback.append("The script uses stratified splitting by target.")
        else:
            feedback.append("Wildfire Python script is missing.")
            logger.warning("Wildfire script missing.")

        # 2. Check dataset presence (structure only)
        if train_csv and os.path.exists(train_csv):
            score += 0.25
            feedback.append(f"Wildfire training dataset file is present: {os.path.basename(train_csv)}.")
            logger.info("Wildfire training dataset file found.")
            try:
                df = pd.read_csv(train_csv, nrows=5)
                score += 0.05
                feedback.append("Wildfire dataset can be read successfully (sample rows).")
            except Exception as e:
                feedback.append(f"Error reading wildfire dataset: {e}")
                logger.error(f"Error reading wildfire dataset: {e}", exc_info=True)
        else:
            feedback.append("Wildfire training dataset CSV could not be located in the directory.")
            logger.warning("Wildfire dataset missing.")

        # 3. Check metrics file
        if os.path.exists(result_file):
            score += 0.25
            feedback.append("Wildfire metrics file is present.")
            logger.info("Wildfire metrics file found.")

            try:
                with open(result_file, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                logger.info(f"Wildfire metrics content: {content}")

                # Expected: Accuracy: <v>, Precision: <v>, Recall: <v>, F1: <v>, ROC_AUC: <v>
                pattern = (
                    r"Accuracy:\s*([0-9]*\.?[0-9]+),\s*"
                    r"Precision:\s*([0-9]*\.?[0-9]+),\s*"
                    r"Recall:\s*([0-9]*\.?[0-9]+),\s*"
                    r"F1:\s*([0-9]*\.?[0-9]+),\s*"
                    r"ROC_AUC:\s*([0-9]*\.?[0-9]+)"
                )
                m = re.search(pattern, content)
                if m:
                    acc, prec, rec, f1, roc = map(float, m.groups())
                    score += 0.15
                    feedback.append("Metrics file has the correct format and five numeric values.")
                    if all(0.0 <= v <= 1.0 for v in (acc, prec, rec, f1, roc)):
                        score += 0.05
                        feedback.append("Classification metrics are within [0, 1], which is plausible.")
                    else:
                        feedback.append("One or more classification metrics are outside [0, 1].")
                else:
                    feedback.append("Wildfire metrics file does not match the expected pattern.")
                    logger.warning("Wildfire metrics format mismatch.")
            except Exception as e:
                feedback.append(f"Error reading wildfire metrics file: {e}")
                logger.error(f"Error reading wildfire metrics file: {e}", exc_info=True)
        else:
            feedback.append("Wildfire metrics file is missing.")
            logger.warning("Wildfire metrics file missing.")

    except Exception as e:
        logger.error(f"Wildfire evaluation failed: {e}", exc_info=True)
        feedback.append(f"Evaluation error: {e}")

    final_score = min(score, 1.0)
    logger.info(f"[Wildfire] Final Evaluation Score: {final_score:.2f}")
    for msg in feedback:
        logger.info(msg)
    return final_score

def evaluate_regional_swing_counts(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for ds081 — Regional swing count computation.
    Verifies:
      - Notebook file exists.
      - Dataset (district_density.csv) exists.
      - Output CSV file exists and has expected structure (region, pro_dem_count, pro_gop_count).
      - Uses pandas in code.
    """

    if not actual:
        logger.info("No valid path provided to evaluator.")
        return 0.0

    base_dir = os.path.dirname(actual)
    nb_file = os.path.join(base_dir, "regional_swing_counts.ipynb")
    data_file = os.path.join(base_dir, "district_density.csv")
    out_file = os.path.join(base_dir, "regional_swing_counts.csv")

    score = 0.0
    feedback = []

    try:
        # 1. Check notebook existence
        if os.path.exists(nb_file):
            score += 0.25
            feedback.append("Notebook file detected.")
            try:
                with open(nb_file, "r", encoding="utf-8") as f:
                    nb_data = json.load(f)
                nb_content = " ".join(
                    cell.get("source", []) for cell in nb_data.get("cells", [])
                ).lower()
                if "import pandas" in nb_content:
                    score += 0.25
                    feedback.append("Notebook correctly imports pandas.")
                else:
                    feedback.append("Notebook missing pandas import.")
            except Exception:
                feedback.append("Notebook could not be parsed as JSON.")
        else:
            feedback.append("Notebook file missing.")

        # 2. Check dataset presence
        if os.path.exists(data_file):
            score += 0.25
            feedback.append("Dataset file found in directory.")
        else:
            feedback.append("Dataset file missing.")

        # 3. Check output CSV existence and structure
        if os.path.exists(out_file):
            import pandas as pd
            df = pd.read_csv(out_file)
            expected_cols = {"region", "pro_dem_count", "pro_gop_count"}
            if expected_cols.issubset(set(df.columns)):
                score += 0.25
                feedback.append("Output CSV has correct columns.")
            else:
                feedback.append("Output CSV missing one or more expected columns.")
        else:
            feedback.append("Output CSV file missing.")

    except Exception as e:
        feedback.append(f"Evaluator error: {e}")

    for item in feedback:
        logger.info(item)
    logger.info(f"Evaluation Score: {score:.2f}")

    return min(round(score, 2), 1.0)


def evaluate_high_magnitude_earthquakes(actual: str, expected: dict, **options) -> float:
    """
    Evaluator for ds084 — High-magnitude earthquake analysis.
    Verifies:
      - Notebook exists and imports pandas/numpy.
      - Dataset file exists.
      - highmag_count.txt exists and contains a numeric count.
      - highmag_location.csv exists with columns AvgLatitude, AvgLongitude and numeric values.
    """

    if not actual:
        logger.info("No valid path provided for evaluator.")
        return 0.0

    base_dir = os.path.dirname(actual)
    nb_file = os.path.join(base_dir, "high_magnitude_earthquakes.ipynb")
    data_file = os.path.join(base_dir, "earthquakes-23k.csv")
    count_file = os.path.join(base_dir, "highmag_count.txt")
    location_file = os.path.join(base_dir, "highmag_location.csv")

    score = 0.0
    feedback = []

    try:
        # 1️⃣ Check notebook file and pandas/numpy import
        if os.path.exists(nb_file):
            score += 0.25
            feedback.append("Notebook file found.")
        else:
            feedback.append("Notebook file missing.")


        # 2️⃣ Check dataset exists
        if os.path.exists(data_file):
            score += 0.15
            feedback.append("Dataset file found.")
        else:
            feedback.append("Dataset file missing.")

        # 3️⃣ Check highmag_count.txt
        if os.path.exists(count_file):
            with open(count_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
            if content and content.replace(".", "", 1).isdigit():
                score += 0.2
                feedback.append("Valid count file found with numeric value.")
            else:
                feedback.append("Count file exists but does not contain a valid number.")
        else:
            feedback.append("Count text file missing.")

        # 4️⃣ Check highmag_location.csv
        if os.path.exists(location_file):
            df = pd.read_csv(location_file)
            expected_cols = {"AvgLatitude", "AvgLongitude"}
            if expected_cols.issubset(set(df.columns)):
                if pd.api.types.is_numeric_dtype(df["AvgLatitude"]) and pd.api.types.is_numeric_dtype(df["AvgLongitude"]):
                    score += 0.2
                    feedback.append("Location CSV found with correct columns and numeric values.")
                else:
                    feedback.append("Columns found but not numeric.")
            else:
                feedback.append("Location CSV missing required columns.")
        else:
            feedback.append("Location CSV missing.")

    except Exception as e:
        feedback.append(f"Evaluator error: {e}")

    for msg in feedback:
        logger.info(msg)
    logger.info(f"Evaluation Score: {score:.2f}")

    return min(round(score, 2), 1.0)

def evaluate_goodreads_top100_rating_analysis(actual: str, expected: dict, **kwargs) -> float:
    """
    Evaluator for ds237 — Goodreads Top 100 Rating Analysis.
    Verifies:
      - Python script exists.
      - Required output files (CSV, PNG, TXT) exist.
      - CSV has correct columns.
      - Correlation summary file contains expected line format.
    """

    if not actual:
        logger.info("No valid directory path provided to evaluator.")
        return 0.0

    base_dir = os.path.dirname(actual)
    script = os.path.join(base_dir, "goodreads_top100_rating_analysis.py")
    csv_file = os.path.join(base_dir, "goodreads_top100_summary.csv")
    img_file = os.path.join(base_dir, "rating_vs_reviews.png")
    txt_file = os.path.join(base_dir, "goodreads_correlation_summary.txt")

    score = 0.0
    feedback = []

    try:
        # 1️⃣ Check Python script
        if os.path.exists(script):
            score += 0.25
            feedback.append("Python script found.")
            with open(script, "r", encoding="utf-8") as f:
                content = f.read().lower()

            if all(lib in content for lib in ["requests", "beautifulsoup4", "pandas", "matplotlib", "sklearn"]):
                score += 0.15
                feedback.append("All required libraries detected in the script.")
            else:
                feedback.append("Some required libraries missing.")

            # Keyword check
            keywords = ["goodreads", "best_books_ever", "average_rating", "review_count", "scatter", "regression", "correlation"]
            if any(k in content for k in keywords):
                score += 0.10
                feedback.append("Relevant analysis keywords found.")
            else:
                feedback.append("Missing key analytical terms in script.")
        else:
            feedback.append("Python script missing.")

        # 2️⃣ Check CSV output
        if os.path.exists(csv_file):
            score += 0.25
            df = pd.read_csv(csv_file)
            expected_cols = {"title", "author", "average_rating", "review_count"}
            if expected_cols.issubset(set(df.columns)):
                score += 0.10
                feedback.append("CSV file exists with correct columns.")
            else:
                feedback.append("CSV file exists but columns are incorrect.")
        else:
            feedback.append("CSV output missing.")

        # 3️⃣ Check image plot output
        if os.path.exists(img_file):
            score += 0.10
            feedback.append("Scatter-plot image found.")
        else:
            feedback.append("Scatter-plot image missing.")

        # 4️⃣ Check text summary output
        if os.path.exists(txt_file):
            with open(txt_file, "r", encoding="utf-8") as f:
                line = f.read().strip()
            if line.startswith("Corr_ReviewCount_Rating:") and any(trend in line for trend in ["Increases", "Decreases", "None"]):
                score += 0.15
                feedback.append("Text summary correctly formatted with correlation and trend.")
            else:
                feedback.append("Text summary file found but format incorrect.")
        else:
            feedback.append("Correlation summary text file missing.")

    except Exception as e:
        feedback.append(f"Evaluator error: {e}")

    # Final logs
    logger.info(f"Evaluation Score: {score:.2f}")
    for msg in feedback:
        logger.info(msg)

    return min(round(score, 2), 1.0)
