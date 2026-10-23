# Economic Slowdown Classification Using Statistical Machine Learning

## Project Overview

This project explores how statistical machine learning techniques can estimate the probability of an economic slowdown using economic indicators. Rather than predicting a binary outcome directly, the analysis focuses on producing probability estimates and evaluating their accuracy using the Brier score.

Completed as part of the **Statistical Machine Learning (ECON5129)** course at the University of Glasgow, the project compares multiple classification models using five-fold cross-validation.

**Note:** The dataset used in this exercise is simulated.

## Objectives

- Establish a baseline forecast using the proportion of slowdowns in the training data.
- Implement Logistic Regression for probabilistic classification.
- Compare alternative classification methods.
- Evaluate predicted probabilities using the Brier score.
- Use five-fold cross-validation to compare model performance.
- Select a final model for generating slowdown probabilities for unseen observations.

## Methodology

1. **Baseline forecast:** Predict the training-set slowdown proportion for every observation.
2. **Logistic Regression:** Estimate slowdown probabilities using a logistic regression model.
3. **Model comparison:** Evaluate Linear Discriminant Analysis (LDA), Quadratic Discriminant Analysis (QDA), K-Nearest Neighbours (KNN), and Naive Bayes.
4. **Cross-validation:** Apply five-fold cross-validation to compare the models using the Brier score.
5. **Model selection:** Consider predictive performance alongside interpretability when selecting the final model.
6. **Probability prediction:** Fit the selected Logistic Regression model on all 100 training observations and generate probabilities for the 40 test observations.

## Key Results

| Model | Five-fold cross-validation Brier score |
|---|---:|
| Baseline forecast | 0.2196 |
| Logistic Regression | 0.1205 |
| QDA | 0.1161 |
| Naive Bayes | 0.1129 |

Naive Bayes achieved the lowest reported Brier score, followed by QDA and Logistic Regression. Logistic Regression was selected for the final predictions because of its interpretability, despite not achieving the lowest cross-validation score.

The results illustrate the importance of comparing probabilistic classification models and balancing predictive performance with interpretability.

## Tools and Techniques

- **Programming language:** Python
- **Development environment:** Google Colab
- **Classification:** Logistic Regression, LDA, QDA, KNN and Naive Bayes
- **Validation:** Five-fold cross-validation
- **Evaluation metric:** Brier score
- **Libraries:** Pandas, NumPy and Scikit-learn

## Repository Structure

| File | Description |
|---|---|
| `README.md` | Project overview, methodology, results and tools used. |
| `economic_slowdown_classification.py` | Python source code for the baseline forecast, classification models, cross-validation, Brier score comparison and final probability predictions. |
| `ex2_train.csv` | Training dataset containing 100 observations, the binary slowdown outcome and economic indicators. |
| `ex2_test.csv` | Test dataset containing 40 observations for which slowdown probabilities are generated. |
| `ex2_variables.csv` | Descriptions of the dataset variables and economic indicators. |

## Skills Demonstrated

- Probabilistic classification
- Logistic Regression and alternative classification algorithms
- Cross-validation and model evaluation
- Brier score analysis
- Model comparison and selection
- Python-based statistical analysis
- Interpretation of predictive results

---

*Academic project — Statistical Machine Learning (ECON5129), University of Glasgow.*
