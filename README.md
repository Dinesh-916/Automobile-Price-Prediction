# Automobile Imports Price Prediction
### PRCP-1017 | Data Science Capstone Project

![Python](https://img.shields.io/badge/Python-Data%20Science-blue)
![Machine Learning](https://img.shields.io/badge/ML-Regression-orange)
![Notebook](https://img.shields.io/badge/Format-Jupyter%20Notebook-orange)
![Project](https://img.shields.io/badge/Project-PRCP--1017-success)

## 1. Project Overview

Automobile Imports Price Prediction is a supervised machine learning project that explores the relationship between automobile specifications and vehicle prices. The project uses historical automobile-import data to analyze pricing patterns, prepare the data for modelling, train and compare regression algorithms, and interpret the factors associated with model predictions.

The central business question is:

> **Can the price of an automobile be estimated from its technical, physical, and categorical specifications, and which specifications are most informative for that estimate?**

The project is implemented in one Jupyter Notebook and covers the full workflow: data validation, exploratory data analysis (EDA), data cleaning, feature engineering, preprocessing, model training, cross-validation, hyperparameter tuning, final evaluation, model interpretation, challenge documentation, and model-artifact creation.

**Important scope note:** This is a historical dataset describing 1985 automobile imports. It is suitable for studying historical pricing relationships and demonstrating a regression workflow. It should not be used as a current-market car valuation tool without collecting recent data and conducting further validation.

---

## 2. Business Problem and Motivation

Automobile prices are associated with many factors, including manufacturer, engine capacity, horsepower, vehicle dimensions, weight, body style, fuel type, and fuel efficiency. Reviewing these factors manually can make comparisons slow and inconsistent.

A predictive model can provide decision support by:

- estimating a reference price from a vehicle's recorded specifications;
- helping compare vehicles with similar or different specifications;
- highlighting vehicles whose observed prices differ substantially from model estimates;
- showing which attributes contribute most to the model's predictions; and
- supporting analysis of historical pricing patterns.

The model identifies statistical relationships in the supplied data. It does **not** prove that changing a particular design feature will cause a specific change in price.

---

## 3. Project Objectives

1. Load the supplied automobile dataset and validate its structure.
2. Understand numerical and categorical attributes through descriptive statistics and visual analysis.
3. identify missing values, duplicate records, unexpected values, outliers, and relationships among variables.
4. Prepare the data without allowing information from the test set to leak into model training.
5. Create meaningful derived features from existing vehicle specifications.
6. Train and compare multiple regression algorithms.
7. Use cross-validation and hyperparameter tuning to support model selection.
8. Evaluate the selected model on a held-out test set.
9. Interpret model predictions using permutation feature importance and, when supported by the environment, SHAP.
10. Document data challenges, business implications, limitations, and future improvements.
11. Save the fitted preprocessing-and-model pipeline and supporting evaluation artifacts.

---

## 4. Dataset Description

### Dataset source

The dataset is the **1985 Auto Imports Database**, documented in the accompanying `auto_imports_names.txt` file. The documentation identifies historical sources including *1985 Model Import Car and Truck Specifications, 1985 Ward's Automotive Yearbook*, insurance information, and collision-report information.

### Dataset structure

| Item | Details |
|---|---|
| Dataset supplied | `auto_imports.csv` |
| Documentation | `auto_imports_names.txt` |
| Rows in supplied CSV | 201 |
| Columns | 26 |
| Predictor variables | 25 |
| Target variable | `price` |
| Column header in raw CSV | No header row |
| Missing-value marker | `?` |
| Documented original dataset size | 205 instances |

The source documentation describes 205 instances and reports four missing target prices. The supplied CSV contains 201 records; the notebook validates this supplied file and assigns the 26 documented column names.

### Target variable

- **`price`** — the automobile price to be predicted.

### Predictor groups

| Group | Example attributes | What they describe |
|---|---|---|
| Risk and loss | `symboling`, `normalized-losses` | Insurance risk rating and normalized loss information |
| Vehicle identity | `make` | Manufacturer |
| Fuel and aspiration | `fuel-type`, `aspiration`, `fuel-system` | Fuel and engine induction/fuel-system details |
| Body and layout | `num-of-doors`, `body-style`, `drive-wheels`, `engine-location` | Vehicle form and mechanical layout |
| Dimensions and mass | `wheel-base`, `length`, `width`, `height`, `curb-weight` | Vehicle size and weight |
| Engine | `engine-type`, `num-of-cylinders`, `engine-size`, `bore`, `stroke`, `compression-ratio` | Engine configuration and measurements |
| Performance | `horsepower`, `peak-rpm` | Engine output and operating speed |
| Fuel economy | `city-mpg`, `highway-mpg` | City and highway fuel efficiency |

### Missing values observed in the supplied CSV

The raw file uses `?` for unknown values. After interpreting that marker as missing, the supplied data contains:

| Column | Missing values |
|---|---:|
| `normalized-losses` | 37 |
| `num-of-doors` | 2 |
| `bore` | 4 |
| `stroke` | 4 |
| `horsepower` | 2 |
| `peak-rpm` | 2 |
| **Total missing cells** | **51** |

There are 42 rows with at least one missing value. The target `price` has no missing values in the supplied 201-row CSV.

---

## 5. Repository Structure

The GitHub repository is organized around the following files:

```text
Automobile-Price-Prediction/
│
├── PRCP_1017_AutoPricePred.ipynb
│   └── Complete analysis, modelling, evaluation, and reporting
│
├── auto_imports.csv
│   └── Raw automobile dataset (headerless CSV)
│
├── auto_imports_names.txt
│   └── Original dataset description and attribute information
│
└── README.md
    └── Project documentation
```

The notebook creates additional working folders and model artifacts when it is executed. These generated files are not required as separate inputs for the notebook's main workflow.

---

## 6. Technology Stack

| Technology / library | Purpose |
|---|---|
| Python | Main programming language |
| Jupyter Notebook | Interactive project implementation and reporting |
| Pandas | Data loading, cleaning, transformation, and tabular analysis |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Preprocessing, regression, pipelines, cross-validation, tuning, and evaluation |
| Joblib | Saving and reloading the fitted model pipeline |
| SHAP (optional) | Model explanation when installed and compatible |
| XGBoost (optional) | Additional regression candidate when installed |

The notebook imports the libraries used in each project phase. SHAP and XGBoost are optional in the notebook; their absence should not prevent the core workflow from running.

---

## 7. End-to-End Project Workflow

### Phase 1 — Data loading and validation

- Read the CSV without a header row.
- Assign column names in the order specified by the dataset documentation.
- Preserve the raw values initially so that `?` markers can be counted.
- Verify the expected dimensions: 201 rows and 26 columns.
- Record a SHA-256 fingerprint of the raw CSV and verify that the source file remains unchanged during cleaning.

**Why this matters:** Validating the source structure early helps detect incorrect files, missing columns, or unexpected changes before analysis begins.

### Phase 2 — Data quality and cleaning

The notebook:

- replaces `?` with proper missing-value markers (`NaN`);
- converts documented numerical columns to numeric data types;
- checks missing-value counts and percentages;
- checks fully duplicated rows;
- checks rows with identical predictor values but potentially different target prices;
- validates numerical values against documented ranges; and
- checks categorical values against documented category lists.

The raw dataset is retained. Imputation and encoding are not performed prematurely; they are handled later inside the modelling pipeline.

### Phase 3 — Exploratory Data Analysis (EDA)

EDA is used to understand the target, predictor distributions, and relationships before selecting modelling approaches.

The notebook includes:

- descriptive statistics for numerical variables;
- category counts for categorical variables;
- histograms and distribution plots;
- price comparisons across categories;
- a numerical correlation heatmap;
- Pearson and Spearman correlation analysis;
- scatter plots for numeric predictors with strong relationships to price;
- multicollinearity checks, including high-correlation pairs and variance inflation factor (VIF);
- outlier inspection using the 1.5 × IQR rule; and
- analysis of missing-value patterns.

**Selected observations recorded in the notebook:**

- The target price distribution is right-skewed.
- `engine-size`, `curb-weight`, and `horsepower` show strong positive numeric associations with price in the supplied data.
- `city-mpg` and `highway-mpg` show negative associations with price.
- `city-mpg` and `highway-mpg` are highly correlated with each other.
- Several vehicle size and engine variables contain overlapping information.
- Some extreme observations correspond to genuine high-end vehicles and are retained rather than automatically removed.

These are dataset-level associations, not causal conclusions.

### Phase 4 — Feature engineering

The notebook creates additional predictors from existing vehicle specifications.

| Engineered feature | Calculation / definition | Purpose |
|---|---|---|
| `cylinders` | Maps cylinder labels such as `four` and `six` to numbers | Represents cylinder count numerically |
| `doors` | Maps `two` and `four` to numbers | Represents door count numerically |
| `power-to-weight` | `horsepower / curb-weight` | Represents power relative to vehicle mass |
| `engine-size-per-cylinder` | `engine-size / cylinders` | Represents engine size per cylinder |
| `body-volume-k` | `length × width × height / 1000` | Creates a combined size measure |
| `avg-mpg` | Average of city and highway MPG | Summarizes fuel economy |
| `mpg-gap` | `highway-mpg - city-mpg` | Represents the difference between highway and city efficiency |
| `bore-stroke-ratio` | `bore / stroke` | Represents an engine geometry ratio |
| `premium-make` | Predefined indicator for selected makes | Encodes a stated brand-positioning assumption |

The feature-engineering function uses each row's own input specifications and does not use the target price. It is included inside the complete model pipeline so the same transformations are applied during training and prediction.

**Note:** `premium-make` is a predefined domain assumption about historical brand positioning. It is not a universal definition of a premium manufacturer, and its usefulness should be judged by model validation results.

### Phase 5 — Data preprocessing

The notebook uses Scikit-learn pipelines and a `ColumnTransformer` to handle numerical and categorical variables separately.

| Data type | Processing | Reason |
|---|---|---|
| Numerical predictors | Median imputation | Provides a robust replacement for missing numerical values |
| Numerical predictors with missing values | Missingness indicators | Preserves information that a source value was absent |
| Numerical predictors | Standard scaling | Supports scale-sensitive algorithms such as linear models and KNN |
| Categorical predictors | Most-frequent imputation | Handles missing categorical values |
| Categorical predictors | One-hot encoding | Converts categories into numeric model inputs without imposing a false order |

The original categorical cylinder and door-count columns are replaced by their engineered numeric forms in the model feature set. The target `price` is excluded from all predictor transformations.

### Phase 6 — Train-test split and leakage prevention

- Training and testing are separated using an **80:20 split**.
- `random_state=42` is used for reproducibility.
- The split is stratified using price quintiles so that the target-price range is represented in both subsets.
- The supplied 201 records produce 160 training records and 41 held-out test records.
- Imputation, scaling, and encoding are fitted using training data only.
- Cross-validation and hyperparameter tuning use the training data; the test set is reserved for final evaluation.

**Why this matters:** If preprocessing learns statistics from the test set, the evaluation can become overly optimistic. Keeping transformations inside the pipeline helps prevent this leakage.

### Phase 7 — Regression model development

The notebook compares the following regression algorithms:

| Algorithm | Main idea |
|---|---|
| Linear Regression | Models a linear relationship between predictors and price |
| Ridge Regression | Uses L2 regularization to reduce coefficient magnitude |
| Lasso Regression | Uses L1 regularization and can reduce less useful coefficients |
| ElasticNet Regression | Combines L1 and L2 regularization |
| K-Nearest Neighbors Regression | Predicts using similar observations |
| Decision Tree Regression | Learns rule-based splits in the feature space |
| Random Forest Regression | Averages predictions from multiple decision trees |
| Extra Trees Regression | Uses randomized tree construction in an ensemble |
| Gradient Boosting Regression | Builds sequential trees to correct earlier errors |
| HistGradientBoosting Regression | Uses histogram-based gradient boosting |
| XGBoost Regression (optional) | Adds an optimized boosting candidate if the package is available |

All evaluated models use the complete feature-engineering and preprocessing pipeline.

### Phase 8 — Cross-validation and model comparison

The notebook uses **5-fold cross-validation** on the training data.

The training set is divided into five folds. Each fold is used once for validation while the remaining folds are used for fitting. This gives a more stable view of performance than relying on a single training/validation split.

The notebook compares:

- training MAE, RMSE, and R²;
- mean and standard deviation of cross-validation MAE;
- mean and standard deviation of cross-validation RMSE;
- mean and standard deviation of cross-validation R²; and
- training-versus-validation gaps as a diagnostic for possible overfitting.

The primary selection metric is **cross-validation MAE**, where lower values indicate smaller average absolute prediction errors.

### Phase 9 — Hyperparameter tuning

The notebook applies `RandomizedSearchCV` to selected ensemble candidates:

- Random Forest;
- Extra Trees; and
- Gradient Boosting.

Ten parameter combinations are sampled per selected candidate using five-fold cross-validation. The search uses negative MAE as its scoring function. The test set is not used for tuning.

The tuned candidates are compared with their original cross-validation scores.

### Phase 10 — Final model selection and evaluation

The selected candidate is chosen using training-set cross-validation MAE, incorporating tuned CV scores where available. The held-out test set is then used once for final evaluation.

The notebook reports:

| Metric | Interpretation | Preferred direction |
|---|---|---|
| MAE | Average absolute difference between actual and predicted prices | Lower |
| MSE | Average squared prediction error; penalizes larger errors more heavily | Lower |
| RMSE | Square root of MSE, expressed in the target's units | Lower |
| R² | Proportion of target variation explained by the model on the evaluated data | Higher |
| Median Absolute Error | Median absolute prediction error | Lower |
| MAPE | Mean absolute percentage error for non-zero actual target values | Lower |

The notebook intentionally does not report adjusted R² for a selected nonlinear tree ensemble. The final conclusion should be based on the actual executed model comparison and test metrics, not on assumed or manually entered scores.

### Phase 11 — Model interpretation

The notebook uses:

**Permutation Feature Importance**
- Measures how model evaluation changes when a feature is shuffled.
- A larger deterioration in the selected scoring metric indicates that the model relies more on that feature for its predictions.
- The notebook uses negative MAE scoring and repeats the permutation process.

**SHAP (optional)**
- Explains how transformed input features contribute to individual model predictions.
- Uses a tree explainer for supported tree models and a model-agnostic explainer for other estimators.
- May be skipped if SHAP is unavailable or incompatible with the installed environment.

Feature importance and SHAP values explain model behaviour. They do not establish causal effects.

### Phase 12 — Prediction and artifact saving

The notebook:

- saves the complete fitted pipeline using Joblib;
- reloads the saved pipeline and checks that its predictions match the in-memory model;
- defines a helper function for predicting a vehicle price from a dictionary of raw specifications;
- checks a sample prediction against direct pipeline prediction; and
- exports model comparison, tuned comparison, prediction, feature-importance, final-metric, and challenge-report files.

The saved pipeline includes feature engineering, preprocessing, and the selected estimator. The notebook notes that a separate script loading the Joblib artifact must define the same `engineer_features` function before loading it.

---

## 8. Model Evaluation and How to Read the Results

The notebook produces model-comparison tables and diagnostic plots. After running all cells, review these outputs:

1. **Training model comparison:** Shows how closely each model fits the training data.
2. **Cross-validation comparison:** Shows performance across training folds and is the primary basis for model selection.
3. **Tuning comparison:** Shows whether the tested hyperparameter search improved the selected candidates' CV MAE.
4. **Actual vs. predicted plot:** Points closer to the diagonal represent more accurate predictions.
5. **Residual distribution:** Shows the distribution of prediction errors.
6. **Residuals vs. predicted plot:** Helps identify patterns such as uneven error spread.
7. **Permutation importance / SHAP:** Helps explain which predictors influence model output.

**Do not select a model solely because it has the lowest training error.** A model that performs well on training observations but poorly during cross-validation may be overfitting. The notebook therefore selects by cross-validation MAE and reports final test-set metrics separately.

No fixed model winner or performance score is stated in this README because those values must come from the notebook's actual execution in the environment used for submission.

---

## 9. Challenges and How They Were Addressed

| Challenge | Why it matters | Technique used in the notebook |
|---|---|---|
| Missing values marked as `?` | The marker is not a valid numeric value and can disrupt analysis or modelling | Convert `?` to `NaN`; impute within training-fitted pipelines |
| Mixed numerical and categorical features | Algorithms require numeric model inputs | Separate numerical and categorical preprocessing with `ColumnTransformer` |
| Different feature scales | Large-valued variables can dominate scale-sensitive models | Standard scaling for numerical predictors |
| Missingness may contain information | A missing value can be associated with a particular data pattern | Add missingness indicators for numerical columns with missing values |
| Correlated predictors | Redundant variables can destabilize linear-model coefficients | Correlation and VIF diagnostics; compare regularized and tree-based models |
| Genuine extreme vehicles | Automatically removing high-price or unusual vehicles can remove meaningful examples | Inspect outliers and retain observations judged to be genuine |
| Small dataset | A single split can give an unstable estimate of generalization | Five-fold cross-validation and a held-out test set |
| Possible overfitting | Flexible models can fit training data too closely | Compare training and CV metrics; tune selected ensemble models |
| Model interpretability | A prediction alone does not explain which inputs the model relies on | Permutation importance and optional SHAP |
| Identical predictors with differing prices | The available columns do not explain every price difference | Identify and document such rows rather than automatically deleting them |
| Historical data | Historical pricing patterns may not represent current markets | State the limitation and recommend validation on recent data |

---

## 10. Business Recommendations

The model can be considered a historical pricing-analysis and decision-support exercise.

Potential uses include:

- **Dealers:** Compare a vehicle's recorded price with a model estimate as an additional review signal.
- **Importers:** Examine how historical specifications relate to recorded prices when comparing vehicles.
- **Pricing analysts:** Investigate large differences between actual and predicted values.
- **Management:** Review model explanations to understand which attributes are most informative in the historical dataset.

The model should not be used to set current retail prices or make financial decisions without recent representative data, external validation, and ongoing monitoring.

---

## 11. How to Run the Project

### Prerequisites

- Python 3
- Jupyter Notebook or JupyterLab
- The project notebook and both supplied data files in the repository

### Step 1 — Clone the repository

```bash
git clone https://github.com/<your-username>/Automobile-Price-Prediction.git
cd Automobile-Price-Prediction
```

Replace `<your-username>` with your GitHub username.

### Step 2 — Install the main dependencies

```bash
python -m pip install pandas numpy matplotlib seaborn scikit-learn joblib jupyter
```

Optional libraries:

```bash
python -m pip install shap xgboost
```

### Step 3 — Open Jupyter Notebook

```bash
jupyter notebook
```

### Step 4 — Run the notebook

Open `PRCP_1017_AutoPricePred.ipynb` and execute the cells from top to bottom.

The notebook checks for the raw CSV in `data/raw/` and falls back to `auto_imports.csv` in the current working directory. For the repository layout shown above, launch Jupyter from the repository root so the fallback can locate the CSV.

Keep `auto_imports.csv` unchanged. The notebook assigns column names internally because the supplied CSV has no header row.

---

## 12. Limitations

1. **Historical period:** The data describes 1985 automobile imports and cannot represent today's vehicle market without new data.
2. **Limited sample size:** The supplied file contains 201 records, so estimates of generalization may be uncertain.
3. **Incomplete attributes:** Some potentially important pricing factors are not present in the dataset.
4. **Unexplained price differences:** Identical recorded predictors can occur with different target prices.
5. **Market changes:** Inflation, taxes, supply and demand, vehicle condition, mileage, location, and changing consumer preferences are not fully represented.
6. **Prediction is not causation:** Feature importance does not show that changing a feature will cause a particular price change.
7. **Deployment validation:** The saved pipeline is an artifact from the notebook, not a fully monitored production service.

---

## 13. Future Enhancements

- Obtain a larger, more recent, and geographically relevant automobile dataset.
- Add vehicle age, mileage, condition, service history, location, and market-demand information where available.
- Evaluate target transformations and robust modelling strategies for the right-skewed price distribution.
- Use repeated or nested validation when a larger dataset is available.
- Validate predictions on external, unseen market data.
- Build a user interface or API for entering vehicle specifications.
- Monitor prediction errors and retrain when market conditions or data distributions change.
- Add automated tests and a reproducible environment file such as `requirements.txt`.

---

## 14. Project Deliverables

The main deliverable is:

- `PRCP_1017_AutoPricePred.ipynb` — the single Jupyter Notebook containing the analysis, model development, comparison, evaluation, explainability, challenges, recommendations, and artifact-saving workflow.

Supporting source files:

- `auto_imports.csv` — raw dataset.
- `auto_imports_names.txt` — dataset documentation.
- `README.md` — this project guide.

---

## 15. Author

**Project:** PRCP-1017 — Automobile Imports Price Prediction  
**Category:** Data Science / Machine Learning  
**Problem type:** Supervised Learning — Regression  
**Primary evaluation metric:** Cross-validation MAE  
**Implementation:** Python and Jupyter Notebook

---

## Disclaimer

This project is developed for educational and portfolio purposes using a historical automobile dataset. Its outputs describe patterns learned from the supplied data and must not be interpreted as current automobile valuations, financial advice, or proof of causal relationships.
