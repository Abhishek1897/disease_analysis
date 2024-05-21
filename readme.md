# Health Disease Prediction

## Overview
This project aims to build a machine learning model to predict the presence of a disease based on various health metrics. The model is trained and evaluated using data that includes features such as age, BMI, blood pressure, exercise frequency, smoking status, family history, diet quality, and cholesterol level.

## Dataset
- **Training Data**: `training_data.csv`
- **Testing Data**: `test_data.csv`

The dataset contains the following columns:
- `Unnamed: 0`: Index
- `gender`: Gender of the individual (Male/Female)
- `age`: Age of the individual
- `bmi`: Body Mass Index
- `systolic_bp`: Systolic Blood Pressure
- `diastolic_bp`: Diastolic Blood Pressure
- `exercise_frequency`: Frequency of exercise (Never, Rarely, Regularly, Frequently)
- `smoker`: Smoking status (0 for non-smoker, 1 for smoker)
- `family_history`: Family history of the disease (0 for no, 1 for yes)
- `diet_quality`: Quality of diet (Poor, Average, Good)
- `us_state`: US State
- `shoe_size`: Shoe size
- `cholesterol_level`: Cholesterol level
- `has_disease`: Target variable indicating the presence of disease (0 for no, 1 for yes)

## Exploratory Data Analysis (EDA)
1. **Initial Inspection**: Displayed the first few rows, shape, and a random sample of the data.
2. **Summary Statistics**: Used `describe()` to get an overview of the numerical features.
3. **Missing Values**: Identified missing values in the dataset.
4. **Outliers**: Detected outliers using box plots and Z-scores.

## Data Preprocessing
1. **Outliers Handling**: Removed rows with outliers based on Z-score (threshold > 3).
2. **Feature Encoding**: 
   - Dropped `us_state` and `Unnamed: 0` columns.
   - Applied one-hot encoding to categorical features.
3. **Standardization**: Standardized numerical features using `StandardScaler`.
4. **Missing Values Imputation**: Handled missing values using `SimpleImputer` with the mean strategy for numerical columns.

## Model Training and Evaluation
1. **Data Splitting**: Split the preprocessed data into training and testing sets (80% train, 20% test).
2. **Model Selection**:
   - **SGDClassifier**: Achieved an accuracy of approximately 86.90%.
   - **Logistic Regression**: Achieved an accuracy of approximately 88.16%.
3. **Cross-Validation**: Used 5-fold cross-validation to evaluate the models:
   - **Random Forest Classifier**: Achieved an average F1-score of 0.866.

## Cross-Validation Control
- Implemented cross-validation to ensure no data leakage and validate model performance.

## Visualization
- **Histograms**: Plotted the distribution of numerical features.
- **Correlation Heatmap**: Visualized relationships between numerical features.

## EDA and Processing of Testing Data
1. **Initial Inspection**: Displayed the first few rows, shape, and summary statistics of the testing data.
2. **Missing Values**: Identified missing values in the testing dataset.
3. **Outliers Handling**: Used Z-scores to identify and handle outliers in the testing data.

## Results
The final model demonstrated a robust performance with the following metrics:
- **Accuracy**: 88.16% (Logistic Regression)
- **Average F1-Score**: 0.866 (Random Forest Classifier)

## Conclusion
The project successfully built and evaluated a machine learning model to predict disease presence based on various health metrics. The preprocessing steps, including handling outliers, encoding categorical features, and standardizing numerical variables, were critical in achieving accurate predictions.

## Files
- `training_data.csv`: Training dataset.
- `test_data.csv`: Testing dataset.
- `README.md`: Project documentation.

## Future Work
- Improve feature engineering to extract more meaningful insights from the data.
- Explore more sophisticated models and hyperparameter tuning to enhance performance.
- Deploy the model for real-time predictions.

## Acknowledgments
Thanks to all contributors and the community for providing the resources and knowledge to complete this project.

---

**Author**: [Your Name]
