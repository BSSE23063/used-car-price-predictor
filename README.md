# Used Car Price Predictor 

A Machine Learning model that estimates the market value of used cars based on mileage, year, horsepower, and other features.

##  Project Overview
* **Goal:** Predict used car prices to help buyers/sellers estimate value.
* **Algorithm:** Multivariate Linear Regression.
* **Accuracy:** Achieved an R² Score of **0.XX** (Replace with your score).

##  Tech Stack
* Python
* Pandas (Data Cleaning & Manipulation)
* Scikit-Learn (Model Training)
* Matplotlib/Seaborn (Visualization)

##  Key Steps
1.  **Data Cleaning:** Handled missing values, removed 'garbage' text (e.g., "300HP" → 300), and filtered outliers.
2.  **Feature Engineering:** Extracted Horsepower from engine text and calculated 'Car Age'.
3.  **One-Hot Encoding:** Converted categorical data (Brand, Fuel Type) into numeric format.
4.  **Model Training:** Trained on 80% of data, tested on 20%.

##  How to Run
1.  Clone the repository:
    ```bash
    [git clone (https://github.com/BSSE23063/used-car-price-predictor)
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Download the dataset from Kaggle and place it in the folder.
4.  Run the script:
    ```bash
    python main.py
    ```
