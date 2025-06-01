
# 🫀 Heart Disease Prediction using Machine Learning

This project aims to predict whether a person has heart disease or not using a machine learning model. The dataset contains health-related information such as age, blood pressure, cholesterol, and more. This notebook walks through loading the data, preprocessing, training a model, and evaluating its performance.

---

## 📁 Project Structure

```
Heart_Disease.ipynb        # Jupyter notebook with code, EDA, model training
```

---

## 📊 Dataset

- **Source**: Likely based on the [UCI Heart Disease dataset](https://archive.ics.uci.edu/ml/datasets/heart+Disease)
- **Columns include**:
  - `age`: Age of the person
  - `sex`: Gender (1 = male; 0 = female)
  - `cp`: Chest pain type (0-3)
  - `trestbps`: Resting blood pressure
  - `chol`: Serum cholesterol in mg/dl
  - `fbs`: Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
  - `restecg`: Resting electrocardiographic results (0,1,2)
  - `thalach`: Maximum heart rate achieved
  - `exang`: Exercise-induced angina
  - `oldpeak`: ST depression
  - `slope`, `ca`, `thal`: Other medical features
  - `target`: 1 = heart disease present, 0 = not present

---

## 🚀 Steps Performed

### 1. **Importing Libraries**
Standard Python libraries like `pandas`, `numpy`, `matplotlib`, and `sklearn` are used for data handling, visualization, and modeling.

### 2. **Loading the Data**
CSV file is loaded into a Pandas DataFrame for exploration.

### 3. **Exploratory Data Analysis (EDA)**
- Dataset preview using `head()`
- Checking missing values
- Understanding column types and data distribution

### 4. **Preprocessing**
- Feature selection and target separation
- Scaling features using **MinMaxScaler**
- Train/test splitting

### 5. **Model Building**
- A `Sequential` neural network model is built using Keras
- Layers use `ReLU` activation and `sigmoid` in the final layer
- `binary_crossentropy` is used as the loss function
- Optimizer: `adam`

### 6. **Model Evaluation**
- Accuracy and loss are printed after training
- Predictions on test set are compared to ground truth

---

## ⚙️ How to Use

1. Install required libraries:
```bash
pip install numpy pandas matplotlib scikit-learn tensorflow
```

2. Run the notebook:
Open `Heart_Disease.ipynb` in Jupyter or Colab and run each cell in order.

---

## 📈 Model Summary

| Metric     | Value       |
|------------|-------------|
| Model Type | Neural Network (Sequential) |
| Accuracy   | ~90% (May vary) |
| Loss       | Binary Crossentropy |

---

## 🧠 Concepts Covered

- Binary classification
- Neural networks
- Data preprocessing
- Activation functions (`ReLU`, `Sigmoid`)
- Evaluation metrics

---

## ✍️ Author

Created by a learner exploring machine learning with real-world health data.
