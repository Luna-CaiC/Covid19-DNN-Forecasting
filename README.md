# COVID-19 Daily Case Prediction using Deep Learning

## 📌 Project Overview
This project implements a **Deep Neural Network (DNN)** to predict the number of daily tested positive COVID-19 cases based on a variety of features spanning multiple days. By leveraging historical data and regression techniques, the model aims to accurately forecast trends to assist in epidemiological analysis.

The solution is built using **PyTorch** and follows a rigorous data science workflow including feature engineering, normalization, regularization, and performance evaluation.

## ⚙️ Technologies & Dependencies
The project is developed using **Python 3.12** and relies on the following core libraries:

- **PyTorch (2.8.0)**: Core deep learning framework for building and training the neural network.
- **Pandas & NumPy**: For efficient data manipulation, processing, and vectorization.
- **Scikit-Learn**: Used for data splitting (`train_test_split`) and feature scaling (`MinMaxScaler`).
- **Matplotlib**: (Optional) For visualizing training progress and loss curves.

**System Requirements:**
- CPU or CUDA-enabled GPU (The code automatically detects the available device).

## 📊 Dataset & Preprocessing
The model utilizes structured CSV data ([covid.train.csv](cci:7://file:///Users/lucky/Documents/3rd%20Semester/INFO536-AML/project1/Proj1github/covid.train.csv:0:0-0:0), [covid.test.csv](cci:7://file:///Users/lucky/Documents/3rd%20Semester/INFO536-AML/project1/Proj1github/covid.test.csv:0:0-0:0)) containing daily state-level statistics. 

### Data Pipeline:
1.  **Feature Selection**: The target variable is identified as `tested_positive.2`. Irrelevant columns (like `id`) are dropped to prevent leakage.
2.  **One-Hot Encoding**: Categorical variables (e.g., state or region identifiers) are converted into numerical vectors using `pd.get_dummies()`.
3.  **Feature Alignment**: Ensures the training and testing datasets have identical feature columns, filling missing columns in the test set with zeros.
4.  **Normalization**: Features are scaled to a range of `[0, 1]` using `MinMaxScaler` to accelerate convergence and improve numerical stability.
5.  **Data Splitting**: The training data is split into **80% Training** and **20% Validation** sets to monitor overfitting.

## 🧠 Model Architecture
The project implements a custom Feed-Forward Neural Network (`DNN` class) designed for regression tasks.

**Structure:**
- **Input Layer**: Dimension matches the number of preprocessed features.
- **Hidden Layer 1**: 128 Neurons | Activation: **ReLU** | **Dropout (30%)**
- **Hidden Layer 2**: 64 Neurons  | Activation: **ReLU** | **Dropout (30%)**
- **Output Layer**: 1 Neuron (Linear activation for continuous value prediction)

*Note: Dropout layers are strategically placed to randomly drop neurons during training, serving as a powerful regularization technique to prevent overfitting.*

## 📉 Training Configuration
The training process is optimized for stability and accuracy:

- **Loss Function**: Mean Squared Error (**MSE**), appropriate for regression.
- **Optimizer**: **Adam** (`lr=0.001`, `weight_decay=1e-5`) for adaptive learning rates and L2 regularization.
- **Training Strategy**:
    - **Max Epochs**: 120
    - **Early Stopping**: Monitors validation loss with a patience of **10 epochs**. If no improvement is observed, training stops to save the best model state.
    - **Model Checkpointing**: The weights yielding the lowest validation loss are restored before final inference.

## 🏆 Performance Results
Based on the final training logs:
- **Best Validation MSE**: ~5.85
- **Final Validation RMSE**: ~2.42

The Root Mean Square Error (RMSE) of 2.42 indicates that, on average, the model's predictions deviate from the actual positive case counts by approximately 2.4 cases on the standardized/transformed scale.

## 🚀 Usage Instructions
1.  **Setup Environment**:
    Ensure all requirements are installed:
    ```bash
    pip install torch torchvision pandas scikit-learn matplotlib
    ```

2.  **Run the Notebook**:
    Execute [AML_Project1.ipynb](cci:7://file:///Users/lucky/Documents/3rd%20Semester/INFO536-AML/project1/Proj1github/AML_Project1.ipynb:0:0-0:0) in a Jupyter environment. The script will:
    - Load and preprocess data.
    - Train the DNN model.
    - Print training logs (MSE per epoch).
    - Generate predictions.

3.  **Output**:
    The script generates a `covid_predictions.csv` file containing the `id` and the predicted `tested_positive` values, ready for submission or analysis.
