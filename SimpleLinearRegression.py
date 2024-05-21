import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class LinearRegression:
    def __init__(self, learning_rate=0.00001, n_iters=1000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.slope = None
        self.intercept = None

    def fit(self, X, y):
        n_samples = len(X)
        self.slope = 0
        self.intercept = 0

        for _ in range(self.n_iters):
            for i in range(n_samples):
                y_predicted = self.intercept + self.slope * X[i]
                dw = X[i] * (y_predicted - y[i])
                db = y_predicted - y[i]
                self.slope -= self.learning_rate * dw
                self.intercept -= self.learning_rate * db

    def predict(self, X):
        return self.intercept + self.slope * X

# Load dữ liệu
df = pd.read_excel('D:\\Downloads\\Admission_Predict_Ver1.1.xlsx')
X = df['CGPA'].values
y = df['Chance of Admit '].values

# Chia thành tập train và tập test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, test_size=0.1, random_state=100)

# Huấn luyện mô hình tự code
model_custom = LinearRegression()
model_custom.fit(X_train, y_train)

# Dự đoán trên tập test với mô hình tự code
predicted_price_custom = model_custom.predict(X_test)

# Tính MSE của mô hình tự code
mse_custom = mean_squared_error(y_test, predicted_price_custom)
print("Mean Squared Error (Custom Model):", mse_custom)

# Sử dụng mô hình từ thư viện sklearn
from sklearn.linear_model import LinearRegression as SklearnLinearRegression

# Huấn luyện mô hình từ thư viện sklearn
model_sklearn = SklearnLinearRegression()
model_sklearn.fit(X_train.reshape(-1, 1), y_train)

# Dự đoán trên tập test với mô hình từ thư viện sklearn
predicted_price_sklearn = model_sklearn.predict(X_test.reshape(-1, 1))

# Tính MSE của mô hình từ thư viện sklearn
mse_sklearn = mean_squared_error(y_test, predicted_price_sklearn)
print("Mean Squared Error (Sklearn Model):", mse_sklearn)

# Vẽ đồ thị
plt.figure(figsize=(10, 6))

# Vẽ dữ liệu
plt.scatter(X_test, y_test, color='black', label='Actual')

# Vẽ đường hồi quy từ mô hình tự code
plt.plot(X_test, predicted_price_custom, color='blue',linestyle='dashed',  label='Custom Model')

# Vẽ đường hồi quy từ mô hình sklearn
plt.plot(X_test, predicted_price_sklearn, color='red', label='Sklearn Model')

plt.xlabel('CGPA')
plt.ylabel('Chance of Admit ')
plt.title('Linear Regression Comparison')
plt.legend()
plt.show()
