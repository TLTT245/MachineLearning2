import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

class MultipleLinearRegression:
    def __init__(self, learning_rate=0.00005, n_iters=100000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.intercept = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.intercept = 0

        for _ in range(self.n_iters):
            y_predicted = self.predict(X)
            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1/n_samples) * np.sum(y_predicted - y)

            self.weights -= self.learning_rate * dw
            self.intercept -= self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.intercept

# Đọc dữ liệu
df = pd.read_csv('D:\\Downloads\\advertising.csv')
X = df[['TV', 'Newspaper', 'Radio']]
y = df['Sales']

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

# Huấn luyện mô hình sklearn
model_sklearn = LinearRegression()
model_sklearn.fit(X_train, y_train)

# Huấn luyện mô hình tự code
model_custom = MultipleLinearRegression()
model_custom.fit(X_train, y_train)

# Dự đoán từ cả hai mô hình
y_pred_sklearn = model_sklearn.predict(X_test)
y_pred_custom = model_custom.predict(X_test)

r_squared_sklearn = r2_score(y_test, y_pred_sklearn)
mse_sklearn = mean_squared_error(y_test, y_pred_sklearn)
per_e_sklearn = mean_absolute_percentage_error(y_test,y_pred_sklearn)
acc1_sklearn = (1-per_e_sklearn)
print('accuracy sk',acc1_sklearn)
print('r_squared sk  ',r_squared_sklearn)
r_squared = r2_score(y_test, y_pred_custom)
mse = mean_squared_error(y_test, y_pred_custom)
per_e = mean_absolute_percentage_error(y_test,y_pred_custom)
acc1 = (1-per_e)
print('Accuracy: ',acc1)
print('R_squared: ',r_squared)
# Vẽ đồ thị hồi quy 3D
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# Vẽ dữ liệu thực tế
ax.scatter(X_test['TV'], X_test['Radio'], y_test, color='black', label='Actual Sales')

# Vẽ đường hồi quy từ mô hình sklearn
surf_sklearn = ax.plot_trisurf(X_test['TV'], X_test['Radio'], y_pred_sklearn, color='blue', alpha=0.5)

# Vẽ đường hồi quy từ mô hình tự code
surf_custom = ax.plot_trisurf(X_test['TV'], X_test['Radio'], y_pred_custom, color='red', alpha=0.5)

# Thêm tiêu đề và nhãn cho trục
ax.set_title('3D Regression Plot')
ax.set_xlabel('TV')
ax.set_ylabel('Radio')
ax.set_zlabel('Sales')

# Hiển thị chú thích
legend1 = plt.Line2D([0], [0], color='blue', lw=4, label='Sklearn Model')
legend2 = plt.Line2D([0], [0], color='red', lw=4, label='Custom Model')
legend3 = plt.Line2D([0], [0], color='black', marker='o', linestyle='', label='Actual Sales')
ax.legend(handles=[legend1, legend2, legend3])

# Hiển thị đồ thị
plt.show()