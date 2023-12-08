# Импорт необходимых библиотек
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Загрузка данных
df = pd.read_csv('C:\Users\diasa\Desktop\fb\fake_bills.csv')

# Начальный анализ и очистка данных
print(df.isnull().sum())
df = df.dropna()
print(df.duplicated().sum())
df = df.drop_duplicates()

# Детальный анализ (EDA)
print(df.describe())
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()

# Визуализация важных переменных
important_vars = ['diagonal', 'height_left', 'height_right', 'margin_low', 'margin_upper', 'length']
sns.pairplot(df[important_vars], hue='is_genuine')
plt.show()

# Выдвижение и проверка гипотезы
fake_bills = df[df['is_genuine'] == 0]
genuine_bills = df[df['is_genuine'] == 1]

for var in important_vars:
    _, p_value = ttest_ind(fake_bills[var], genuine_bills[var])
    print(f'p-value для {var}: {p_value}')

# Выбор типа регрессии и построение предсказания
X = df[important_vars]
y = df['is_genuine']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print('Accuracy:', accuracy_score(y_test, y_pred))
print('Classification Report:\n', classification_report(y_test, y_pred))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
