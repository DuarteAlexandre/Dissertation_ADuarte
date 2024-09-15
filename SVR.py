import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy import stats
from scipy.stats import norm
import numpy as np
from scipy.stats import f

data = pd.read_excel(file_path)


X = data[['Energy_Estimated(%)', 'Air_Humidity(%)', 'Distance_Traveled(km)', 'Rain(mm)']]
y = data['Real_Energy_Expenditure(%)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SVR model
model = SVR(kernel='linear')
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# R^2 and MAPE
r2 = r2_score(y_test, y_pred)
print(f"R-squared: {r2}")
mape = mean_absolute_percentage_error(y_test, y_pred)
print(f"MAPE: {mape * 100:.2f}%")


# Grid Search
param_grid = {
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'gamma': ['scale', 'auto'],
    'degree': [2, 3, 4]  # Aplicável apenas ao kernel 'poly'
}
svr = SVR()
grid_search = GridSearchCV(estimator=svr, param_grid=param_grid, cv=5, scoring='neg_mean_absolute_percentage_error', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# Best hyperparameters
best_params = grid_search.best_params_
print("Melhores Hiperparâmetros Encontrados:")
for param, value in best_params.items():
    print(f"{param}: {value}")

# Best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)

# MAPE, MSE and R^2
mape = mean_absolute_percentage_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nDesempenho do Modelo:")
print(f"MAPE: {mape * 100:.2f}%")
print(f"MSE: {mse:.4f}")
print(f"R²: {r2:.4f}")

# Predicted vs actual values 
sns.set(style="whitegrid")
plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred, color='blue', edgecolors='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
plt.xlabel('Predicted energy consumption ($y_{pred}$)')
plt.ylabel('Actual energy consumption ($y_{test}$)')
plt.title('Scatter plot - $y_{pred}$ vs $y_{test}$ - Hybrid SVR Model')
plt.savefig('pred_SVR.png', dpi=300, bbox_inches='tight')
plt.show()

# Residue
residuos = [y_t - y_p for y_t, y_p in zip(y_test, y_pred)]
residuos = np.array(residuos, dtype=np.float64)
residuos = pd.Series(residuos)

# Residue scatter 
media_residuos = np.mean(residuos)
plt.scatter(range(len(residuos)), residuos, color='blue', alpha=0.7, label='Residues - Hybrid SVR model #12')
plt.axhline(y=media_residuos, color='red', linestyle='--', label='Residues avg.')
plt.xlabel('Point index')
plt.ylabel('Residues - Hybrid SVR model #12')
plt.title('Residue scatter plot - Hybrid SVR model #12')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.5)
plt.savefig('scatter_plot_SVR.png', dpi=300, bbox_inches='tight')
plt.show()

# Residue density plot
mean = np.mean(residuos)
std = np.std(residuos)
sns.kdeplot(residuos, fill=True, color='blue', label='Density Plot - Hybrid SVR model #12', bw_adjust=0.5)
x_values = np.linspace(min(residuos), max(residuos), 100)
plt.plot(x_values, norm.pdf(x_values, mean, std), color='red', linestyle='dashed', label='Normal distribution')
plt.title('Residues - density plot - Hybrid SVR model #12')
plt.xlabel('Residues')
plt.ylabel('Density')
plt.legend(loc='upper left')
plt.grid(True, alpha=0.5)
plt.savefig('density_plot_SVR.png', dpi=300, bbox_inches='tight')
plt.show()

# Shapiro Francia
shapiro_test = stats.shapiro(residuos)
print("Estatística de teste:", shapiro_test[0])
print("Valor-p:", shapiro_test[1])

# Breusch-Pagan/Cook-Weisberg
X_test_v2 = sm.add_constant(X_test)
lm, lm_p_value, fvalue, f_p_value = het_breuschpagan(residuos, X_test_v2)
print("Estatística LM:", lm)
print("Valor-p LM:", lm_p_value)
print("Estatística F:", fvalue)
print("Valor-p F:", f_p_value)

# Durbin-Watson
durbin_watson_test = sm.stats.stattools.durbin_watson(residuos)
print("Estatística de Durbin-Watson:", durbin_watson_test)
nobs = len(residuos)
nvars = 4
d_critico = sm.stats.stattools.durbin_watson(residuos)
DL = 1.521 - 1.25 * (nvars / nobs)
DU = 1.521 + 1.25 * (nvars / nobs)
quatro_DL = 4.0 - DL
quatro_DU = 4.0 - DU
print("DL:", DL)
print("DU:", DU)
print("4-DL:", quatro_DL)
print("4-DU:", quatro_DU)

# F test
n = X_test.shape[0]
k = X_test.shape[1]
y_test_mean = np.mean(y_test)
SSR = np.sum((y_pred - y_test_mean) ** 2)
SSE = np.sum((y_test - y_pred) ** 2)
SST = np.sum((y_test - y_test_mean) ** 2)
MSR = SSR / k
MSE = SSE / (n - k - 1)
F_statistic = MSR / MSE
print(f'SSR (Sum of Squares Regression): {SSR}')
print(f'SSE (Sum of Squares Error): {SSE}')
print(f'MSR (Mean Square Regression): {MSR}')
print(f'MSE (Mean Square Error): {MSE}')
print(f'Estatística F: {F_statistic}')
df1 = k  
df2 = n - k - 1  
p_value = f.sf(F_statistic, df1, df2)
print(f'Prob (F-statistic): {p_value}')
