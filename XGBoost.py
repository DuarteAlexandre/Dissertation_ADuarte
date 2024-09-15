import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from xgboost import plot_importance
import seaborn as sns
from scipy import stats
from scipy.stats import norm
import numpy as np
from scipy.stats import f
import xgboost as xgb
from statsmodels.stats.diagnostic import het_breuschpagan

data = pd.read_excel(file_path)
X = data[['Energy_Estimated(%)', 'Distance_Traveled(km)', 'Avg_Temperature(C)', 'Air_Humidity(%)']]
y = data['Real_Energy_Expenditure(%)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGB
xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)

# CRMSE
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
print(f'Root Mean Squared Error (RMSE): {rmse}')

# Grid Search
xgb_model = XGBRegressor(objective='reg:squarederror', random_state=42)
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.15, 0.2],
    'max_depth': [3, 5, 7, 9],
    'min_child_weight': [1, 3, 5, 7, 9, 11],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.85, 0.9, 0.95, 1.0]
}
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='neg_mean_absolute_percentage_error', cv=5, verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)
print("Best hyperparameters:", grid_search.best_params_)
print("Best performance:", -grid_search.best_score_)
best_model = grid_search.best_estimator_

# Feature importance
plt.figure(figsize=(10, 6))
plt.barh(X.columns, best_model.feature_importances_)
plt.xlabel('Importância')
plt.ylabel('Variável')
plt.title('Importância das Variáveis')
plt.show()
best_model.feature_importances_

# Save best model
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)
best_model.save_model('xgboost_bestmodel.json')

# MAPE, MSE and R^2
y_pred = best_model.predict(X_test)
mape = mean_absolute_percentage_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nDesempenho do Modelo:")
print(f"MAPE: {mape * 100:.2f}%")
print(f"MSE: {mse:.4f}")
print(f"R²: {r2:.4f}")

# Load model
modelo_carregado = xgb.XGBRegressor()
modelo_carregado.load_model('xgboost_bestmodel.json')

# Residue
teste_y_pred = modelo_carregado.predict(X_test)
residuos = [y_t - y_p for y_t, y_p in zip(y_test, teste_y_pred)]
residuos = np.array(residuos, dtype=np.float64)
residuos = pd.Series(residuos)

# Residue scatter 
media_residuos = np.mean(residuos)
plt.scatter(range(len(residuos)), residuos, color='blue', alpha=0.7, label='Residues - Hybrid XGB model #3')
plt.axhline(y=media_residuos, color='red', linestyle='--', label='Residues avg.')
plt.xlabel('Point index')
plt.ylabel('Residues - Hybrid XGB model #3')
plt.title('Residue scatter plot - Hybrid XGB model #3')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.5)
plt.savefig('scatter_plot_XGB.png', dpi=300, bbox_inches='tight')
plt.show()

# Residue density plot
mean = np.mean(residuos)
std = np.std(residuos)
sns.kdeplot(residuos, fill=True, color='blue', label='Density Plot - Hybrid XGB model #3', bw_adjust=0.5)
x_values = np.linspace(min(residuos), max(residuos), 100)
plt.plot(x_values, norm.pdf(x_values, mean, std), color='red', linestyle='dashed', label='Normal distribution')
plt.title('Residues - density plot - Hybrid XGB model #3')
plt.xlabel('Residues')
plt.ylabel('Density')
plt.legend(loc='upper left')
plt.grid(True, alpha=0.5)
plt.savefig('density_plot_XGB.png', dpi=300, bbox_inches='tight')
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
SSR = np.sum((teste_y_pred - y_test_mean) ** 2)
SSE = np.sum((y_test - teste_y_pred) ** 2)
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


# Predicted vs Actual values
sns.set(style="whitegrid")
plt.figure(figsize=(10,6))
plt.scatter(y_test, teste_y_pred, color='blue', edgecolors='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
plt.xlabel('Predicted energy consumption ($y_{pred}$)')
plt.ylabel('Actual energy consumption ($y_{test}$)')
plt.title('Scatter plot - $y_{pred}$ vs $y_{test}$ - Hybrid XGB Model')
plt.savefig('pred_XGB.png', dpi=300, bbox_inches='tight')
plt.show()

