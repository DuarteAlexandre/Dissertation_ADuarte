import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy.stats import norm
import joblib

def mean_absolute_percentage_error(y_true, y_pred):
    nonzero_mask = y_true != 0
    return np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100

def mape_scorer(estimator, X, y):
    y_pred = estimator.predict(X)
    return mean_absolute_percentage_error(y, y_pred)

mape_scorer = make_scorer(mean_absolute_percentage_error, greater_is_better=False)

data = pd.read_excel(file_path)

X = data[['Energy_Estimated(%)', 'Distance_Traveled(km)']]
y = data['Real_Energy_Expenditure(%)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forrest
model = RandomForestRegressor(random_state=42)


# Grid Search
param_grid = {
    'n_estimators': [50, 75, 100, 125, 150, 175, 200],
    'max_depth': [None, 5, 10, 15, 20],
    'min_samples_split': [2, 3, 4, 5],
    'min_samples_leaf': [2, 3, 4, 5, 6, 7, 8],
    'max_features': [0.1, 0.15, 0.2, 0.25, 0.5], 
    'bootstrap': [True]
}
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring=mape_scorer, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)
print("Best hyperparameters:", grid_search.best_params_)
best_model = grid_search.best_estimator_

# k-folds
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_results = cross_val_score(best_model, X, y, cv=kf, scoring=mape_scorer)
print(f'MAPE por Fold (validação cruzada): {-cv_results}')
print(f'Média do MAPE (validação cruzada): {-cv_results.mean()}')
print(f'Desvio padrão do MAPE (validação cruzada): {cv_results.std()}')

# Best model
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

# MAPE and R^2
test_mape = mean_absolute_percentage_error(y_test, y_pred)
test_r2 = r2_score(y_test, y_pred)
print(f'MAPE no conjunto de teste: {test_mape:.2f}%')
print(f'R² no conjunto de teste: {test_r2:.2f}')

# Load model
joblib.dump(best_model, 'best_model_random_forest.pkl')
importances = best_model.feature_importances_

# Feature importance
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)
print(importance_df)
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.xlabel('Importância')
plt.title('Importância das Variáveis')
plt.gca().invert_yaxis()
plt.show()

# Residue
residuos = [y_t - y_p for y_t, y_p in zip(y_test, y_pred)]
residuos = np.array(residuos, dtype=np.float64)
residuos = pd.Series(residuos)

# Residue scatter
media_residuos = np.mean(residuos)
plt.scatter(range(len(residuos)), residuos, color='blue', alpha=0.7, label='Residues - Hybrid RF model #5')
plt.axhline(y=media_residuos, color='red', linestyle='--', label='Residues avg.')
plt.xlabel('Point index')
plt.ylabel('Residues - Hybrid RF model #5')
plt.title('Residue scatter plot - Hybrid RF model #5')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.5)
plt.savefig('scatter_plot_RF.png', dpi=300, bbox_inches='tight')
plt.show()

# Residue density plot
mean = np.mean(residuos)
std = np.std(residuos)
sns.kdeplot(residuos, fill=True, color='blue', label='Density Plot - Hybrid RF model #5', bw_adjust=0.5)
x_values = np.linspace(min(residuos), max(residuos), 100)
plt.plot(x_values, norm.pdf(x_values, mean, std), color='red', linestyle='dashed', label='Normal distribution')
plt.title('Residues - density plot - Hybrid RF model #5')
plt.xlabel('Residues')
plt.ylabel('Density')
plt.legend(loc='upper left')
plt.grid(True, alpha=0.5)
plt.savefig('density_plot_RF.png', dpi=300, bbox_inches='tight')
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
nvars = 2
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

#Predicted vs actual values
sns.set(style="whitegrid")
plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred, color='blue', edgecolors='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
plt.xlabel('Predicted energy consumption ($y_{pred}$)')
plt.ylabel('Actual energy consumption ($y_{test}$)')
plt.title('Scatter plot - $y_{pred}$ vs $y_{test}$ - Hybrid RF Model')
plt.savefig('pred_RF.png', dpi=300, bbox_inches='tight')
plt.show()
