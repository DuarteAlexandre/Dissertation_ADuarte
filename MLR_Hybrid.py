import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error
import seaborn as sns
from scipy.stats import norm
from statsmodels.stats.diagnostic import het_breuschpagan

data = pd.read_excel(file_path)

# Correlation Matrix
correlation_matrix = data.drop(columns=['ID_Rota', 'Real_Energy_Expenditure(%)']).corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1, linewidths=0.5, annot_kws={"size": 10})
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.title('Matriz de Correlações')
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# Regression

X = data[['Energy_Estimated(%)','Distance_Traveled(km)', 'Air_Humidity(%)']]
y = data['Real_Energy_Expenditure(%)']
X = sm.add_constant(X)
model = sm.OLS(y, X)
results = model.fit()
residuos = results.resid
print(results.summary())

y_pred = results.predict(X)

residuos = pd.Series(residuos)
residuos.replace([np.inf, -np.inf], np.nan, inplace=True)

residuos = residuos.dropna()

# Scatter residues
media_residuos = np.mean(residuos)
plt.scatter(range(len(residuos)), residuos, color='blue', alpha=0.7, label='Residues - Hybrid MLR')
plt.axhline(y=media_residuos, color='red', linestyle='--', label='Residues avg.')
plt.xlabel('Point index')
plt.ylabel('Residues - Hybrid MLR')
plt.title('Residue scatter plot - Hybrid MLR')
plt.legend()
plt.grid(True, alpha=0.5)
plt.savefig('scatter_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# Density Plot residues
mean = np.mean(residuos)
std = np.std(residuos)
sns.kdeplot(residuos, fill=True, color='blue', label='Density Plot - Hybrid MLR')
x_values = np.linspace(min(residuos), max(residuos), 100)
plt.plot(x_values, norm.pdf(x_values, mean, std), color='red', linestyle='dashed', label='Normal distribution')
plt.title('Residues - density plot - Hybrid MLR')
plt.xlabel('Residues')
plt.ylabel('Density')
plt.legend()
plt.grid(True, alpha=0.5)
plt.savefig('density_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# Statistical tests
df = pd.concat([y_pred, y, residuos], axis=1)
df.columns = ['y_pred', 'y', 'Resíduos']

# Shapiro-Wilk
shapiro_test = stats.shapiro(residuos)
print("Estatística de teste:", shapiro_test[0])
print("Valor-p:", shapiro_test[1])

# Breusch-Pagan/Cook-Weisberg
lm, lm_p_value, fvalue, f_p_value = het_breuschpagan(residuos, X)
print("Estatística LM:", lm)
print("Valor-p LM:", lm_p_value)
print("Estatística F:", fvalue)
print("Valor-p F:", f_p_value)

# Durbin-Watson
durbin_watson_test = sm.stats.stattools.durbin_watson(residuos)
print("Estatística de Durbin-Watson:", durbin_watson_test)
nobs = len(residuos)
nvars = 3
d_critico = sm.stats.stattools.durbin_watson(residuos)
DL = 1.521 - 1.25 * (nvars / nobs)
DU = 1.521 + 1.25 * (nvars / nobs)
quatro_DL = 4.0 - DL
quatro_DU = 4.0 - DU
print("DL:", DL)
print("DU:", DU)
print("4-DL:", quatro_DL)
print("4-DU:", quatro_DU)

# MAPE
mape = np.mean(np.abs((y - y_pred) / y)) * 100
print(f"MAPE: {mape:.2f}%")
