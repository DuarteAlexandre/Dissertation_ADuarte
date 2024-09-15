import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import statsmodels.api as sm
import scipy.stats as stats
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Input
from sklearn.metrics import accuracy_score
from tensorflow.keras.metrics import MeanSquaredError
from tensorflow.keras.losses import MeanAbsolutePercentageError
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import r2_score
import seaborn as sns
from scipy.stats import f
from scipy.stats import norm
from tensorflow.keras.models import load_model
from statsmodels.stats.diagnostic import het_breuschpagan


data = pd.read_excel(file_path)

X = data[['Energy_Estimated(%)', 'Distance_Traveled(km)', 'Avg_Temperature(C)', 'Rain(mm)']]
y = data['Real_Energy_Expenditure(%)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Arquitetura
model = Sequential([
    Input(shape=(X.shape[1],)),
    Dense(64, activation='relu'),  
    Dense(128, activation='relu'),                    
    Dense(64, activation='relu'),                    
    Dense(1, activation='linear')                  
])

learning_rate = 0.001
optimizer = Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss=MeanAbsolutePercentageError())

# Loss function
history = model.fit(X_train, y_train, epochs=200, batch_size=16, verbose=1) 
plt.plot(history.history['loss'])
plt.title('Decaimento do Loss durante o Treinamento')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

loss = model.evaluate(X_test, y_test)
loss


predictions = model.predict(X_test)
y_pred = pd.DataFrame(predictions)
y_test = pd.DataFrame(y_test)
y_pred.index = y_test.index
df_combined = pd.concat([y_pred,y_test], axis=1)
df_combined.columns = ['y_pred', 'y_test']
df_combined['dif'] = (df_combined['y_pred'] - df_combined['y_test'])

# MAPE and R^2
mape = mean_absolute_percentage_error(df_combined['y_test'], df_combined['y_pred'])
print("Mean Absolute Percentage Error (MAPE):", mape*100)
r2 = r2_score(df_combined['y_test'], df_combined['y_pred'])
print('R² =', r2)


model.save('ANN.keras')

# Import model
model = load_model('ANN.keras')

X = data[['Energy_Estimated(%)', 'Distance_Traveled(km)', 'Avg_Temperature(C)']]
y = data['Real_Energy_Expenditure(%)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

verificacao = model.predict(X_test)

# MAPE and R^2
mape = mean_absolute_percentage_error(y_test, verificacao)
print("Mean Absolute Percentage Error (MAPE):", mape*100)
r2 = r2_score(y_test, verificacao)
print('R² =', r2)

residuos = [y_t - y_p for y_t, y_p in zip(y_test, verificacao)]
residuos = np.array([x[0] for x in residuos], dtype=np.float64)
residuos = pd.Series(residuos)

# Scatter residues
media_residuos = np.mean(residuos)
plt.scatter(range(len(residuos)), residuos, color='blue', alpha=0.7, label='Residues - Hybrid ANN model #6')
plt.axhline(y=media_residuos, color='red', linestyle='--', label='Residues avg.')
plt.xlabel('Point index')
plt.ylabel('Residues - Hybrid ANN model #6')
plt.title('Residue scatter plot - Hybrid ANN model #6')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.5)
plt.savefig('scatter_plot_ANN.png', dpi=300, bbox_inches='tight')
plt.show()

# Density plot residues
mean = np.mean(residuos)
std = np.std(residuos)
sns.kdeplot(residuos, fill=True, color='blue', label='Density Plot - Hybrid ANN model #6', bw_adjust=0.5)
x_values = np.linspace(min(residuos), max(residuos), 100)
plt.plot(x_values, norm.pdf(x_values, mean, std), color='red', linestyle='dashed', label='Normal distribution')
plt.title('Residues - density plot - Hybrid ANN model #6')
plt.xlabel('Residues')
plt.ylabel('Density')
plt.legend(loc='upper left')
plt.grid(True, alpha=0.5)
plt.savefig('density_plot_ANN.png', dpi=300, bbox_inches='tight')
plt.show()

# Shapiro Francia
shapiro_test = stats.shapiro(residuos)
print("Estatística de teste:", shapiro_test[0])
print("Valor-p:", shapiro_test[1])

# Breusch-Pagan/Cook-Weisberg
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
verificacao = verificacao.ravel()
y_test_mean = np.mean(y_test)
SSR = np.sum((verificacao - y_test_mean) ** 2)
SSE = np.sum((y_test - verificacao) ** 2)
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

# Fitted vs Real values
sns.set(style="whitegrid")
plt.figure(figsize=(10,6))
plt.scatter(y_test, verificacao, color='blue', edgecolors='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
plt.xlabel('Predicted energy consumption ($y_{pred}$)')
plt.ylabel('Actual energy consumption ($y_{test}$)')
plt.title('Scatter plot - $y_{pred}$ vs $y_{test}$ - Hybrid ANN Model')
plt.savefig('pred_ANN.png', dpi=300, bbox_inches='tight')
plt.show()
