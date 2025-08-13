import numpy as np
from sklearn.linear_model import LinearRegression

# No fijamos semilla para que los datos sean diferentes en cada ejecución

n = 100

# Alturas entre 1.5m y 2.0m
alturas = np.random.uniform(1.5, 2.0, n)

# Pesos entre 50kg y 100kg
pesos = np.random.uniform(50, 100, n)

# IMC real
imc_real = pesos / (alturas ** 2)

# Features: peso y altura
X = np.column_stack((pesos, alturas))
y = imc_real

# Crear y entrenar modelo regresión lineal
modelo = LinearRegression()
modelo.fit(X, y)

# Predicción con datos nuevos
peso_nuevo = np.random.uniform(50, 100)
altura_nueva = np.random.uniform(1.5, 2.0)

X_nuevo = np.array([[peso_nuevo, altura_nueva]])
imc_predicho = modelo.predict(X_nuevo)[0]

print(f"Datos nuevos -> Peso: {peso_nuevo:.2f} kg, Altura: {altura_nueva:.2f} m")
print(f"IMC real calculado: {peso_nuevo / altura_nueva**2:.2f}")
print(f"IMC predicho por modelo regresión lineal: {imc_predicho:.2f}")