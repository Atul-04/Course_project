import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from deap import base, creator, tools, algorithms
import random
import pickle

df = pd.read_csv("biofuel_engine_dataset.csv")

df = df[df['Ethanol (%)'] >= 0]
X = df[['Diesel (%)', 'Biodiesel (%)', 'Ethanol (%)', 'RPM']]  
y_bte = df['BTE (%)']                                       
y_co2 = df['CO2 (g/kWh)']                                


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_bte_train, y_bte_test = train_test_split(X_scaled, y_bte, test_size=0.3, random_state=42)
_, _, y_co2_train, y_co2_test = train_test_split(X_scaled, y_co2, test_size=0.3, random_state=42)

bte_model = GridSearchCV(RandomForestRegressor(), param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}, cv=3)
bte_model.fit(X_train, y_bte_train)

co2_model = GridSearchCV(RandomForestRegressor(), param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}, cv=3)
co2_model.fit(X_train, y_co2_train)

print("BTE RMSE:", np.sqrt(mean_squared_error(y_bte_test, bte_model.predict(X_test))))
print("CO2 RMSE:", np.sqrt(mean_squared_error(y_co2_test, co2_model.predict(X_test))))

pickle.dump(bte_model, open("bte_model.pkl", "wb"))
pickle.dump(co2_model, open("co2_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))  
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, 0, 100)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, 4)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluate(individual):
    total = sum(individual[:3])
    if total > 100:  # Ensure valid blend
        return -999, 999
    cols = ['Diesel (%)', 'Biodiesel (%)', 'Ethanol (%)', 'RPM']
    X = pd.DataFrame([individual], columns=cols)
    X_scaled = scaler.transform(X)
    bte = bte_model.predict(X_scaled)[0]
    co2 = co2_model.predict(X_scaled)[0]
    return bte, co2

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=0, up=100, eta=20.0)
toolbox.register("mutate", tools.mutPolynomialBounded, low=0, up=100, eta=20.0, indpb=0.2)
toolbox.register("select", tools.selNSGA2)
toolbox.register("map", map)

pop = toolbox.population(n=100)
algorithms.eaMuPlusLambda(pop, toolbox, mu=100, lambda_=200, cxpb=0.7, mutpb=0.3, ngen=40, verbose=True)

pareto_front = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]

pareto_sorted = sorted(pareto_front, key=lambda x: x.fitness.values[1])
print("Top 3 Pareto-optimal blends:")
for i, ind in enumerate(pareto_sorted[:3]):
    print(f"{i+1}: Blend = {ind}, BTE = {ind.fitness.values[0]:.2f}%, CO2 = {ind.fitness.values[1]:.2f} g/kWh")
