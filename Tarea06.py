"""
Tarea 06 - Market Basket Analysis (Bread Basket Dataset)
Adaptado para ejecutarse como script en Visual Studio Code (no notebook).

El dataset consta de 21293 observaciones de una panadería. Contiene cuatro
variables: Date, Time, Transaction ID e Item. El Transaction ID va de 1 a
9684, aunque hay números saltados y entradas duplicadas. La columna Item
contiene "Adjustment", "NONE" y "Afternoon with the baker"; las dos primeras
no son compras reales, mientras que la tercera sí podría serlo.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# -----------------------------------------------------------------------
# 1. Leer y cargar el dataset desde disco
# -----------------------------------------------------------------------
bread_basket_data = pd.read_csv(r'C:/Users/PC/Desktop/BreadBasket_DMS.csv')

print("\n--- Primeras 10 filas ---")
print(bread_basket_data.head(10))

print("\n--- Forma del dataset ---")
print(bread_basket_data.shape)

# -----------------------------------------------------------------------
# 2. Revisar y eliminar valores 'NONE'
# -----------------------------------------------------------------------
print("\n--- Conteo de valores 'NONE' ---")
print(bread_basket_data.loc[bread_basket_data['Item'] == 'NONE'].count())

bread_basket_data = bread_basket_data.drop(
    bread_basket_data.loc[bread_basket_data['Item'] == 'NONE'].index)

print("\n--- Conteo de 'NONE' tras eliminarlos (debe ser 0) ---")
print(bread_basket_data.loc[bread_basket_data['Item'] == 'NONE'].count())

print("\n--- Forma del dataset tras eliminar 'NONE' ---")
print(bread_basket_data.shape)

print("\n--- Valores únicos en Item ---")
print(bread_basket_data['Item'].unique())

# -----------------------------------------------------------------------
# 3. Calcular frecuencia de cada producto
# -----------------------------------------------------------------------
basket_items = {}

for item in bread_basket_data['Item']:
    if item in basket_items:
        basket_items[item] = basket_items[item] + 1
    else:
        basket_items[item] = 1

item_names = []
item_frequencies = []

for key, val in basket_items.items():
    item_names.append(key)
    item_frequencies.append(val)

items_table = pd.DataFrame({'Names': item_names,
                             'Frequencies': item_frequencies})

print("\n--- Tabla de frecuencias de items ---")
print(items_table)

# -----------------------------------------------------------------------
# 4. Graficar productos vs frecuencia
# -----------------------------------------------------------------------
print("\n--- Top 5 productos más frecuentes ---")
print(items_table.sort_values('Frequencies', ascending=False).head())

items_table.plot.bar(y='Frequencies', x='Names', figsize=(12, 8))
plt.title("Frecuencia de todos los productos")
plt.tight_layout()
plt.show()

items_table.sort_values('Frequencies', ascending=False).head(20).\
    plot.bar(y='Frequencies', x='Names', figsize=(12, 8))
plt.title("Top 20 productos más frecuentes")
plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------
# 5. Describir el dataset
# -----------------------------------------------------------------------
print("\n--- Descripción de la columna Transaction ---")
print(bread_basket_data['Transaction'].describe())

# -----------------------------------------------------------------------
# 6. Agrupar items por número de transacción
# -----------------------------------------------------------------------
bread_basket_data = bread_basket_data.groupby('Transaction').agg(
    lambda x: ','.join(x)).reset_index()

bread_basket_data = bread_basket_data.drop(['Date', 'Time'], axis=1)

print("\n--- Transacciones agrupadas (primeras 10) ---")
print(bread_basket_data.head(10))

print("\n--- Forma tras agrupar ---")
print(bread_basket_data.shape)

# -----------------------------------------------------------------------
# 7. Preparar conjuntos de items para Apriori
# -----------------------------------------------------------------------
items_data = bread_basket_data['Item']

print("\n--- Items por transacción (primeras 10) ---")
print(items_data.head(10))

items_list = [item.split(',') for item in items_data]

items_list_df = pd.DataFrame({'Items': items_list})

print("\n--- Items convertidos en listas (primeras 10) ---")
print(items_list_df.head(10))

# -----------------------------------------------------------------------
# 8. Minería de reglas de asociación
# -----------------------------------------------------------------------
transencoder = TransactionEncoder()
transencoder_array = transencoder.fit(items_list).transform(items_list)

encoded_df = pd.DataFrame(transencoder_array, columns=transencoder.columns_)

print("\n--- Dataset codificado (One-Hot) ---")
print(encoded_df.head(10))

item_support_df = apriori(encoded_df, min_support=0.01, use_colnames=True)

print("\n--- Forma del set de items frecuentes ---")
print(item_support_df.shape)

print("\n--- Muestra de items frecuentes ---")
print(item_support_df.sample(10))

rules = association_rules(item_support_df,
                           metric='confidence', min_threshold=0.1)

print("\n--- Forma de las reglas de asociación ---")
print(rules.shape)

print("\n--- Muestra de reglas ---")
print(rules.sample(10))

print("\n--- Top 10 reglas por confianza ---")
print(rules.sort_values('confidence', ascending=False).head(10))

rules["antecedent_len"] = rules["antecedents"].apply(lambda x: len(x))

print("\n--- Muestra de reglas con longitud de antecedente ---")
print(rules.sample(5))

# -----------------------------------------------------------------------
# 9. Consultas específicas de reglas
# -----------------------------------------------------------------------
print("\n--- Reglas donde el antecedente es 'Juice' ---")
print(rules[rules['antecedents'] == {'Juice'}])

print("\n--- Reglas donde el antecedente es 'Cake' ---")
print(rules[rules['antecedents'] == {'Cake'}])

print("\n--- Reglas donde el antecedente es 'Sandwich' ---")
print(rules[rules['antecedents'] == {'Sandwich'}])