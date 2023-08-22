import streamlit as st
import datetime
import pandas as pd
import numpy as np

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from numpy import sqrt
modelo = XGBRegressor()
modelo.load_model('XGBoost1.json')
modelo1 = XGBRegressor()
modelo1.load_model('WVXGBoost.json')

st.title('Predicción de consumo de energía')
st.divider()

# Crear un rango de fechas con una frecuencia de 15 segundos para el 19 de mayo de 2023
fecha_inicio = st.date_input("Ingrese fecha de inicio", datetime.date(2023, 5, 19))
fecha_fin = st.date_input("Ingrese fecha fin", datetime.date(2023, 5, 20))

st.write(str(fecha_inicio) + " hasta " + str(fecha_fin))

indice_fecha = pd.date_range(start=fecha_inicio, end=fecha_fin, freq='15S')

# Crear un DataFrame usando este rango de fechas como índice y con una columna ficticia
dataset = pd.DataFrame(index=indice_fecha)
dataset['Hora'] = dataset.index.hour
dataset['Día de Semana'] = dataset.index.dayofweek
dataset['Mes'] = dataset.index.month
dataset['Día del Año'] = dataset.index.dayofyear
dataset['Día del Mes'] = dataset.index.day
dataset['Semana del Año'] = dataset.index.isocalendar().week.astype(int)
dataset['Es Fin de Semana'] = (dataset['Día de Semana'] >= 5).astype(int)
dataset.index.tz_localize(None)

if st.checkbox('Mostrar tabla'):
    st.write(dataset)

caracteristicas = ['Hora', 'Día de Semana', 'Mes', 'Día del Año', 'Día del Mes', 'Semana del Año', 'Es Fin de Semana']

dataset['Predicción'] = modelo.predict(dataset[caracteristicas])
dataset['Predicción1'] = modelo1.predict(dataset[caracteristicas])

predicciones = dataset[['Predicción','Predicción1']]

# Crear el gráfico
st.line_chart(dataset, x=None, y=['Predicción','Predicción1'], width=0, height=0, use_container_width=True)
st.divider()
if st.checkbox('Mostrar Predicciones'):
    st.write(predicciones)