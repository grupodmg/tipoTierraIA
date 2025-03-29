import pickle
from fastapi import FastAPI, APIRouter
import numpy as np
from interfaces import TierraData

router = APIRouter()

with open("svcPredict.pkl", "rb") as file:
    model = pickle.load(file)

import pandas as pd

@router.post("/predict")
def predict(data: TierraData):
    data = data.model_dump()
    print(data)

    # Extraer valores de entrada
    N = data['N']
    P = data['P']
    K = data['K']
    temperature = data['temperature']
    humidity = data['humidity']
    ph = data['ph']
    rainfall = data['rainfall']

    # Crear un DataFrame con nombres de características
    feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    data_df = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]], columns=feature_names)

    # Realizar la predicción
    prediction = model.predict(data_df)
    print("Prediction:", prediction)

    # Mapeo de etiquetas
    label_map = {
        0: 'rice',
        1: 'maize',
        2: 'chickpea',
        3: 'kidneybeans',
        4: 'pigeonpeas',
        5: 'mothbeans',
        6: 'mungbean',
        7: 'blackgram',
        8: 'lentil',
        9: 'pomegranate',
        10: 'banana',
        11: 'mango',
        12: 'grapes',
        13: 'watermelon',
        14: 'muskmelon',
        15: 'apple',
        16: 'orange',
        17: 'papaya',
        18: 'coconut',
        19: 'cotton',
        20: 'jute',
        21: 'coffee'
    }

    # Manejar predicciones fuera del rango esperado
    label = [label_map.get(p, "unknown") for p in prediction]
    print("Label: ", label)
    return {"message": str(prediction)}

if __name__ == "__main__":
    router.run()