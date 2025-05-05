import joblib
import numpy as np
import pandas as pd
import requests
import time

from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split 

class WeatherClient:
    def __init__(self , api_key, cache_duration=600):
        self.api_key=api_key
        self.cache_duration=cache_duration
        self.cache={}# stocker infos meteo + le temps 
    def get_weather(self,city):
        now=time.time()
        if city in self.cache:
            cached=self.cache[city]
            if now -cached["timestamp"]<self.cache_duration:
                print(f"✅ Données en cache pour {city}")
                return cached["data"]
            
        print(f"🌐 Requête API pour {city}")
        url=f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={self.api_key}&units=metric"
        response=requests.get(url).json()
        if "main" not in response:
            raise ValueError(f"Erreur API :{response}")
        data={
            "temperature":response["main"]["temp"],
            "humidity":response["main"]["humidity"],
            "description":response["weather"][0]["description"]
        }
        self.cache[city]={"data":data,"timestamp":now}
        return data    

class AsthmaPredictor:# pour sauvegarder le modele entrainé avec joblib 
    def __init__(self , model_path=None):
        if model_path:
            self.model=joblib.load(model_path)

        else:
            self.model=None

    def prepare_data(self,data_path,weather_data):
        df=pd.read_csv(data_path)
        X=df.drop(['Severity_Mild','Severity_Moderate','Severity_None'],axis=1)
        y=np.where((df['Severity_Mild']==1) | (df['Severity_Moderate']==1),1,0)
        if weather_data:
            weather_features={
                "temperature": weather_data["temperature"],
                "humidity": weather_data["humidity"],
                "is_rainy": 1 if "rain" in weather_data["description"].lower() else 0,
                "is_cloudy":1 if "cloud" in weather_data["description"].lower() else 0
            }
        weather_df=pd.DataFrame([weather_features]*len(df))
        X=pd.concat([X,weather_df],axis=1)
        return X,y

    def train(self,X,y):
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
        self.model=RandomForestClassifier(n_estimators=200,random_state=42)
        self.model.fit(X_train,y_train)
        y_pred=self.model.predict(X_test)
        accuracy=accuracy_score(y_test,y_pred)
        report=classification_report(y_test,y_pred)
        print(f"Précision du modèle: {accuracy:.2f}")
        print("Rapport de classification:")
        print(report)
        return accuracy
    def save_model(self , path="asthma-model.pkl"):
        if self.model:
            joblib.dump(self.model,path)
            print(f"Modèle sauvegardé sous {path}")
        else:
            print("Aucun modèle à sauvegarder")
    def predict(self,symptoms,weather_data):
        if not self.model:
            raise ValueError("Le modèle n'est pas entraîné")
        weather_features={
            "temperature": weather_data["temperature"],
            "humidity": weather_data["humidity"],
        }
        weather_desc=weather_data["description"].lower()
        is_rainy=1 if "rain" in weather_desc or "drizzle" in weather_desc else 0
        is_cloudy=1 if "cloud" in weather_desc or "overcast" in weather_desc else 0
        weather_features["is_rainy"]=is_rainy
        weather_features["is_cloudy"]=is_cloudy
        all_features={**symptoms,**weather_features}
        input_df=pd.DataFrame([all_features])
        expected_columns=20
        current_columns=len(input_df.columns)
        if current_columns < expected_columns:
            for i in range(current_columns,expected_columns):
                input_df[f'feature{i}']=0
        prediction=self.model.predict(input_df)
        probability = self.model.predict_proba(input_df)
        return {
            "risk": bool(prediction[0]),
            "probability": float(probability[0][1]),
            "risk_level": "Élevé" if probability[0][1] > 0.7 else "Modéré" if probability[0][1] > 0.4 else "Faible"
        }

class AsthmaApp:
    def __init__(self,weather_api_key,city):
        self.weather_client=WeatherClient(api_key=weather_api_key)
        self.predictor=AsthmaPredictor()
        self.city=city
    def train_model(self , data_path="asthma-data.csv"):
        print("Préparation des données...")
        weather_data = self.weather_client.get_weather(self.city)
        X,y=self.predictor.prepare_data(data_path,weather_data)
        print("Entraînement du modèle...")
        accuracy=self.predictor.train(X,y)
        print("Sauvegarde du modèle...")
        self.predictor.save_model()
        return accuracy
    def load_model(self,model_path="asthma-model.pkl"):
        self.predictor=AsthmaPredictor(model_path)
    def predict_asthma_risk(self,symptoms):
        try:
            weather_data=self.weather_client.get_weather(self.city)
            prediction=self.predictor.predict(symptoms,weather_data)
            return {
                "prediction":prediction,
                "weather":weather_data
            }
        except Exception as e :
            return {"error":str(e)}



# Remplace par ta vraie clé OpenWeatherMap ici
WEATHER_API_KEY = "3b45cc93243338ae5715503217d0560d"

def main():
    # Création de l'application avec la clé météo
    app = AsthmaApp(WEATHER_API_KEY, "Avignon")

    # Étape 1 : Entraîner le modèle
    print("=== Entraînement du modèle ===")
    accuracy = app.train_model(data_path="asthma-data.csv")
    print(f"✅ Entraînement terminé avec une précision de {accuracy:.2f}\n")

    # Étape 2 : Charger le modèle (pas obligatoire ici, car déjà entraîné)
    app.load_model()

    # Étape 3 : Symptômes simulés (à adapter selon ton CSV)
    # Dans la fonction main(), remplace example_symptoms par :
    example_symptoms = {
    "Tiredness": 1,
    "Dry-Cough": 1,
    "Difficulty-in-Breathing": 1,
    "Sore-Throat": 1,
    "None_Sympton": 0,
    "Pains": 1,
    "Nasal-Congestion": 1,
    "Runny-Nose": 1,
    "None_Experiencing": 0,
    "Age_0-9": 1,
    "Age_10-19": 0,
    "Age_20-24": 0,
    "Age_25-59": 0,
    "Age_60+": 0,
    "Gender_Female": 1,
    "Gender_Male": 0,
    }

    # Étape 4 : Faire une prédiction avec météo
    print("=== Prédiction d'un risque d'asthme ===")
    city = "Avignon"
    result = app.predict_asthma_risk(example_symptoms)

    if "error" in result:
        print("❌ Erreur :", result["error"])
    else:
        print(f"Météo : {result['weather']}")
        print(f"Risque d'asthme : {result['prediction']['risk']}")
        print(f"Niveau de risque : {result['prediction']['risk_level']}")
        print(f"Probabilité : {result['prediction']['probability']:.2f}")

if __name__ == "__main__":
    main()

