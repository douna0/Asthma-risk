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
            "description":response["weather"][0]["description"],
            "wind_speed":response.get("wind",{}).get("speed",0)
        }
        self.cache[city]={"data":data,"timestamp":now}
        return data    


class WeatherRiskCalculator:
    def calculate_weather_risk(self ,weather_data):
        """
            calculer le facteur de risque basé sur des conditions météréologiques
            regles : 
            -Température basse (<10c ):augmente le risque 
            -L'air froid : augmente 
            -Humidité élévé (>70%):augmente 
            -Humidité trés basse (<30%):augmente 
            -Presence de pluie ou orage : peut augmenter le risque (allergènes)
            -Vent fort : peut augmenter le rique (disperssion des allergènes)
        """
        risk_score=0
        risk_factors=[]
        temp = weather_data["temperature"]
        humidity=weather_data["humidity"]
        description=weather_data["description"].lower()
        wind_speed =weather_data.get("wind_speed",0)

        if temp <10 or (temp <10 and wind_speed>20):
            risk_score+=0.2
            risk_factors.append("Température basse")
        elif temp >30:
            risk_score+=0.1
            risk_factors.append("Température élévée")
        if humidity >70 :
            risk_score+=0.15
            risk_factors.append("Humidité élevée")
        elif humidity <30:
            risk_score+=0.15
            risk_factors.append("Air très sec")
        if "rain" in description or "drizzle" in description:
            risk_score=+0.1
            risk_factors.append("Présence de pluie")
        if "storm" in description or "thunder" in description:
            risk_score+=0.15
            risk_factors.append("Orage")
        if wind_speed>20:
            risk_score+=0.15
            risk_factors.append("vent fort")
        elif wind_speed>10:
            risk_score+=0.05
            risk_factors.append("Vent modéré")
        return{
            "weather_risk_score":risk_score,
            "weather_risk_factors":risk_factors
        }

class AsthmaPredictor:# pour sauvegarder le modele entrainé avec joblib 
    def __init__(self , model_path=None):
        if model_path:
            self.model=joblib.load(model_path)

        else:
            self.model=None

    def prepare_data(self,data_path):
        df=pd.read_csv(data_path)
        X = df.drop(['PatientID', 'Diagnosis', 'DoctorInCharge'], axis=1)
        y = df['Diagnosis']
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
    def predict(self,symptoms):
        if not self.model:
            raise ValueError("Le modèle n'est pas entraîné")
    
        input_df=pd.DataFrame([symptoms])
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
        }

class AsthmaApp:
    def __init__(self,weather_api_key,city):
        self.weather_client=WeatherClient(api_key=weather_api_key)
        self.predictor=AsthmaPredictor()
        self.city=city
        self.weather_risk_calculator=WeatherRiskCalculator()
    def train_model(self , data_path="asthma_disease_data.csv"):
        print("Préparation des données...")
        weather_data = self.weather_client.get_weather(self.city)
        X,y=self.predictor.prepare_data(data_path)
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
            symptoms_prediction=self.predictor.predict(symptoms)
            weather_risk=self.weather_risk_calculator.calculate_weather_risk(weather_data)
            symptoms_probability=symptoms_prediction["probability"]
            weather_probability=weather_risk["weather_risk_score"]
            combined_probability=min(1.0,symptoms_probability*(1+weather_probability))
            if combined_probability > 0.7:
                risk_level = "Élevé"
            elif combined_probability > 0.4:
                risk_level = "Modéré"
            else:
                risk_level = "Faible"
            return {
                "prediction":{
                    "risk": combined_probability > 0.5,
                    "probability": combined_probability,
                    "risk_level": risk_level,
                    "base_medical_risk": symptoms_probability,
                    "weather_risk_contribution": weather_probability,
                    "weather_risk_factors": weather_risk["weather_risk_factors"]
                },
                "weather":weather_data
            }
        except Exception as e :
            return {"error":str(e)}

# Remplace par ta vraie clé OpenWeatherMap ici
WEATHER_API_KEY = "3b45cc93243338ae5715503217d0560d"
def main():
    # Création de l'application avec la clé météo
    app = AsthmaApp(WEATHER_API_KEY, "Avignon")

    # Étape 1 : Entraîner le modèle (uniquement avec les données médicales)
    print("=== Entraînement du modèle ===")
    accuracy = app.train_model(data_path="asthma_disease_data.csv")
    print(f"✅ Entraînement terminé avec une précision de {accuracy:.2f}\n")

    # Étape 2 : Charger le modèle (pas obligatoire ici, car déjà entraîné)
    app.load_model()

    # Étape 3 : Symptômes simulés
    example_symptoms = {
        "Age": 45,
        "Gender": 1,
        "Ethnicity": 2,
        "EducationLevel": 1,
        "BMI": 22.5,
        "Smoking": 0,
        "PhysicalActivity": 0.7,
        "DietQuality": 5.0,
        "SleepQuality": 6.5,
        "PollutionExposure": 7.2,
        "PollenExposure": 3.0,
        "DustExposure": 1.0,
        "PetAllergy": 0,
        "FamilyHistoryAsthma": 1,
        "HistoryOfAllergies": 1,
        "Eczema": 0,
        "HayFever": 1,
        "GastroesophagealReflux": 0,
        "LungFunctionFEV1": 2.0,
        "LungFunctionFVC": 4.5,
        "Wheezing": 1,
        "ShortnessOfBreath": 1,
        "ChestTightness": 0,
        "Coughing": 1,
        "NighttimeSymptoms": 1,
        "ExerciseInduced": 0
    }

    # Étape 4 : Faire une prédiction avec intégration de la météo
    print("=== Prédiction d'un risque d'asthme ===")
    city = "Avignon"
    result = app.predict_asthma_risk(example_symptoms)

    if "error" in result:
        print(" Erreur :", result["error"])
    else:
        print(f"Météo à {city}:")
        print(f"  Température: {result['weather']['temperature']}°C")
        print(f"  Humidité: {result['weather']['humidity']}%")
        print(f"  Conditions: {result['weather']['description']}")
        print("\nRésultat de la prédiction:")
        print(f"  Risque d'asthme: {'Positif' if result['prediction']['risk'] else 'Négatif'}")
        print(f"  Niveau de risque: {result['prediction']['risk_level']}")
        print(f"  Probabilité: {result['prediction']['probability']:.2f}")
        
        # Afficher les facteurs météo identifiés
        if result['prediction']['weather_risk_factors']:
            print("\nFacteurs météo aggravants identifiés:")
            for factor in result['prediction']['weather_risk_factors']:
                print(f"  • {factor}")
    

if __name__ == "__main__":
    main()