import requests
import time 

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

client = WeatherClient(api_key="3b45cc93243338ae5715503217d0560d")
print(client.get_weather("Paris"))