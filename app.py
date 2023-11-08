from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from joblib import load
import pathlib
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

origins = [ "*" ]

app = FastAPI(title = 'prediccion de uso de pistolas respecto a horas de juego')

app.add_middleware(
CORSMiddleware,
allow_origins=["*"],
allow_credentials=True,
allow_methods=["*"],
allow_headers=["*"],
)
model = load(pathlib.Path('./model/lidfordid-disease-v1.joblib'))

class InputData(BaseModel):
   Username :int=4 ,
   Pistol_Shots:int=1,
   Pistol_Kills:int=1,
   Pistol_Usage:float=8.342,
   Magnum_Shots :int=1,
   Magnum_Kills :int=1,
   Magnum_Usage :float=6.234,
   Uzi_Shots :int=1,
   Uzi_Kills :int=1,
   Uzi_Usage :float=5.234234,
   Silenced_SMG_Shots :int=1,
   Silenced_SMG_Kills :int=1,
   Silenced_SMG_Usage :int=1,
   MP5_Shots:int=1,
   MP5_Kills :int=1,
   MP5_Usage :int=1,
   Pump_Shotgun_Shots :int=1,
   Pump_Shotgun_Kills :int=1,
   Pump_Shotgun_Usage :int=1,
   Chrome_Shotgun_Shots :int=1,
   Chrome_Shotgun_Kills :int=1,
   Chrome_Shotgun_Usage :int=1,
   Tactical_Shotgun_Shots :int=1,
   Tactical_Shotgun_Kills :int=1,
   Tactical_Shotgun_Usage :int=1,
   Combat_Shotgun_Shots :int=1,
   Combat_Shotgun_Kills :int=1,
   Combat_Shotgun_Usage :int=1,
   Assault_Rifle_Shots :int=1,
   Assault_Rifle_Kills :int=1,
   Assault_Rifle_Usage :int=1,
   Desert_Rifle_Shots :int=1,
   Desert_Rifle_Kills :int=1,
   Desert_Rifle_Usage :int=1,
   AK_47_Shots :int=1,
   AK_47_Kills :int=1,
   AK_47_Usage :int=1,
   SG_552_Shots :int=1,
   SG_552_Kills :int=1,
   SG_552_Usage :int=1,
   Hunting_Rifle_Shots :int=1,
   Hunting_Rifle_Kills :int=1,
   Hunting_Rifle_Usage :int=1,
   Military_Sniper_Rifle_Shots :int=1,
   Military_Sniper_Rifle_Kills :int=1,
   Military_Sniper_Rifle_Usage :int=1,
   AWP_Shots :int=1,
   AWP_Kills :int=1,
   AWP_Usage :int=1,
   Scout_Shots :int=1,
   Scout_Kills:int=1,
   Scout_Usage :int=1,
   Grenade_Launcher_Shots :int=1,
   Grenade_Launcher_Kills :int=1,
   Grenade_Launcher_Usage :int=1,
   M60_Shots :int=1,
   M60_Kills :int=1,
   M60_Usage :int=1,
   Minigun_Shots:int=1,
   Minigun_Kills :int=1,
   Minigun_Usage :int=1,
   Baseball_Bat_Shots :int=1,
   Baseball_Bat_Kills :int=1,
   Baseball_Bat_Usage :int=1,
   Chainsaw_Shots :int=1,
   Chainsaw_Kills :int=1,
   Chainsaw_Usage :int=1,
   Cricket_Bat_Shots :int=1,
   Cricket_Bat_Kills :int=1,
   Cricket_Bat_Usage :int=1,
   Crowbar_Shots :int=1,
   Crowbar_Kills :int=1,
   Crowbar_Usage :int=1,
   Electric_Guitar_Shots :int=1,
   Electric_Guitar_Kills :int=1,
   Electric_Guitar_Usage :int=1,
   Fire_Axe_Shots :int=1,
   Fire_Axe_Kills :int=1,
   Fire_Axe_Usage :int=1,
   Frying_Pan_Shots :int=1,
   Frying_Pan_Kills :int=1,
   Frying_Pan_Usage :int=1,
   Katana_Shots :int=1,
   Katana_Kills :int=1,
   Katana_Usage :int=1,
   Machete_Shots :int=1,
   Machete_Kills :int=1,
   Machete_Usage :int=1,
   Tonfa_Shots :int=1,
   Tonfa_Kills :int=1,
   Tonfa_Usage :int=1,
   Golf_Club_Shots :int=1,
   Golf_Club_Kills :int=1,
   Golf_Club_Usage :int=1,
   Pitchfork_Shots :int=1,
   Pitchfork_Kills :int=1,
   Pitchfork_Usage :int=1,
   Shovel_Shots :int=1,
   Shovel_Kills :int=1,
   Shovel_Usage :int=1,
   Knife_Shots :int=1,
   Knife_Kills :int=1,
   Knife_Usage :int=1,
   Molotovs_Thrown :int=1,
   Molotov_Kills :int=1,
   Pipe_Bombs_Thrown :int=1,
   Pipe_Bomb_Kills :int=1,
   Bile_Jars_Thrown :int=1,
   Bile_Jar_Hits :int=1,
   Most_Friendly_Fire :int=1,
   Difficulty:int=1,
   Average_Friendly_Fire :int=1,

class OutputData(BaseModel):
    score:float

@app.post('/score', response_model = OutputData)
def score(data:InputData):
    df = pd.DataFrame(data.dict(), index=[0])

    # Realiza la predicción utilizando el modelo
    model_input = df.values  # Convierte el DataFrame en un array NumPy
    result = model.predict(model_input)

    return {'score':result}
