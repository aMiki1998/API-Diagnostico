from fastapi import FastAPI

from pydantic import BaseModel

import pandas as pd

from sklearn import tree 
from sklearn.model_selection import train_test_split


class diagnostico(BaseModel):
    tos: bool
    cefalea: bool
    congestion_nasal: bool						
    dificultad_respiratoria: bool
    dolor_garganta: bool
    fiebre: bool
    diarrea: bool
    nauseas: bool
    anosmia_hiposmia: bool
    dolor_abdominal: bool
    dolor_articulaciones: bool
    dolor_muscular: bool
    dolor_pecho: bool
    otros_sintomas: bool

app = FastAPI(title="API de diagnóstico de covid 19",
            description="test",
            version="1.0.0")

data = pd.read_excel ('Dataset_No_Dupliclates_200000.xlsx')

X = pd.DataFrame(data, columns= ['tos','cefalea','congestion_nasal','dificultad_respiratoria','dolor_garganta','fiebre','diarrea','nauseas','anosmia_hiposmia','dolor_abdominal','dolor_articulaciones','dolor_muscular','dolor_pecho','otros_sintomas'])
y = pd.DataFrame(data, columns= ['Flag_sospechoso'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

classifier = tree.DecisionTreeClassifier(max_depth = 15,
                                         class_weight={0:2.77})

classifier.fit(X_train, y_train)

@app.get('/')
async def index():
    return 'tests'

@app.post('/Diagnostico')
async def diagnostico(diagnostico:diagnostico):
    res = classifier.predict([[
        diagnostico.tos,
        diagnostico.cefalea,
        diagnostico.congestion_nasal,
        diagnostico.dificultad_respiratoria,
        diagnostico.dolor_garganta,
        diagnostico.fiebre,
        diagnostico.diarrea,
        diagnostico.nauseas,
        diagnostico.anosmia_hiposmia,
        diagnostico.dolor_abdominal,
        diagnostico.dolor_articulaciones,
        diagnostico.dolor_muscular,
        diagnostico.dolor_pecho,
        diagnostico.otros_sintomas]])
    return str(res[0])