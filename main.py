from fastapi import FastAPI

from pydantic import BaseModel

import pandas as pd
from sklearn import tree

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

data = pd.read_excel ('Datasetpequeño.xlsx')
df = pd.DataFrame(data, columns= ['tos','cefalea','congestion_nasal','dificultad_respiratoria','dolor_garganta','fiebre','diarrea','nauseas','anosmia_hiposmia','dolor_abdominal','dolor_articulaciones','dolor_muscular','dolor_pecho','otros_sintomas'])
labels = pd.DataFrame(data, columns= ['Flag_sospechoso'])
classifier = tree.DecisionTreeClassifier()
classifier.fit(df,labels)

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