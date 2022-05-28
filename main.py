from fastapi import FastAPI

app = FastAPI(title="API de diagnóstico de covid 19",
           description="test",
           version="1.0.0")

@app.get("/inicio")
async def ruta_prueba():
    return "Hola mundo 1"

    