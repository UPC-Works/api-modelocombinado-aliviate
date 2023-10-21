import os
from fastapi import FastAPI, Query, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from services.modelo_combinado import ModeloCombinadoService
from services.informacion_entorno import EnvironmentService
from services.entrenamiento_rf import TrainRFService
from services.entrenamiento_tfidf import TrainTFIDFService
from services.modelo_combinado import ModeloCombinadoService

app = FastAPI()

# Configurar CORS
origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:3000",
    "null",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def read_root():
    return {"Acceso no autorizado"}

@app.get("/informacion-entorno")
async def informacion_entorno():
    return EnvironmentService.informacion()

@app.post("/entrenar-modelo-combinado")
async def entrenar_modelo_combinado():
    TrainRFService.entrenamiento_rf()
    TrainTFIDFService.entrenamiento_tfidf()
    return {"data": "OK"}

@app.post("/subir-archivo")
async def subir_archivo(archivo: UploadFile):
    
    print("MODELO-------------------->0")
    # Asegúrate de que la carpeta exista, si no, créala
    if not os.path.exists("files"):
        os.makedirs("files")
    
    print("MODELO-------------------->0.1")
    # Ruta completa del archivo a guardar
    file_path = os.path.join("files", archivo.filename)
    with open(file_path, "wb") as f:
        f.write(archivo.file.read())
    
    print("MODELO-------------------->0.2")
    return ModeloCombinadoService.modelo_combinado(archivo.filename)
