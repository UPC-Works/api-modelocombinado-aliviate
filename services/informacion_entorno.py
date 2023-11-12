import platform
import psutil
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 
import tensorflow as tf
import sys
import json

class EnvironmentService:

    def informacion():
        data = {
            "Sistema operativo": platform.platform(),
            "CPU": platform.processor(), 
            "Núcleos de CPU físicos": psutil.cpu_count(logical=False),
            "Núcleos de CPU totales": psutil.cpu_count(logical=True),
            "Memoria RAM total": round(psutil.virtual_memory().total / (1024 ** 3), 2),
            "Memoria RAM disponible": round(psutil.virtual_memory().available / (1024 ** 3), 2),
            "Version de python": sys.version
        }
        
        return {"data":data}