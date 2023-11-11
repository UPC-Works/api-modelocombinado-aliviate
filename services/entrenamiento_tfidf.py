import pandas as pd
import numpy as np
import seaborn as sns
import pickle
import redis
import matplotlib.pyplot as plt
from sklearn.preprocessing import  LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer



class TrainTFIDFService:

    def entrenamiento_tfidf():
        
        print("PASO 2.1 -------------> 1.0")
                
        #Leer archivo excel
        df_01 = pd.read_excel('./files/BD_HISTORIAS_CLINICAS_ENTRENAMIENTO.xlsx')
        
        print("PASO 2.1 -------------> 2.0")
        
        #Leer archivo excel
        df_02 = pd.read_excel('./files/BD_SIGNOS_SINTOMAS.xlsx')
        
        print("PASO 2.1 -------------> 3.0")
        
        # Hacemos una segunda copia de df_01
        df_01_copy2 = df_01.copy()
        # Hacemos una segunda copia de df_02
        df_02_copy = df_02.copy()
        
        print("PASO 2.1 -------------> 4.0")

        # Transformamos los datos y los preparamos de df_01_copy2

        # Eliminamos todos los campos menos ENFERMEDAD y SIGNOS_SINTOMAS
        df_01_copy2 = df_01_copy2[['ENFERMEDAD', 'SIGNOS_SINTOMAS']]
        
        print("PASO 2.1 -------------> 5.0")
        
        # Transformamos los datos y los preparamos de df_02_copy

        # Eliminamos todos los campos menos ENFERMEDAD y SIGNOS_SINTOMAS
        df_02_copy = df_02_copy[['ENFERMEDAD', 'SIGNOS_SINTOMAS']]
        
        print("PASO 2.1 -------------> 6.0")
        
        #Unimos df_02_copy y df_01_copy2
        df_append = pd.concat([df_02_copy,df_01_copy2])
        
        print("PASO 2.1 -------------> 7.0")
        
        # Inicializar el vectorizador TF-IDF
        vectorizer = TfidfVectorizer()
        
        print("PASO 2.1 -------------> 8.0")
        
        # Ajustar y transformar los signos y síntomas en vectores TF-IDF
        tfidf_matrix = vectorizer.fit_transform(df_append['SIGNOS_SINTOMAS'])
        
        print("PASO 2.1 -------------> 9.0")
        
        # Guardar el modelo en un archivo .pkl
        with open('tfidf_matrix.pkl', 'wb') as f:
            pickle.dump(tfidf_matrix, f)
        
        print("PASO 2.1 -------------> 10.0")
        
        # Guardar el modelo en un archivo .pkl
        with open('vectorizer.pkl', 'wb') as f:
            pickle.dump(vectorizer, f)
        
        print("PASO 2.1 -------------> 11.0")
    





















































































