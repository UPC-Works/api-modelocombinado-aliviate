import pandas as pd
import numpy as np
import seaborn as sns
import pickle
import redis
import matplotlib.pyplot as plt
from sklearn.preprocessing import  LabelEncoder
from sklearn.ensemble import RandomForestClassifier


class TrainRFService:

    def entrenamiento_rf():
        
        print("PASO 1.1 -------------> 1.0")
        
        #Leer archivo excel
        df_01 = pd.read_excel('./files/BD_HISTORIAS_CLINICAS_ENTRENAMIENTO.xlsx')
        
        print("PASO 1.1 -------------> 2.0")
            
        # Hacemos una copia de df_01
        df_01_copy = df_01.copy()
        
        print("PASO 1.1 -------------> 3.0")
        
        # Eliminamos los signos y sintomas
        df_01_copy = df_01_copy.drop('SIGNOS_SINTOMAS', axis=1)
        
        print("PASO 1.1 -------------> 4.0")
        
        # Aplicar la codificación ordinal para transformar variables categóricas a numericas
        encoder = LabelEncoder()
        
        print("PASO 1.1 -------------> 5.0")
        
        # Definir las columnas para aplicar el encoding
        columns_to_encode = ['SEXO','GRUPO_SANGUINEO','GRADO_INSTITUCION','ESTADO_CIVIL','TUVO_TUBERCULOSIS','TIENE_INF_RENAL_GLAUCOMA','TIENE_INF_TRANS_SEX','TIENE_SIDA','TIENE_ITS','TIENE_HEPATITIS','TIENE_DIABETES','TIENE_HTA','TIENE_SOBREPESO','TIENE_DISLIPENIA','TIENE_DEPRESION_ESQUIZOFRENIA','TIENE_HOSPITALIZACION_TRANSFUCIONES','TIENE_INTER_QUIRURJICA','TIENE_PREMATURO','TIENE_ABORTO','TIENE_PARTO','FLUJO_VAG_PATOLOGICO','TIENE_EXAM_PROSTATA','TIENE_VIOLENCIA','TIENE_DBM','TIENE_INFARTO','TIENE_CANCER','TIENE_DEPRESION','TIENE_PROB_PSIQUIATRICOS','RS_MISMO_SEXO','MEDICAMENTO_FRECUENTE','REACCION_MEDICAMENTOS','TIENE_CONSUMO_TABACO','TIENE_CONSUMO_ALCOHOL','EDAD_INI_RELACION_SEXUAL','NUM_PAREJAS','DISMENORREA','TIENE_EMBARAZO','TIENE_FIEBRE_15_DIAS','TIENE_TOS_15_DIAS','TIENE_VAC_ANTITETANICA','TIENE_VAC_ANTIAMERILICA','TIENE_VAC_ANTIHEPATITIS_B','TIENE_ENCIAS','TIENE_CARIES','TIENE_EDENTULISMO_TOTAL','TIENE_ANSIEDAD','TIENE_EDENTULISMO_PARCIAL','TIENE_EXAM_VISUAL','TIENE_URG_TRATAMIENTO_BUCAL','TIENE_MAMOGRAFIA','TIENE_EXAM_COLESTEROL','TIENE_EXAM_MAMAS','TIENE_EXAM_GLUCOSA','TIENE_HAB_FISICA','TIENE_PLANIFICACION_SEXUAL','TIENE_HAB_ALCOHOL','TIENE_HAB_DROGAS']
        df_01_copy[columns_to_encode] = df_01_copy[columns_to_encode].apply(encoder.fit_transform)
        
        print("PASO 1.1 -------------> 6.0")
        
        # Eliminamos el ID, el nombre y las fechas
        columns_to_drop = ['ID_HISTORIA_CLINICA','FECHA_REGISTRO','DIRECCION','FECHA_ULT_REGLA','TIENE_EXAM_PELVICO_PAP','NOMBRES','APELLIDOS','EDAD','FECHA_NACIMIENTO','MENARQUIA','GESTACION','PRESION_ARTERIAL','LESIONES_GENITALES','TEMPERATURA','TALLA']
        df_01_copy = df_01_copy.drop(columns_to_drop, axis=1)
        
        print("PASO 1.1 -------------> 7.0")
        
        # Definimos la X e y
        y_rf = df_01_copy['ENFERMEDAD'].tolist()
        X_rf = df_01_copy.drop('ENFERMEDAD', axis=1)
        
        print("PASO 1.1 -------------> 8.0")
        
        # Modelo entrenado con el conjunto de datos
        clf_rnd = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        clf_rnd.fit(X_rf, y_rf)
        
        print("PASO 1.1 -------------> 9.0")
   
        # Guardar el modelo en un archivo .pkl
        with open('modelo_rf.pkl', 'wb') as f:
            pickle.dump(clf_rnd, f)
        
        print("PASO 1.1 -------------> 10.0")




















































































