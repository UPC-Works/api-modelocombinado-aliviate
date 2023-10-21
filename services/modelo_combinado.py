import pandas as pd
import os
import numpy as np
import platform
import psutil
import tensorflow as tf
import sys
import pickle
import redis
from sklearn.preprocessing import  LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
import json

class ModeloCombinadoService:

    def modelo_combinado(nombre_documento):
        
        #Leer archivo excel
        df_historia_clinica = pd.read_excel(f'./files/{nombre_documento}')
        
        #Leer archivo excel
        df_01 = pd.read_excel('./files/BD_HISTORIAS_CLINICAS_ENTRENAMIENTO.xlsx')
        
        #Leer archivo excel
        df_02 = pd.read_excel('./files/BD_SIGNOS_SINTOMAS.xlsx')
        
        #Unimos df_02_copy y df_01_copy2
        df_append = pd.concat([df_02,df_01])
        
        def remove_labels(df, label_name):
            X = df.drop(label_name, axis=1)
            y = df[label_name].copy()
            return (X, y)
        
        # Traemos los datos desde Redis
                
        # Cargar el modelo desde el archivo .pkl
        with open('modelo_rf.pkl', 'rb') as f:
            clf_rnd = pickle.load(f)
 
        # Cargar el modelo desde el archivo .pkl
        with open('tfidf_matrix.pkl', 'rb') as f:
            tfidf_matrix = pickle.load(f)
 
        # Cargar el modelo desde el archivo .pkl
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
                    
        
        # Segmentamos el dataframe de historia clínica en dos copias
        df_hc_copy_1=df_historia_clinica.copy()
        df_hc_copy_2=df_historia_clinica.copy()
        

        #################################################################################################################################################

        # ========================== Segmentamos el dataframe df_hc_copy_1 para obtener los datos clinicos =============================
        # Eliminamos los signos y sintomas
        df_hc_copy_1 = df_hc_copy_1.drop('SIGNOS_SINTOMAS', axis=1)

        # Aplicar la codificación ordinal para transformar variables categóricas a numericas
        encoder = LabelEncoder()
        
        # Definir las columnas para aplicar el encoding
        columns_to_encode = ['SEXO','GRUPO_SANGUINEO','GRADO_INSTITUCION','ESTADO_CIVIL','TUVO_TUBERCULOSIS','TIENE_INF_RENAL_GLAUCOMA','TIENE_INF_TRANS_SEX','TIENE_SIDA','TIENE_ITS','TIENE_HEPATITIS','TIENE_DIABETES','TIENE_HTA','TIENE_SOBREPESO','TIENE_DISLIPENIA','TIENE_DEPRESION_ESQUIZOFRENIA','TIENE_HOSPITALIZACION_TRANSFUCIONES','TIENE_INTER_QUIRURJICA','TIENE_PREMATURO','TIENE_ABORTO','TIENE_PARTO','FLUJO_VAG_PATOLOGICO','TIENE_EXAM_PROSTATA','TIENE_VIOLENCIA','TIENE_DBM','TIENE_INFARTO','TIENE_CANCER','TIENE_DEPRESION','TIENE_PROB_PSIQUIATRICOS','RS_MISMO_SEXO','MEDICAMENTO_FRECUENTE','REACCION_MEDICAMENTOS','TIENE_CONSUMO_TABACO','TIENE_CONSUMO_ALCOHOL','EDAD_INI_RELACION_SEXUAL','NUM_PAREJAS','DISMENORREA','TIENE_EMBARAZO','TIENE_FIEBRE_15_DIAS','TIENE_TOS_15_DIAS','TIENE_VAC_ANTITETANICA','TIENE_VAC_ANTIAMERILICA','TIENE_VAC_ANTIHEPATITIS_B','TIENE_ENCIAS','TIENE_CARIES','TIENE_EDENTULISMO_TOTAL','TIENE_ANSIEDAD','TIENE_EDENTULISMO_PARCIAL','TIENE_EXAM_VISUAL','TIENE_URG_TRATAMIENTO_BUCAL','TIENE_MAMOGRAFIA','TIENE_EXAM_COLESTEROL','TIENE_EXAM_MAMAS','TIENE_EXAM_GLUCOSA','TIENE_HAB_FISICA','TIENE_PLANIFICACION_SEXUAL','TIENE_HAB_ALCOHOL','TIENE_HAB_DROGAS']
        df_hc_copy_1[columns_to_encode] = df_hc_copy_1[columns_to_encode].apply(encoder.fit_transform)

        # Eliminamos el ID, el nombre y las fechas
        columns_to_drop = ['ID_HISTORIA_CLINICA','FECHA_REGISTRO','DIRECCION','FECHA_ULT_REGLA','TIENE_EXAM_PELVICO_PAP','NOMBRES','APELLIDOS','EDAD','FECHA_NACIMIENTO','MENARQUIA','GESTACION','PRESION_ARTERIAL','LESIONES_GENITALES','TEMPERATURA','TALLA','PRESION_ARTERIAL']

        #Definimos el segmento de datos clínicos
        df_hc_copy_1 = df_hc_copy_1.drop(columns_to_drop, axis=1)
        X_datos_clinicos, y_datos_clinicos = remove_labels(df_hc_copy_1, 'ENFERMEDAD')
        clf_rnd.fit(X_datos_clinicos, y_datos_clinicos)

        # ========================== Segmentamos el dataframe df_hc_copy_2 para obtener los signos y sintomas ==========================
        # Definimos el segmento de signos y sintomas
        df_hc_signos_sintomas=df_hc_copy_2['SIGNOS_SINTOMAS'].iloc[0]

        #######################################################################################################################################################

        # ============= Obtenemos el top de las 3 enfermedades mas probables tomando el segmento de datos clinicos =====================
        # Obtener las probabilidades tomando en cuenta los datos clinicos
        probabilities = clf_rnd.predict_proba(X_datos_clinicos)

        # Obtener la primera instancia
        first_instance = probabilities[0]
        
        # Crear un DataFrame a partir de las probabilidades y las clases
        df_prob = pd.DataFrame({'ENFERMEDAD': clf_rnd.classes_, 'PROBABILIDAD': first_instance})

        # Ordenar el DataFrame por probabilidad en orden descendente y obtener las tres posibles enfermedades
        df_top_3_enfermedades_datos_clinicos = df_prob.sort_values('PROBABILIDAD', ascending=False).head(3)

        
        # =================== Obtenemos el top de las 3 enfermedades tomando el segmento de signos y sintomas =========================
        # Obtener las probabilidades tomando en cuenta los signos y sintomas
        def search_symptoms(symptoms, num_results=3):
            # Transformar los síntomas de entrada en un vector TF-IDF
            tfidf_symptoms = vectorizer.transform([symptoms])

            # Calcular la similitud del coseno entre los síntomas de entrada y todos los signos y síntomas en la base de datos
            cosine_similarities = cosine_similarity(tfidf_symptoms, tfidf_matrix).flatten()

            # Crear un dataframe que contenga las similitudes y sus índices correspondientes
            similarities_df = pd.DataFrame({
                'similarity': cosine_similarities,
                'index': np.arange(cosine_similarities.size)
            })

            # Ordenar el dataframe por similitud y eliminar filas con índices duplicados, conservando solo la fila con la mayor similitud
            similarities_df = similarities_df.sort_values(by='similarity', ascending=False).drop_duplicates(subset='index', keep='first')

            # Obtener los índices de los signos y síntomas más similares
            most_similar_indices = similarities_df['index'].values[:num_results * 2]  # obtener el doble de los resultados requeridos inicialmente

            # Devolver las enfermedades y los tratamientos correspondientes a esos signos y síntomas, junto con las similitudes del coseno
            results = df_append.iloc[most_similar_indices][['ENFERMEDAD']]
            results['PROBABILIDAD'] = cosine_similarities[most_similar_indices]

            # Eliminar duplicados en 'enfermedad' y conservar solo los primeros 'num_results' resultados
            results = results.drop_duplicates(subset='ENFERMEDAD', keep='first')[:num_results]

            return results
        
        #######################################################################################################################################################

        # De las 6 enfermedades resultantes (3 de datos clínicos y 3 de signos y síntomas), extraeremos las 3 más probables
        # Se combianan los dos dataframes
        df_combined = pd.concat([df_top_3_enfermedades_datos_clinicos, search_symptoms(df_hc_signos_sintomas)])

        # Se ordena el dataframe combinado por probabilidad en orden descendente
        df_combined_sorted = df_combined.sort_values(by='PROBABILIDAD', ascending=False)

        # Eliminar duplicados en la columna de enfermedad
        df_combined_sorted_unique = df_combined_sorted.drop_duplicates(subset='ENFERMEDAD')

        # Seleccionamos las tres enfermedades más probables sin duplicados
        df_top3_enfermedades = df_combined_sorted_unique['ENFERMEDAD'].head(3)

        # Verificar si hay menos de tres enfermedades únicas
        if len(df_top3_enfermedades) < 3:
            # Obtener enfermedades adicionales que no están en las primeras tres posiciones
            df_rest_enfermedades = df_combined_sorted_unique[~df_combined_sorted_unique['ENFERMEDAD'].isin(df_top3_enfermedades)]['ENFERMEDAD']
            # Completar las tres enfermedades utilizando las adicionales
            df_top3_enfermedades = pd.concat([df_top3_enfermedades, df_rest_enfermedades]).head(3)

        # Obtener las probabilidades correspondientes a las enfermedades seleccionadas
        df_top3_probabilidades = df_combined_sorted.loc[df_combined_sorted['ENFERMEDAD'].isin(df_top3_enfermedades), 'PROBABILIDAD']

        # Imprimir las tres enfermedades más probables con sus respectivas probabilidades
        data = {'ENFERMEDAD': df_top3_enfermedades, 'PROBABILIDAD': df_top3_probabilidades}
        df_resultados = pd.DataFrame(data)
        # Eliminar filas con valores NaN
        df_resultados = df_resultados.dropna()

        # Filtrar las probabilidades con valor distinto de 1
        df_resultados_filtrado = df_resultados[df_resultados['PROBABILIDAD'] != 1]

        # Reiniciar los índices
        df_resultados = df_resultados.reset_index(drop=True)

        # Especificamos el archivo
        archivo_a_borrar = f'./files/{nombre_documento}'
        
        # Intenta eliminar el archivo
        try:
            os.remove(archivo_a_borrar)
            print(f"El archivo '{archivo_a_borrar}' ha sido eliminado correctamente.")
        except FileNotFoundError:
            print(f"El archivo '{archivo_a_borrar}' no se encontró y no pudo ser eliminado.")
        except Exception as e:
            print(f"Ocurrió un error al intentar eliminar el archivo: {e}")
 
        return json.loads(df_resultados_filtrado.to_json(orient='records'))

