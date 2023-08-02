from fastapi import FastAPI
import pandas as pd
import numpy as np
from scikit-learn.feature_extraction.text import TfidfVectorizer
from scikit-learn.metrics.pairwise import cosine_similarity
app=FastAPI()
df=pd.read_csv('data.csv')
df['release_date']=pd.to_datetime(df['release_date'], errors='coerce')
data_similitud=pd.read_csv('similitud.csv')
@app.get('/')
def index():
    return 'Proyecto individual final del curso de data science de Henry. Mauricio Ljungberg'


@app.get('/cantidad_ﬁlmaciones_mes/{mes}')
def cantidad_ﬁlmaciones_mes(mes):
    '''Se ingresa un mes en idioma Español. Debe devolver la cantidad de películas que fueron
estrenadas en el mes consultado en la totalidad del dataset.
Ejemplo de retorno: X cantidad de películas fueron estrenadas en el mes de X'''
    meses = {
        'enero': 'January',
        'febrero': 'February',
        'marzo': 'March',
        'abril': 'April',
        'mayo': 'May',
        'junio': 'June',
        'julio': 'July',
        'agosto': 'August',
        'septiembre': 'September',
        'octubre': 'October',
        'noviembre': 'November',
        'diciembre': 'December'
        }
    df_mes = df[df['release_date'].dt.month_name() == meses[mes].capitalize()]
    
    cantidad = len(df_mes)
    #return '{} cantidad de películas fueron estrenadas el mes de {}'.format(cantidad,mes)
    return {'mes':mes, 'cantidad':cantidad}



@app.get('/cantidad_filmaciones_dia/{dia}')
def cantidad_ﬁlmaciones_dia(dia):
    '''Se ingresa un día en idioma Español. Debe devolver la cantidad de películas que fueron
estrenadas en día consultado en la totalidad del dataset.
Ejemplo de retorno: X cantidad de películas fueron estrenadas en los días X'''
    dias = {
    'lunes': 'Monday',
    'martes': 'Tuesday',
    'miércoles': 'Wednesday',
    'miercoles': 'Wednesday',
    'jueves': 'Thursday',
    'viernes': 'Friday',
    'sábado': 'Saturday',
    'sabado': 'Saturday',
    'domingo': 'Sunday'
    }
    df_dia = df[df['release_date'].dt.day_name() == dias[dia].capitalize()]
    
    cantidad = len(df_dia)
    
    
    return {'dia:':dia,'cantidad:':cantidad}


@app.get('/score_titulo/{titulo}')
def score_titulo(titulo):
    '''Se ingresa el título de una ﬁlmación esperando como respuesta el título, el año de estreno y el score.
Ejemplo de retorno: La película X fue estrenada en el año X con un score/popularidad de X'''
    data=df.loc[df['title']==titulo]
    if data.shape[0]>1:
        data=data.iloc[0]
    score=float(data['popularity'])
    anio=int(data['release_year'])
    return ({'titulo':titulo, 'año':anio, 'popularidad':score})



@app.get('/votos_titulo/{titulo}')
def votos_titulo(titulo):
    '''Se ingresa el título de una ﬁlmación esperando como respuesta el título, la cantidad de votos y el valor 
    promedio de las votaciones. La misma variable deberá de contar con al menos 2000 valoraciones, caso contrario, 
    debemos contar con un mensaje avisando que no cumple esta condición y que por ende, 
    no se devuelve ningun valor.
    Ejemplo de retorno: La película X fue estrenada en el año X. La misma cuenta con un total de X 
    valoraciones, con un promedio de X'''
    data=df.loc[df['title']==titulo]
    if data.shape[0]>1:
        data=data.iloc[0]
    if float(data['vote_count'])<2000:
        return 'La pelicula no cuenta con suficientes votos'
    votos_promedio=data['vote_average']
    votos_cantidad=data['vote_count']
    anio=int(data['release_year'])
    return {'titulo':titulo, 'año':anio, 'voto_total':votos_cantidad, 'voto_promedio':votos_promedio}

@app.get('/get_actor/{actor}')
def get_actor(nombre_actor):
    '''Se ingresa el nombre de un actor que se encuentre dentro de un dataset debiendo devolver el 
    éxito del mismo medido a través del retorno. Además, la cantidad de películas que en las que ha 
    participado y el promedio de retorno. La deﬁnición no deberá considerar directores.
    Ejemplo de retorno: El actor X ha participado de X cantidad de ﬁlmaciones, el mismo ha
    conseguido un retorno de X con un promedio de X por ﬁlmación'''
    cantidad = 0
    retorno = 0
    for index, row in df.iterrows():
        if nombre_actor in row['actores']:
            cantidad += 1
            retorno += row['return']
            if row['return']==0:
                cantidad-=1
    promedio=retorno/cantidad
    return {'actor':nombre_actor, 'cantidad_filmaciones':cantidad, 'retorno_total':retorno, 'retorno_promedio':promedio}


@app.get('/get_director/{director}')
def get_director(nombre_director):
    '''Se ingresa el nombre de un director que se encuentre dentro de un dataset debiendo devolver el
    éxito del mismo medido a través del retorno. Además, deberá devolver el nombre de cada
    película con la fecha de lanzamiento, retorno individual, costo y ganancia de la misma.'''
    data=df.loc[df['director']==nombre_director]
    retorno=float(data['return'].sum())
    peliculas=data['title']
    print('el retorno total es:',retorno)
    #data.drop(['popularity','vote_average','vote_count','release_year','actores','director'],axis=1,inplace=True)
    #devolver=data.to_json()
    #return {'director':nombre_director, 'retorno_total_director':retorno, 'peliculas':data['title'], 'fecha de estreno':data['release_date'], 'retorno_pelicula':data['return'], 'budget_pelicula':data['budget'], 'revenue_pelicula':data['revenue']}
    return {'director':nombre_director,'retorno_total':retorno}

@app.get('/recomendacion/{titulo}')
def recomendacion(titulo: str):
    '''Ingresa un nombre de película y recibe una recomendación de cinco películas similares'''
    if titulo in data_similitud.columns:
        # Obtén los valores de similitud de la película con todas las demás películas
        similitudes = data_similitud[titulo].values
        # Ordena los valores de similitud de forma descendente y obtén los índices correspondientes
        indices_ordenados = np.argsort(similitudes)[::-1]
        # Obtén los nombres de las películas más similares (excluyendo la película dada)
        peliculas_similares = [data_similitud.columns[idx] for idx in indices_ordenados if data_similitud.columns[idx] != titulo]
        # Limita la lista a las 5 películas más similares
        peliculas_similares = peliculas_similares[:5]
    else:
        # No se encontró el título exacto, se realizará una búsqueda aproximada
        vectorizer = TfidfVectorizer()
        titulos_peliculas = list(data_similitud.columns)
        titulos_peliculas.append(titulo)  # Agrega el título ingresado a la lista de títulos
        matriz_tfidf = vectorizer.fit_transform(titulos_peliculas)
        # Calcula la similitud coseno entre el título ingresado y todos los títulos
        similitud = cosine_similarity(matriz_tfidf[-1], matriz_tfidf[:-1])
        # Encuentra el índice de la película más similar en orden descendente
        indice_pelicula_similar = np.argmax(similitud)
        # Obtén el nombre de la película más similar
        pelicula_similar = titulos_peliculas[indice_pelicula_similar]
        # Obtén los valores de similitud de la película similar con todas las demás películas
        similitudes = data_similitud[pelicula_similar].values
        # Ordena los valores de similitud de forma descendente y obtén los índices correspondientes
        indices_ordenados = np.argsort(similitudes)[::-1]
        # Obtén los nombres de las películas más similares (excluyendo la película similar)
        peliculas_similares = [data_similitud.columns[idx] for idx in indices_ordenados if data_similitud.columns[idx] != pelicula_similar]
        # Limita la lista a las 5 películas más similares
        peliculas_similares = peliculas_similares[:5]
    return {'lista_recomendada': peliculas_similares}