import pandas as pd
import streamlit as st 
import utils

import joblib

from sklearn.model_selection import train_test_split

import os

SEED = 62532

DATA_FOLDER = os.path.join('data')
DATA_FILE = os.path.join(DATA_FOLDER, 'Obesity.csv')

MODEL_FOLDER = os.path.join('model')
MODEL_FILE = os.path.join(MODEL_FOLDER, 'model.joblib')

df = pd.read_csv(DATA_FILE)

features = [
    'age',
    'height',
    'family_history',
    'favc',
    'caec',
    'scc',
    'faf',
    'mtrans_automobile', 'mtrans_bike', 'mtrans_motorbike', 'mtrans_public_transportation', 'mtrans_walking'
]


df_ML = utils.apply_pipeline_ML(df)
X = df_ML[features]
y = df_ML.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y, random_state=SEED)

model = joblib.load(MODEL_FILE)

### Website
st.title("Análise e previsão de risco de Obesidade")

def write_header(text):
    st.write(f'### {text}')

write_header("Idade")
input_age = int(st.slider("Selecione sua idade", 16, 100))

write_header("Altura")
input_height = float(st.slider("Selecione sua altura", 0.5, 2.5))

# write_header("Peso")

write_header("Histórico familiar") #family_history
input_history = st.checkbox("Alguém na minha família tem ou já teve obesidade")
dict_history = {
    True:'yes',
    False:'no'
}
input_history = dict_history[input_history]

write_header("Alimentos calóricos") #FAVC
input_favc = st.checkbox("Consumo alimentos calóricos com frequência")
dict_favc = {
    True:'yes',
    False:'no'
}
input_favc = dict_favc[input_favc]
# write_header("Alimentos vegetais") #FCVC

# write_header("Quantidade de refeições") #NCP

write_header("Alimentação entre refeições") #CAEC
dict_between_meals = {
    'Não':'no',
    'Às vezes':'Sometimes',
    'Frequentemente':'Frequently',
    'Sempre':'Always'
}

input_between_meals = st.selectbox(label=
    "Com qual frequência você se alimenta entre refeições (café da manhã, almoço, janta)?",
    options = dict_between_meals.keys())

input_between_meals = dict_between_meals[input_between_meals]

# write_header("Fuma") #Smoke

# write_header("Ingestão de água") #CH2O

write_header("Monitoramento de calorias") #SCC
input_scc = st.checkbox("Monitoro minha ingestão de calorias")
dict_scc = {
    True:'yes',
    False:'no'
}
input_scc = dict_scc[input_scc]

write_header("Atividade física") #FAF
dict_faf = {
    'Nenhuma':0,
    '1 - 2 vezes por semana':1,
    '3 - 4 vezes por semana':2,
    '5 ou mais vezes por semana':3
}
input_faf = st.selectbox(
    "Com qual frequência você practica alguma atividade física?",
    dict_faf.keys()
)
input_faf = dict_faf[input_faf]

# write_header("Tempo em equipamentos eletrônicos") #TUE

# write_header("Consumo de álcool") #CALC

write_header("Meio de transporte") #MTRANS
dict_mtrans = {
    'Andar':'Walking',
    'Bike':'Bicicleta',
    'Automóvel':'Automobile',
    'Motocicleta':'Motorbike',
    'Transporte público':'Public_Transportation',
}
input_mtrans = st.selectbox(
    "Qual seu meio de transporte mais utilizado?",
    dict_mtrans.keys()
)
input_mtrans = dict_mtrans[input_mtrans]

user_data = [
    "Male", # gender
    input_age,
    input_height,
    input_history,
    input_favc,
    "no", # favc
    0, # fcvc
    0, # ncp
    "no", # caec
    "no",
    0, # ch2o
    input_scc, # scc
    input_faf, # faf
    0, # tue
    input_mtrans,
    0 # target
]

all_columns = [
    'Gender', 'Age', 'Height', 'Weight', 'family_history', 'FAVC', 'FCVC',
    'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE', 'CALC', 'MTRANS',
    'Obesity']

features = [
    'age',
    'height',
    'family_history',
    'favc',
    'caec',
    'scc',
    'faf',
    'mtrans_automobile', 'mtrans_bike', 'mtrans_motorbike', 'mtrans_public_transportation', 'mtrans_walking'
]

user_df = pd.DataFrame(
    [user_data],
    columns = features
)


if st.button("Enviar"):
    feature_names = model.feature_names_in
    user = df_ML[feature_names]
    user_pred = model.predict(user)

    if user_pred[-1] == 0:
        st.success("### Não foi detectado risco elevado")
    else:
        st.error("### Há um risco elevado de obesidade")