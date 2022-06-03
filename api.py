from fastapi import FastAPI, UploadFile
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = FastAPI()


def train(input_file):
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.utils import resample, shuffle
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report

    df = pd.read_csv(input_file)
    df = df.drop(["Unnamed: 0", "isFlaggedFraud"], axis=1)
    df = df.rename(
        columns={
            "step": "Tiempo",
            "type": "Tipo",
            "amount": "Monto",
            "nameOrig": "Nombre_origen",
            "oldbalanceOrg": "Saldo_orig_inicial",
            "newbalanceOrig": "Saldo_orig_fin",
            "nameDest": "Nombre_destino",
            "oldbalanceDest": "saldo_dest_inicial",
            "newbalanceDest": "saldo_dest_final",
            "isFraud": "Fraude",
        }
    )
    df = df.replace(
        to_replace=["CASH_OUT", "PAYMENT", "CASH_IN", "TRANSFER", "DEBIT"],
        value=[
            "1",
            "2",
            "3",
            "4",
            "5",
        ],
    )
    df["Tipo"] = df["Tipo"].astype(int)
    df1 = df.drop(["Nombre_origen", "Nombre_destino"], axis=1)
    scalar = StandardScaler()
    df1_scaled = scalar.fit_transform(df1.drop(columns="Fraude"))
    df1_scaled = pd.DataFrame(df1_scaled, columns=df1.drop(columns="Fraude").columns)
    X = df1_scaled
    y = df1["Fraude"]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    x_train["Fraude"] = y_train
    df_majority = x_train[x_train["Fraude"] == 0]
    df_minority = x_train[x_train["Fraude"] == 1]
    df_majority_downsampled = resample(
        df_majority, replace=False, n_samples=len(df_minority), random_state=123
    )
    df_downsampled = shhuffle(pd.concat([df_majority_downsampled, df_minority]))
    x_train = df_downsampled.drop(columns="Fraude")
    y_train = df_downsampled["Fraude"]
    rf = RandomForestClassifier(
        bootstrap=False, max_depth=50, min_samples_split=10, n_estimators=2000
    )
    rf.fit(x_train, y_train)
    rf_preds = rf.predict(x_test)
    with open("./modelo_fraude.pickle", "wb") as model_file:
        pickle.dump(rf, model_file)
    return classification_report(y_test, rf_preds, output_dict=True)


@app.post("/train")
async def trainer(file: UploadFile):
    result = train(input_file=file.filename)
    return result


@app.post("/predict")
async def predict(
    Tiempo: float,
    Tipo: float,
    Monto: float,
    Saldo_orig_inicial: float,
    Saldo_orig_fin: float,
    Saldo_dest_inicial: float,
    Saldo_dest_final: float,
):
    with open("./modelo_fraude.pickle", "rb") as file:
        model = pickle.load(file)
    input_dict = {
        "Tiempo": Tiempo,
        "Tipo": Tipo,
        "Monto": Monto,
        "Saldo_orig_inicial": Saldo_orig_inicial,
        "Saldo_orig_fin": Saldo_orig_fin,
        "Saldo_dest_inicial": Saldo_dest_inicial,
        "Saldo_dest_final": Saldo_dest_final,
    }
    df = pd.DataFrame.from_dict(input_dict, orient="index").T
    print(df)
    scaler = StandardScaler()
    columns = df.columns
    df = scaler.fit_transform(df)
    df = pd.DataFrame(df, columns=columns)
    preds = model.predict(df)
    print(preds)
    if preds == 1:
        return "Fraude"
    elif preds == 0:
        return "No Fraude"
