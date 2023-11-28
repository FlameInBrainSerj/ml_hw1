from fastapi import FastAPI, UploadFile
from fastapi.exceptions import HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, confloat
from typing import List

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import TargetEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

from pathlib import Path
from io import BytesIO, StringIO
import logging
import pickle

import warnings

warnings.filterwarnings('ignore')
PATH_UTILS = Path.cwd() / "utils"

app = FastAPI()


class Item(BaseModel):
    name: str
    year: int
    selling_price: int | None
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: confloat(ge=0)


class Items(BaseModel):
    objects: List[Item]


def props(cls) -> List:
    """Get all the attributes of the class, exclduing default ones"""

    return [i for i in cls.__dict__.keys() if i[:1] != "_"]


def item_to_df(item: Item) -> pd.DataFrame:
    """Turns Item instance to pd.Dataframe instance"""

    dct = {f"{prop}": [getattr(item, prop)] for prop in props(item)}
    return pd.DataFrame.from_dict(dct, orient="columns")


def read_pickle(filename: str):
    """Read the content of .pkl file"""

    with open(filename, "rb") as file:
        return pickle.load(file)


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data according to certain pipeline"""

    # Get all the transformers and the model's weights
    try:
        targ_enc = read_pickle((PATH_UTILS / "target_encoder.pkl"))
        ss = read_pickle((PATH_UTILS / "standard_scaler.pkl"))
        poly = read_pickle((PATH_UTILS / "polynomial_features.pkl"))
        coef_lst = []
        with open((PATH_UTILS / "poly_lst.txt"), "r") as file:
            for line in file:
                coef_lst.append(line.strip())
    except Exception as e:
        logging.error(e)
        logging.error("Some of the files were not read correctly")

    try:
        # Preprocess the data (str to float)
        data.drop(["name", "selling_price", "torque"], axis=1, inplace=True)
        for col in ["mileage", "engine", "max_power"]:
            data[col] = data[col].str.replace(r"\D", "", regex=True).astype(float)

        # Preprocess the data (TargetEncoder)
        cat_colls = ["fuel", "seller_type", "transmission", "owner", "seats"]
        data = pd.concat(
            [
                data.drop([*cat_colls], axis=1),
                pd.DataFrame(targ_enc.transform(data[cat_colls]), columns=cat_colls),
            ],
            axis=1,
        )

        # Preprocess the data (log-transform)
        for col in ["km_driven", "max_power"]:
            data.loc[:, col] = np.log(data["km_driven"] + 0.1)

        # Preprocess the data (StandardScaler)
        data_scaled = ss.transform(data)
        data_scaled = pd.DataFrame(data_scaled, columns=data.columns)

        # Preprocess the data (PolynomialFeatures)
        data_scaled_new = poly.transform(data_scaled)
        data_scaled_new = pd.DataFrame(
            data_scaled_new, columns=poly.get_feature_names_out(data_scaled.columns)
        )

        # Filter the columns created with PolynomialFeatures
        data_scaled_new = data_scaled_new[coef_lst]

        return data_scaled_new

    except ValueError as e:
        raise HTTPException(
            status_code=409,
            detail="Your file contains NaN(s), clear them, so we can give you a prediction :)",
        )


@app.post("/predict_item", response_model=float, summary="Predict item")
def predict_item(item: Item) -> float:
    """Gets the .json - observation and returns the prediction of the observation (car) price"""

    # Import model
    model = read_pickle((PATH_UTILS / "final_model.pkl"))

    # Turn to pd.DataFrame
    df = item_to_df(item)
    # Preprocess
    df = preprocess_data(df)
    # Prediction
    pred = np.exp(model.predict(df))

    return pred


@app.post("/predict_items", summary="Predict items")
def predict_items(file: UploadFile) -> StreamingResponse:
    """Get the .csv file with observations and returns same .csv file with
    prediction column - prediciton of each observation' (car) price"""

    # Import model
    model = read_pickle((PATH_UTILS / "final_model.pkl"))

    # Read the file content
    content = file.file.read()
    df = pd.read_csv(BytesIO(content))

    # Preprocess
    df_new = preprocess_data(df.copy())
    # Prediction
    pred = pd.DataFrame(np.exp(model.predict(df_new)), columns=["pred"])

    # Final df
    final_df = pd.concat([df, pred], axis=1)

    stream = StringIO()
    final_df.to_csv(stream, index=False)
    response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=pred.csv"

    return response
