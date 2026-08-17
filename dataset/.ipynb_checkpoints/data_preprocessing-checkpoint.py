# dataset/data_preprocessing.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder


def preprocess_data(df):
    """
    Common preprocessing for train and test data
    """

    df = df.copy()

    # Missing values

    df["Type_Occupation"] = df["Type_Occupation"].fillna(
        "Unknown"
    )

    df["GENDER"] = df["GENDER"].fillna(
        "M"
    )

    df["Annual_income"] = df["Annual_income"].fillna(
        df["Annual_income"].mode()[0]
    )

    df["Birthday_count"] = df["Birthday_count"].fillna(
        df["Birthday_count"].mode()[0]
    )

    # Remove Mobile_phone

    if "Mobile_phone" in df.columns:
        df.drop(
            columns=["Mobile_phone"],
            inplace=True
        )

    # Age

    df["Age"] = abs(
        df["Birthday_count"]
    ) / 365

    # Remove Birthday_count

    df.drop(
        columns=["Birthday_count"],
        inplace=True
    )

    return df


def fit_encoders(df):
    """
    Fit encoders on training data
    """

    encoders = {}

    categorical_columns = (
        df.select_dtypes(
            include="object"
        ).columns
    )

    for col in categorical_columns:

        encoder = LabelEncoder()

        df[col] = encoder.fit_transform(
            df[col].astype(str)
        )

        encoders[col] = encoder

    return df, encoders


def transform_data(df, encoders):
    """
    Transform test data using fitted encoders
    """

    for col, encoder in encoders.items():

        mapping = {
            label: idx
            for idx, label
            in enumerate(encoder.classes_)
        }

        df[col] = (
            df[col]
            .astype(str)
            .map(mapping)
            .fillna(-1)
            .astype(int)
        )

    return df