from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
from typing import Union, Tuple, List, Dict
import yaml


class TouristDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        countries: Union[Tuple, List],
        inp_seq_len: int,
        out_seq_len: int,
        normalize_para: Dict[str, pd.Series] = None,
        country=None,
    ) -> None:
        window_width = inp_seq_len + out_seq_len

        countries.sort()

        df = df.copy()
        df.index = df["date"].apply(pd.to_datetime)
        df.drop("date", axis=1, inplace=True)
        df.sort_index(inplace=True)
        # one hot encode the country
        for cntry in countries:
            df.loc[df["country"] == cntry, cntry] = 1
        df.fillna(0, inplace=True)
        label_counts = []
        for cntry in countries:
            length = len(df[df["country"] == cntry])
            label_count = length - window_width
            label_counts.append(label_count)
        df.drop("country", axis=1, inplace=True)
        df = df.astype(np.float32)
        if normalize_para is None:
            offset = df.min()
            scale = df.max() - df.min()
            normalize_para = {"scale": scale, "offset": offset}
        else:
            scale = normalize_para["scale"]
            offset = normalize_para["offset"]

        df = (df - offset) / scale
        df.sort_index(inplace=True, axis=1)

        self.countries = countries
        self.country = country
        self.df = df
        self.normalize_para = normalize_para
        self.inp_seq_len = inp_seq_len
        self.out_seq_len = out_seq_len
        self.window_width = window_width
        self.label_counts = label_counts
        self.columns = df.columns

        self._validate_self()

    def _validate_self(self):
        for i, (x, y) in enumerate(self):
            assert x.shape == torch.Size([self.inp_seq_len, len(self.columns)]), (
                x.shape,
                i,
            )
            assert y.shape == torch.Size([self.out_seq_len]), (y.shape, i)
            assert x.dtype == torch.float32, (x.dtype, i)
            assert y.dtype == torch.float32, (y.dtype, i)

    def set_country(self, country: str = None):
        if country is not None:
            if country not in self.countries:
                raise ValueError(
                    f"'{country}' not in self.countries. Available countries are {self.countries}"
                )
        self.country = country

    def __len__(self):
        if self.country is None:
            return sum(self.label_counts)
        else:
            country_rec_count = len(self.df[self.df[self.country] == 1])
            label_count = country_rec_count - self.window_width + 1
            label_count = max(label_count, 0)
            return label_count

    def __getitem__(self, idx: int):
        if idx >= len(self):
            raise StopIteration
        if self.country is None:
            label_id = 0
            while idx - self.label_counts[label_id] > 0:
                idx -= self.label_counts[label_id]
                label_id += 1
            country = self.countries[label_id]
        else:
            country = self.country

        rel_recs = self.df[self.df[country] == 1]
        frame = rel_recs[idx : idx + self.window_width]
        x = frame[: self.inp_seq_len]
        y = frame[self.inp_seq_len :]["count"]
        x, y = torch.Tensor(x.to_numpy()), torch.Tensor(y.to_numpy())

        return x, y

    def get_date(self, idx=None):
        if idx is not None:
            if idx == -1:
                assert self.country is not None, "Country is not selected"
                rel_recs = self.df[self.df[self.country] == 1]
                rel_recs = rel_recs[-self.inp_seq_len :]
                dates = rel_recs.index
                return dates
            else:
                if idx > len(self):
                    raise ValueError(f"Data index ({idx}) out of range")
                if self.country is None:
                    label_id = 0
                    while idx - self.label_counts[label_id] > 0:
                        idx -= self.label_counts[label_id]
                        label_id += 1
                    country = self.countries[label_id]
                else:
                    country = self.country

                rel_recs = self.df[self.df[country] == 1]
                frame = rel_recs[idx : idx + self.window_width]
        else:
            if self.country is None:
                frame = self.df
            else:
                frame = self.df[self.df[self.country] == 1]

        date = frame.index

        return date

    def get_counts(self, start_date, end_date):
        assert self.country is not None, "Country is not selected"
        rel_recs = self.df[self.df[self.country] == 1]
        frame = rel_recs[(start_date <= rel_recs.index) & (rel_recs.index <= end_date)]
        scale, offset = (
            self.normalize_para["scale"]["count"],
            self.normalize_para["offset"]["count"],
        )
        counts = frame["count"]
        counts = counts * scale + offset
        return counts

    def get_last_instance(self) -> np.ndarray:
        assert self.country is not None, "Country is not selected"
        rel_recs = self.df[self.df[self.country] == 1]
        rel_recs = rel_recs[-self.inp_seq_len :]
        instance = torch.Tensor(rel_recs.to_numpy())
        return instance


def load_data(
    inp_seq_len,
    out_seq_len,
    data_path="./data",
    train_ratio=0.67,
    normalize_para: Dict[str, Union[pd.Series, Dict[str, float]]] = None,
):
    if normalize_para is not None:
        assert "scale" in normalize_para.keys() and "offset" in normalize_para.keys()
        if type(normalize_para["scale"]) == dict:
            normalize_para = {
                "scale": pd.Series(normalize_para["scale"]),
                "offset": pd.Series(normalize_para["offset"]),
            }

    tourist_df = pd.read_csv(f"{data_path}/total-tourist-count.csv")
    tourist_df = tourist_df.sort_values(["date", "country"])
    context_df = pd.read_csv(f"{data_path}/contextual-data.csv")
    df = tourist_df.merge(context_df, "left", "date")
    df.sort_values(["date", "country"], inplace=True)
    countries = []
    with open(f"{data_path}/countries.yaml") as handler:
        countries = yaml.load(handler, Loader=yaml.FullLoader)["names"]

    # train-test split for time series
    train_size = int(len(df) * train_ratio)
    test_size = len(df) - train_size
    train_df, test_df = df[:train_size], df[train_size:]
    train_ds = TouristDataset(
        train_df, countries, inp_seq_len, out_seq_len, normalize_para
    )
    test_ds = TouristDataset(
        test_df,
        countries,
        inp_seq_len,
        out_seq_len,
        normalize_para=train_ds.normalize_para,
    )
    normalize_para = train_ds.normalize_para

    return train_ds, test_ds


def load_dataset(config_dir: str) -> TouristDataset:
    config = {}
    with open(f"{config_dir}/config.yaml") as handler:
        config = yaml.load(handler, Loader=yaml.FullLoader)

    normalize_para = config["normalize_para"]
    if normalize_para is not None:
        assert "scale" in normalize_para.keys() and "offset" in normalize_para.keys()
        if type(normalize_para["scale"]) == dict:
            normalize_para = {
                "scale": pd.Series(normalize_para["scale"]),
                "offset": pd.Series(normalize_para["offset"]),
            }

    tourist_df = pd.read_csv(f"{config_dir}/total-tourist-count.csv")
    tourist_df = tourist_df.sort_values(["date", "country"])
    context_df = pd.read_csv(f"{config_dir}/contextual-data.csv")
    df = tourist_df.merge(context_df, "left", "date")
    df.sort_values(["date", "country"], inplace=True)
    countries = config["countries"]
    inp_seq_len = config["window"]["inp_seq_len"]
    out_seq_len = config["window"]["out_seq_len"]
    ds = TouristDataset(
        df,
        countries,
        inp_seq_len,
        out_seq_len,
        normalize_para=normalize_para,
    )

    return ds
