import torch.nn as nn
import torch
import torch.optim as optim
import yaml
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .data import TouristDataset
from typing import Dict, List, Union


class ForecastModel(nn.Module):
    def __init__(
        self,
        inp_features: Tuple[str],
        out_features: Tuple[str],
        inp_seq_len: int,
        out_seq_len: int,
        hid_feature_count: int = 16,
        num_layers: int = 1,
    ):
        super().__init__()
        inp_feature_count = len(inp_features)
        out_feature_count = len(out_features)
        self.inp_features = inp_features
        self.out_features = out_features
        self.inp_feature_count = inp_feature_count
        self.out_feature_count = out_feature_count
        self.hid_feature_count = hid_feature_count
        self.inp_seq_len = inp_seq_len
        self.out_seq_len = out_seq_len
        self.window_width = inp_seq_len + out_seq_len
        self.num_layers = num_layers
        self.encoder = nn.LSTM(
            inp_feature_count, hid_feature_count, num_layers=num_layers
        )
        self.hidd_s_map = nn.Linear(
            num_layers * hid_feature_count, num_layers * out_feature_count
        )
        self.cell_s_map = nn.Linear(
            num_layers * hid_feature_count, num_layers * out_feature_count
        )
        self.decoder = nn.LSTM(
            hid_feature_count, out_feature_count, num_layers=num_layers
        )

    def forward(self, X: torch.Tensor):
        batch_input = True
        if len(X.shape) == 2:
            batch_input = False
            X = X.reshape((self.inp_seq_len, -1, self.inp_feature_count))

        batch_size = X.shape[1]
        enc_inp_hid = (
            torch.randn(self.num_layers, batch_size, self.hid_feature_count),
            torch.randn(self.num_layers, batch_size, self.hid_feature_count),
        )  # clean out hidden state
        enc_out, enc_out_hid = self.encoder(X, enc_inp_hid)

        dec_inp_val = enc_out[-self.out_seq_len :]
        map_inputs = [enc_out_hid[i].reshape(batch_size, -1) for i in range(2)]
        map_outputs = [self.hidd_s_map(map_inputs[0]), self.cell_s_map(map_inputs[1])]
        dec_inp_hid = [
            map_outputs[i].reshape(self.num_layers, batch_size, self.out_feature_count)
            for i in range(2)
        ]
        pred, dec_out_hid = self.decoder(dec_inp_val, dec_inp_hid)
        if not batch_input:
            pred = pred.reshape(self.out_seq_len, self.out_feature_count)

        return pred

    def plot(self, x, y, date, normalize_para, ax=None, title=None) -> plt.Figure:
        pred = self(x)

        # normalize the values
        inp = np.ones(self.window_width) * np.nan
        labels = np.ones(self.window_width) * np.nan
        preds = np.ones(self.window_width) * np.nan

        count_scale = normalize_para["scale"]["count"]
        count_offset = normalize_para["offset"]["count"]

        count_idx = self.inp_features.index("count")
        inp[: self.inp_seq_len] = (
            (x[:, count_idx] * count_scale + count_offset).detach().numpy().astype(int)
        )
        labels[self.inp_seq_len :] = (
            (y * count_scale + count_offset).detach().numpy().astype(int)
        )
        preds[self.inp_seq_len :] = (
            (pred * count_scale + count_offset).detach().numpy().astype(int).squeeze()
        )
        date_str = [f"{dt.year} {dt.month_name()}" for dt in date]

        # plot
        def_lower_lim, def_upper_lim = 0, count_scale + count_offset
        lower_lim = min(def_lower_lim, np.nanmin(np.hstack((inp, labels, preds))))
        upper_lim = max(def_upper_lim, np.nanmax(np.hstack((inp, labels, preds))))
        if ax is None:
            fig = plt.figure()
            if title:
                plt.title(title)
            plt.ylim(lower_lim, upper_lim)
            plt.plot(inp, label="Input")
            plt.plot(labels, linewidth=0, marker="o", markersize=5, label="Labels")
            plt.plot(preds, linewidth=0, marker="x", markersize=5, label="Predictions")
            plt.xticks(ticks=np.arange(len(inp)), labels=date_str, rotation=75)
            plt.legend()
            plt.grid()
            plt.close()

            return fig
        else:
            if title:
                ax.set_title(title)
            ax.set_ylim(lower_lim, upper_lim)
            ax.plot(inp, label="Input")
            ax.plot(labels, linewidth=0, marker="o", markersize=5, label="Labels")
            ax.plot(preds, linewidth=0, marker="x", markersize=5, label="Predictions")
            ax.set_xticks(ticks=np.arange(len(inp)), labels=date_str, rotation=75)
            ax.legend()
            ax.grid()


def collate_fn(inp: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
    X = [label[0] for label in inp]
    Y = [label[1] for label in inp]
    X_cat = torch.stack(X, axis=1)
    Y_cat = torch.stack(Y, axis=1)
    return X_cat, Y_cat


def load_run(path: str, best=True):
    model_details = {}
    with open(f"{path}/details.yaml") as handler:
        model_details = yaml.load(handler, Loader=yaml.FullLoader)

    model = ForecastModel(
        inp_features=model_details["inp_features"],
        out_features=model_details["out_features"],
        inp_seq_len=model_details["inp_seq_len"],
        out_seq_len=model_details["out_seq_len"],
    )
    optimizer = optim.Adam(model.parameters())
    # checkpoints = torch.load(f"{path}/checkpoints.tar")
    if best:
        print(
            f"---- Loading best state dicts at epoch {model_details['best_epoch']} ----"
        )
        # optimizer.load_state_dict(checkpoints["best_optimizer"])
        # model.load_state_dict(checkpoints["best_model"])
        optimizer.load_state_dict(torch.load(f"{path}/checkpoints/best_optimizer.pt"))
        model.load_state_dict(torch.load(f"{path}/checkpoints/best_model.pt"))
    else:
        print(
            f"---- Loading final state dicts at epoch {model_details['final_epoch']} ----"
        )
        # optimizer.load_state_dict(checkpoints["final_optimizer"])
        # model.load_state_dict(checkpoints["final_model"])
        optimizer.load_state_dict(torch.load(f"{path}/checkpoints/final_optimizer.pt"))
        model.load_state_dict(torch.load(f"{path}/checkpoints/final_model.pt"))

    return model, optimizer, model_details


def load_model(model_config_dir: str) -> ForecastModel:
    config_path = f"{model_config_dir}/config.yaml"
    weights_path = f"{model_config_dir}/best_model.pt"
    config = {config_path}
    with open(config_path) as handler:
        config = yaml.load(handler, yaml.FullLoader)
    model = ForecastModel(
        inp_features=config["inp_features"],
        out_features=config["out_features"],
        inp_seq_len=config["inp_seq_len"],
        out_seq_len=config["out_seq_len"],
        hid_feature_count=config["hid_feature_count"],
        num_layers=config["num_layers"],
    )
    model.load_state_dict(torch.load(weights_path))
    return model


def counts_df_2_dict(counts: pd.DataFrame) -> Dict[str, List]:
    dates = [f"{dt.year} {dt.month_name()}" for dt in counts.index]
    count = counts["count"].tolist()
    predicted = counts["predicted"].tolist()
    dic = {"date": dates, "count": count, "predicted": predicted}
    return dic


def get_counts(
    country: str,
    month: str,
    year: int,
    window_width: int,
    ds: TouristDataset,
    model: ForecastModel,
    parse_output: bool = True,
) -> Union[pd.DataFrame, Dict[str, List]]:
    months = []
    for next_month_date in range(12):
        months.append(
            (pd.to_datetime("2022-01-01") + pd.DateOffset(months=next_month_date))
            .month_name()
            .upper()
        )
    month = month.upper()

    assert country in ds.countries
    assert month in months

    ds.set_country(country)
    available_dates = ds.get_date()
    query_date = pd.to_datetime(f"{year}-{month}-15")

    predictable_dates = []
    last_date = available_dates[-1]
    for i in range(1, ds.out_seq_len + 1):
        predictable_dates.append(last_date + pd.DateOffset(months=i))

    def predict_final_counts(ds, model):
        x = ds.get_last_instance()
        predicted_counts = model(x).detach().numpy().squeeze()
        count_col_idx = ds.columns.tolist().index("count")
        available_counts = x[:, count_col_idx].numpy()

        counts = np.hstack((available_counts, predicted_counts))
        scale, offset = (
            ds.normalize_para["scale"]["count"],
            ds.normalize_para["offset"]["count"],
        )
        counts = np.round(counts * scale + offset).astype(int)
        x_dates = ds.get_date(-1).tolist()
        predicted_dates = [
            x_dates[-1] + pd.DateOffset(months=i + 1) for i in range(ds.out_seq_len)
        ]
        dates = x_dates + predicted_dates
        predicted_truth = [1 for _ in range(ds.inp_seq_len)] + [
            0 for _ in range(ds.out_seq_len)
        ]
        counts = pd.DataFrame({"count": counts, "predicted": predicted_truth})
        counts.index = dates

        return counts

    if query_date in available_dates:
        start_idx_arr = np.where(available_dates == query_date)[0]
        # query date already available
        start_idx = start_idx_arr[0]
        start_date = query_date
        end_date = query_date + pd.DateOffset(months=window_width)
        if start_idx + window_width < len(available_dates):
            # full window is already available
            counts = ds.get_counts(start_date, end_date)
            counts = pd.DataFrame(counts)
            counts["predicted"] = 1
        else:
            # full window not available, will need to predict the trail
            counts = predict_final_counts(ds, model)
            counts = counts[(start_date <= counts.index) & (counts.index <= end_date)]
    else:
        if query_date in predictable_dates:
            # query dates's count will be predicted
            start_idx_arr = np.where(predictable_dates == query_date)
            # query date already available
            start_idx = start_idx_arr[0]
            start_date = query_date
            end_date = query_date + pd.DateOffset(months=window_width)
            counts = predict_final_counts(ds, model)
            counts = counts[(start_date <= counts.index) & (counts.index <= end_date)]
        else:
            raise IndexError("Query date out of range")

    if parse_output:
        counts = counts_df_2_dict(counts)

    return counts
