import os, yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import collections
from tqdm.auto import tqdm
from typing import Dict, List
from .infer import ForecastModel
from .data import TouristDataset
from .metrices import get_confusion_matrix, get_accuracy, plot_accuracy
from torchinfo import summary


def save_metrices(model, ds, history, path) -> None:
    # confusion matix
    plt.figure()
    cm_display = get_confusion_matrix(model, ds, n_bins=5)
    cm_display.plot()
    plt.savefig(f"{path}/confusion-matrix.jpg")
    plt.close()

    # accuracy
    plot_accuracy(history, f"{path}/accuracy.jpg")


def plot_examples(
    save_dir: str,
    model: ForecastModel,
    train_ds: TouristDataset,
    test_ds: TouristDataset,
    figsize=(10, 10),
):
    normalize_para = train_ds.normalize_para
    countries = train_ds.countries
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(2, 2, figsize=figsize)
    datasets = {"train": train_ds, "test": test_ds}
    for split, ds in datasets.items():
        batch = 0
        for i, country in enumerate(countries):
            ds.set_country(country)
            data_id = np.random.randint(len(ds))
            x, y = ds[data_id]
            date = ds.get_date(data_id)
            col = i % 2
            row = int((i % 4) / 2)
            model.plot(x, y, date, normalize_para, ax[row][col], country)
            if row == 1 and col == 1:
                save_path = f"{save_dir}/{split}-{batch}.jpg"
                batch += 1
                plt.tight_layout()
                fig.savefig(save_path)
                plt.close()
                fig, ax = plt.subplots(2, 2, figsize=figsize)

        if row != 1 or col != 1:
            save_path = f"{save_dir}/{split}-{batch}.jpg"
            batch += 1
            plt.tight_layout()
            fig.savefig(save_path)
            plt.close()
            fig, ax = plt.subplots(2, 2, figsize=figsize)
        ds.set_country(None)
    plt.close()


def plot_history(history: Dict[str, List[float]], path: str) -> None:
    plt.plot(history["test_rmse"], label="Test set")
    plt.plot(history["train_rmse"], label="Train set")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.title("Variation of RMSE over epochs")
    plt.legend()
    plt.grid()
    plt.savefig(f"{path}/history.jpg")
    plt.close()


def save_run(
    save_dir: str,
    name: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.modules.loss._Loss,
    history: Dict[str, List[float]],
    best_epoch: int,
    best_model_weights: collections.OrderedDict,
    best_optimizer_weights: dict,
    final_epoch: int,
    train_ds,
    test_ds,
):
    normalize_para = train_ds.normalize_para
    final_model_weights = model.state_dict()
    final_optimizer_weights = optimizer.state_dict()
    model.load_state_dict(best_model_weights)

    # determine the save location
    if os.path.exists(save_dir):
        available_runs = os.listdir(save_dir)
        same_nms = [
            run
            for run in available_runs
            if run.startswith(name)
            and (run.lstrip(name) == "" or run.lstrip(name).isnumeric())
        ]
        if name in same_nms:
            name_id = same_nms.index(name)
            same_nms[name_id] = name + "1"

        if len(same_nms) == 0:
            save_path = f"{save_dir}/{name}"
        else:
            latest_run_num = max([int(nm.lstrip(name)) for nm in same_nms])
            this_run_name = name + str(latest_run_num + 1)
            save_path = f"{save_dir}/{this_run_name}"
    else:
        save_path = f"{save_dir}/{name}"
    os.makedirs(save_path)

    # save
    # Save history
    plot_history(history, save_path)

    ### plot examples
    plot_examples(f"{save_path}/results", model, train_ds, test_ds)

    ### Save metrices
    save_metrices(model, test_ds, history, save_path)

    ### Checkpoints
    weight_dir = f"{save_path}/checkpoints"
    os.makedirs(weight_dir)
    save_weights = {
        "best_model": best_model_weights,
        "best_optimizer": best_optimizer_weights,
        "final_model": final_model_weights,
        "final_optimizer": final_optimizer_weights,
    }
    for name, weights in save_weights.items():
        torch.save(weights, f"{weight_dir}/{name}.pt")

    ### History
    history_path = f"{save_path}/history.csv"
    history["epoch"] = (np.arange(len(history["train_rmse"])) + 1).tolist()
    history = pd.DataFrame(history)
    history.to_csv(history_path, index=False)

    ### Training details
    save_details_path = f"{save_path}/details"
    param_count = summary(model).trainable_params
    save_details = [
        f"Model: {type(model).__name__}",
        f"Model layers: {model.num_layers}",
        f"Input feature count: {model.inp_feature_count}",
        f"Output feature count: {model.out_feature_count}",
        f"Input sequence length: {model.inp_seq_len}",
        f"Output sequence length: {model.out_seq_len}",
        f"Hidden feature count: {model.hid_feature_count}",
        f"Optimizer: {type(optimizer).__name__}",
        f"Loss function: {type(loss_fn).__name__}",
        f"Best epoch: {best_epoch}",
        f"Final epoch: {final_epoch}",
        f"Parameter count: {param_count}",
    ]
    save_details = "\n".join(save_details)
    with open(f"{save_details_path}.txt", "w") as handler:
        handler.write(save_details)
    new_normalize_para = {}
    new_normalize_para["scale"] = normalize_para["scale"].to_dict()
    new_normalize_para["offset"] = normalize_para["offset"].to_dict()
    save_details = {
        "model": type(model).__name__,
        "num_layers": model.num_layers,
        "inp_feature_count": model.inp_feature_count,
        "out_feature_count": model.out_feature_count,
        "inp_features": model.inp_features,
        "out_features": model.out_features,
        "inp_seq_len": model.inp_seq_len,
        "out_seq_len": model.out_seq_len,
        "hid_feature_count": model.hid_feature_count,
        "optimizer": type(optimizer).__name__,
        "loss_function": type(loss_fn).__name__,
        "best_epoch": best_epoch,
        "final_epoch": final_epoch,
        "parameter_count": param_count,
        "data": {"normalize_para": new_normalize_para},
    }
    with open(f"{save_details_path}.yaml", "w") as handler:
        yaml.dump(save_details, handler)

    ### Model summary
    model_summary = str(summary(model))
    with open(f"{save_path}/model-summary.txt", "w") as handler:
        handler.write(model_summary)

    print(f"---- Artifacts saved at {os.path.abspath(save_path)} ----")


def start(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.modules.loss._Loss,
    train_dl: torch.utils.data.DataLoader,
    test_dl: torch.utils.data.DataLoader,
    n_epochs: int,
    tollerance: int = 50,
    save_dir="./runs",
    name="forecast",
):
    if name is None:
        raise AttributeError("name should not be None")

    history = {
        "train_rmse": [],
        "test_rmse": [],
        "train_accuracy": [],
        "test_accuracy": [],
    }
    best_rmse = np.inf
    best_model_weights = None
    best_optimizer_weights = None
    best_epoch = 0

    count_scale = train_dl.dataset.normalize_para["scale"]["count"]

    # train
    with tqdm(total=n_epochs) as pbar:
        for epoch in range(n_epochs):
            # train loop
            model.train()
            train_rmse_glob = 0
            batch_count = 0
            sample_count = 0
            train_acc = 0
            for X_batch, y_batch in train_dl:
                y_pred = model(X_batch)
                train_rmse = loss_fn(y_pred, y_batch)
                optimizer.zero_grad()
                train_rmse.backward()
                optimizer.step()
                y_pred, y_batch = y_pred.detach().numpy(), y_batch.detach().numpy()
                train_rmse_glob += train_rmse.detach().numpy().item()
                batch_count += 1
                sample_count += y_batch.size
                train_acc += get_accuracy(y_pred, y_batch, count_scale).sum()
            train_rmse_glob = np.sqrt(train_rmse_glob / batch_count)
            train_acc /= sample_count

            # validation loop
            model.eval()
            test_rmse_glob = 0
            batch_count = 0
            sample_count = 0
            test_acc = 0
            with torch.no_grad():
                for X_batch, y_batch in test_dl:
                    y_pred = model(X_batch)
                    test_rmse = loss_fn(y_pred, y_batch)
                    y_pred, y_batch = y_pred.numpy(), y_batch.numpy()
                    test_rmse_glob += test_rmse.detach().numpy().item()
                    batch_count += 1
                    sample_count += y_batch.size
                    test_acc += get_accuracy(y_pred, y_batch, count_scale).sum()
                test_rmse_glob = np.sqrt(test_rmse_glob / batch_count)
                test_acc /= sample_count

                if test_rmse_glob < best_rmse:
                    best_rmse = test_rmse_glob
                    best_model_weights = model.state_dict()
                    best_optimizer_weights = optimizer.state_dict()
                    best_epoch = epoch

            history["train_rmse"].append(train_rmse_glob)
            history["test_rmse"].append(test_rmse_glob)
            history["train_accuracy"].append(train_acc)
            history["test_accuracy"].append(test_acc)

            pbar.update(1)
            pbar.set_postfix(
                {"train_rmse": train_rmse_glob, "test_rmse": test_rmse_glob}
            )

            if epoch - best_epoch > tollerance:
                print(
                    f"---- Stoping early at epoch {epoch}. Best test set RMSE observed at epoch {best_epoch} ----"
                )
                break

    final_epoch = epoch
    save_run(
        save_dir,
        name,
        model,
        optimizer,
        loss_fn,
        history,
        best_epoch,
        best_model_weights,
        best_optimizer_weights,
        final_epoch,
        train_dl.dataset,
        test_dl.dataset,
    )
