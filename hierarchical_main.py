import pandas as pd
from pytorch_lightning.callbacks import ModelCheckpoint
from neuralforecast.common._base_model import BaseModel
import build_dataset
import config
from utils import fit_and_predict
import cachier


def __calculate_hash(args: tuple, kwargs: dict):
    arg_names = [
        ("dataset", lambda x: type(x).__name__),
        ("models_cls", lambda x: tuple(i.__name__ for i in x)),
        ("output_length", lambda x: x),
        ("val_length", lambda x: x),
        ("test_length", lambda x: x),
        ("model_kwargs", lambda x: tuple(sorted(dict.items(x)))),
    ]

    i = 0
    for i, arg in enumerate(args):
        arg_names[i] = (arg_names[i][0], arg_names[i][1](arg))
    for j in range(i, len(arg_names)):
        arg_names[j] = (arg_names[j][0], arg_names[j][1](kwargs[arg_names[j][0]]))

    return tuple(arg_names)


def aggregate_dfs_stats(dfs: list[pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not dfs:
        raise ValueError

    cols = dfs[0].columns
    for df in dfs[1:]:
        if not df.columns.equals(cols):
            raise ValueError

    id_col = cols[0]
    value_cols = cols[1:]

    arrs = np.stack([df[value_cols].to_numpy() for df in dfs], axis=0)

    mean_arr = np.mean(arrs, axis=0)
    std_arr = np.std(arrs, axis=0, ddof=1)

    mean_df = pd.DataFrame(mean_arr, columns=value_cols)
    mean_df.insert(0, id_col, dfs[0][id_col].values)

    std_df = pd.DataFrame(std_arr, columns=value_cols)
    std_df.insert(0, id_col, dfs[0][id_col].values)

    return mean_df, std_df


@cachier.cachier(hash_func=__calculate_hash)
def hierarchical_main(
    dataset: build_dataset.MyDataset,
    models_cls: list[type[BaseModel]] = config.nf_models_cls,
    output_length: int = config.output_length,
    val_length: int = config.val_length,
    test_length: int = config.test_length,
    model_kwargs: dict[str, int] = config.nf_model_kwargs,
):
    Y_df = pd.concat(
        [
            dataset.to_dataframe(use_columns=False),
            dataset.to_unique_dataframe(use_columns=False),
        ],
        ignore_index=True,
    ).sort_values("ds")
    n_series = Y_df["unique_id"].unique().size

    Y_train_df: pd.DataFrame = Y_df[: -test_length * n_series]
    Y_test_df = Y_df[-test_length * n_series :]

    models = [
        cls(
            **model_kwargs,
            n_series=n_series,
            enable_checkpointing=True,
            callbacks=[ModelCheckpoint(monitor="ptl/val_loss")],
        )
        for cls in models_cls
    ]

    Y_hat_df: pd.DataFrame
    Y_hat_df, Y_train_df = fit_and_predict(
        models=models,
        freq=dataset.freq,
        Y_train_df=Y_train_df,
        Y_df=Y_df,
        n_series=n_series,
        val_length=val_length,
        test_length=test_length,
        output_length=output_length,
        insample=True,
    )

    return Y_train_df, Y_test_df, Y_hat_df


if __name__ == "__main__":
    from hierarchicalforecast.core import (
        HierarchicalReconciliation,
        HReconciler,
        _build_fn_name,
    )
    from darts import TimeSeries
    from hierarchicalforecast.methods import BottomUp, MinTrace
    import numpy as np

    metric_list = config.metric_list

    dataset = build_dataset.ECommerce()
    S_df = pd.DataFrame(
        {
            "unique_id": [*dataset.columns, *dataset.uniques],
            **{
                u: [float(dataset.mat(c, u)) for c in dataset.columns]
                + [float(j == i) for j in range(len(dataset.uniques))]
                for i, u in enumerate(dataset.uniques)
            },
        }
    )
    tags: dict[str, np.ndarray] = {
        "columns": np.array(dataset.columns),
        "uniques": np.array(dataset.uniques),
    }

    dfs: list[pd.DataFrame] = []
    for random_seed in config.random_states:
        Y_train_df, Y_test_df, Y_hat_df = hierarchical_main(
            dataset=dataset,
            model_kwargs={**config.nf_model_kwargs, "random_seed": random_seed},
        )

        reconcilers: list[HReconciler] = [
            BottomUp(),
            MinTrace(method="ols", nonnegative=True),
            MinTrace(method="wls_struct", nonnegative=True),
            MinTrace(method="wls_var", nonnegative=True),
        ]
        hrec = HierarchicalReconciliation(reconcilers=reconcilers)
        Y_rec_df: pd.DataFrame = hrec.reconcile(
            Y_hat_df=Y_hat_df, Y_df=Y_train_df, S=S_df, tags=tags
        )

        test = TimeSeries.from_dataframe(
            Y_test_df.pivot(index="ds", columns="unique_id", values="y")[
                dataset.columns
            ]
        )

        data = {
            "model": [],
            **{
                f"{rec_name}-{metric.__name__}": []
                for metric in metric_list
                for rec_name in ("raw", *map(_build_fn_name, hrec.orig_reconcilers))
            },
        }

        for model_name in Y_hat_df.columns[2:]:
            data["model"].append(model_name)

            pred = TimeSeries.from_dataframe(
                Y_hat_df.pivot(index="ds", columns="unique_id", values=model_name)[
                    dataset.columns
                ]
            )

            for metric in metric_list:
                data[f"raw-{metric.__name__}"].append(metric(test, pred))

            for rec_name in map(_build_fn_name, hrec.orig_reconcilers):
                rec = TimeSeries.from_dataframe(
                    Y_rec_df.pivot(
                        index="ds",
                        columns="unique_id",
                        values=f"{model_name}/{rec_name}",
                    )[dataset.columns]
                )

                for metric in metric_list:
                    data[f"{rec_name}-{metric.__name__}"].append(metric(test, rec))

        dfs.append(
            pd.DataFrame(data)[
                [
                    "model",
                    *(
                        f"{rec_name}-{metric}"
                        for metric in ["mae", "mse", "rmse"]
                        for rec_name in (
                            "raw",
                            *map(_build_fn_name, hrec.orig_reconcilers),
                        )
                    ),
                ]
            ]
        )

    mean_df, std_df = aggregate_dfs_stats(dfs)
    mean_df.to_csv(f"{type(dataset).__name__}-hierarchical-mean.csv", index=False)
    std_df.to_csv(f"{type(dataset).__name__}-hierarchical-std.csv", index=False)
