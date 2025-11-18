import torch
import pandas as pd
from pytorch_lightning.callbacks import ModelCheckpoint
from neuralforecast.common._base_model import BaseModel
import utilsforecast.processing as ufp
from darts import TimeSeries
import build_dataset
import config
import reconciliare_opt as ropt
from utils import ResultCollector, fit_and_predict


def main(
    dataset: build_dataset.MyDataset,
    method: ropt.ProjectFunc = ropt.torch_oblique,
    models_cls: list[type[BaseModel]] = config.nf_models_cls,
    support_lengths: int | list[int] = config.support_length,
    metric_list: list[config.Metric] = config.metric_list,
    output_length: int = config.output_length,
    val_length: int = config.val_length,
    test_length: int = config.test_length,
    model_kwargs: dict[str, int] = config.nf_model_kwargs,
) -> ResultCollector:
    Y_df = dataset.to_dataframe(use_columns=False).sort_values("ds")
    n_series = len(dataset.columns)

    uids: list[str] = ufp.counts_by_id(Y_df, "unique_id")["unique_id"].tolist()
    mat = torch.zeros((len(dataset.columns), len(dataset.uniques)))
    for tidx, tag in enumerate(uids):
        for uidx, unique in enumerate(dataset.uniques):
            mat[tidx, uidx] = dataset.mat(tag, unique)

    Y_train_df = Y_df[: -test_length * n_series]
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
    Y_hat_df, _ = fit_and_predict(
        models=models,
        freq=dataset.freq,
        Y_train_df=Y_train_df,
        Y_df=Y_df,
        n_series=n_series,
        val_length=val_length,
        test_length=test_length,
        output_length=output_length,
        _random_state=model_kwargs.get("random_seed", 42),
    )

    test = TimeSeries.from_dataframe(
        Y_test_df.pivot(index="ds", columns="unique_id", values="y")
    )
    train = TimeSeries.from_dataframe(
        Y_train_df.pivot(index="ds", columns="unique_id", values="y")
    )

    res = ResultCollector()
    for model in map(type, models):
        pred = TimeSeries.from_dataframe(
            Y_hat_df.pivot(index="ds", columns="unique_id", values=model.__name__)
        )

        for metric in metric_list:
            res.add(model.__name__, metric.__name__, metric(test, pred))

        if isinstance(support_lengths, int):
            support_lengths = [support_lengths]
        for support_length in support_lengths:
            df_pred: pd.DataFrame = train.concatenate(pred)[
                -(support_length + len(pred)) :
            ].to_dataframe()[uids]
            df_true: pd.DataFrame = train.concatenate(test)[
                -(support_length + len(test)) :
            ].to_dataframe()[uids]

            new_df = method(mat, df_pred=df_pred, df_true=df_true, h=support_length)
            new_pred = TimeSeries.from_dataframe(new_df)[-len(pred) :]

            for metric in metric_list:
                res.add(
                    model.__name__ + f"-support_length={support_length}",
                    metric.__name__,
                    metric(test, new_pred),
                )

    return res


if __name__ == "__main__":
    import build_dataset
    import shelve
    import neuralforecast.models

    with shelve.open("nf_main_results.db") as db:
        for dataset in [build_dataset.RH(), build_dataset.ECommerce()]:
            for random_state in config.random_states:
                for method_name, method, kwargs in [
                    ("ortho", ropt.tntnn_orthogonal, dict(support_lengths=0)),
                    ("obliq", ropt.torch_oblique, {}),
                ]:
                    res = main(
                        dataset=dataset,
                        method=method,
                        models_cls=[neuralforecast.models.PatchTST],
                        model_kwargs={
                            **config.nf_model_kwargs,
                            "random_seed": random_state,
                        },
                        **kwargs,
                    )
                    result = res.export_typst()
                    print(result)
                    db[f"{type(dataset).__name__}_{method_name}_{random_state}"] = res
