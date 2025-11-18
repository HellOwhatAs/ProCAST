from collections import defaultdict
from functools import reduce
from copy import deepcopy
from pytorch_lightning.callbacks import ModelCheckpoint
from neuralforecast import NeuralForecast
from neuralforecast.common._base_model import BaseModel
import pandas as pd
import more_itertools
import cachier
import json


def _hash_func(args, kwargs: dict):
    assert not args
    models: list[BaseModel] = kwargs.pop("models")
    Y_train_df: pd.DataFrame = kwargs.pop("Y_train_df")
    Y_df: pd.DataFrame = kwargs.pop("Y_df")

    d = {
        "models": [type(model).__name__ for model in models],
        "Y_train_df": Y_train_df.to_json(),
        "Y_df": Y_df.to_json(),
    }
    d.update(kwargs)
    return json.dumps(d)


@cachier.cachier(hash_func=_hash_func)
def fit_and_predict(
    models: list[BaseModel],
    freq: str,
    Y_train_df: pd.DataFrame,
    Y_df: pd.DataFrame,
    n_series: int,
    val_length: int,
    test_length: int,
    output_length: int,
    insample: bool = False,
    _random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    nf = NeuralForecast(models=models, freq=freq)

    nf.fit(df=Y_train_df, val_size=val_length)

    nf.models = [
        type(model).load_from_checkpoint(
            more_itertools.first(
                x
                for x in model.trainer_kwargs["callbacks"]
                if isinstance(x, ModelCheckpoint)
            ).best_model_path,
            early_stop_patience_steps=0,
            enable_checkpointing=False,
            callbacks=[],
        )
        if isinstance(model, BaseModel)
        else model
        for model in nf.models
    ]

    Y_hat_df: pd.DataFrame = pd.concat(
        nf.predict(Y_df[: -((test_length - i) * n_series)])
        for i in range(0, test_length, output_length)
    )

    if insample:
        Y_insample_df = nf.cross_validation(df=Y_train_df)
        return Y_hat_df, Y_insample_df

    return Y_hat_df, Y_train_df


class ResultCollector:
    def __init__(self):
        self.table: defaultdict[str, dict[str, float]] = defaultdict(dict)

    def add(self, model: str, metric: str, value: float):
        self.table[model][metric] = value
        print(f"{model} {metric}:", value)

    def export_typst(self):
        table = deepcopy(self.table)
        models = list(table.keys())
        metrics = sorted(list(reduce(set.union, map(set, table.values()))))

        if len(models) > 1:
            for metric in metrics:
                [m1, m2] = sorted(models, key=lambda model: table[model][metric])[:2]
                table[m1][metric] = f"*{table[m1][metric]}*"
                table[m2][metric] = f"#underline[{table[m2][metric]}]"

        return "\n".join(
            (
                "#table(",
                f"    columns: {len(metrics) + 1},",
                "    [], " + "".join(f"[{i}], " for i in metrics),
                *(
                    f"    [{model}], "
                    + "".join(f"[{table[model][metric]}], " for metric in metrics)
                    for model in models
                ),
                ")",
            )
        )
