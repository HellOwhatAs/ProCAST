from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import darts.dataprocessing.transformers
import darts.metrics
import numpy as np
import typing
import neuralforecast2
import neuralforecast.models
import neuralforecast.common._base_model

random_state = 42
random_states = [42, 1, 2, 3, 4]
input_length = 24
output_length = 12
support_length = 72
val_length = 36
test_length = 72
max_epochs = 500
patience = 50

model_encoders = {
    "datetime_attribute": {
        "past": ["month", "year"],
        "future": ["month", "year"],
    },
    "transformer": darts.dataprocessing.transformers.Scaler(),
}

regression_model_kwargs = dict(
    lags=input_length,
    random_state=random_state,
    add_encoders=model_encoders,
)

darts_torch_model_kwargs = dict(
    input_chunk_length=input_length,
    output_chunk_length=output_length,
    random_state=random_state,
    add_encoders=model_encoders,
    n_epochs=max_epochs,
    pl_trainer_kwargs={
        "callbacks": [
            EarlyStopping(
                monitor="val_loss",
                patience=patience,
                mode="min",
            ),
        ],
    },
)


nf_models_cls: list[type[neuralforecast.common._base_model.BaseModel]] = [
    neuralforecast.models.NBEATS,
    neuralforecast.models.NBEATSx,
    neuralforecast.models.NHITS,
    neuralforecast.models.TimesNet,
    neuralforecast.models.TCN,
    neuralforecast.models.BiTCN,
    neuralforecast.models.DeepNPTS,
    neuralforecast.models.TFT,
    neuralforecast.models.TiDE,
    neuralforecast.models.DLinear,
    neuralforecast.models.Informer,
    neuralforecast.models.Autoformer,
    neuralforecast.models.FEDformer,
    neuralforecast2.PatchTST,
    neuralforecast2.TimeXer,
    neuralforecast.models.TimeMixer,
    neuralforecast.models.TSMixer,
    neuralforecast.models.TSMixerx,
    neuralforecast.models.iTransformer,
    neuralforecast.models.RMoK,
    neuralforecast.models.SOFTS,
    neuralforecast.models.StemGNN,
]


nf_model_kwargs = dict(
    h=output_length,
    input_size=input_length,
    val_check_steps=1,
    early_stop_patience_steps=patience,
    random_seed=random_state,
    max_steps=max_epochs,
)


class Metric(typing.Protocol):
    def __call__(
        self, ts1: darts.TimeSeries, ts2: darts.TimeSeries, *args, **kwargs
    ) -> float: ...


def try_metric(metric, epsilon: float = 1.0) -> Metric:
    def zero2epsilon(x: np.ndarray) -> np.ndarray:
        return np.where(x == 0, epsilon, x)

    def metric_func(ts1: darts.TimeSeries, ts2: darts.TimeSeries, *args, **kwargs):
        if "mape" in metric.__name__:
            ts1 = ts1.map(zero2epsilon)
            ts2 = ts2.map(zero2epsilon)
        try:
            return metric(ts1, ts2, *args, **kwargs)
        except Exception:
            return float("inf")

    metric_func.__name__ = metric.__name__

    return metric_func


metric_list: list[Metric] = list(
    map(
        try_metric,
        (
            darts.metrics.mae,
            darts.metrics.mse,
            darts.metrics.rmse,
        ),
    )
)
