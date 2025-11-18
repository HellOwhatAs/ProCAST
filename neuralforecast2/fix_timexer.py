import neuralforecast.models


class TimeXer(neuralforecast.models.TimeXer):
    def __init__(self, *args, **kwargs):
        if "exclude_insample_y" in kwargs:
            assert kwargs.pop("exclude_insample_y") is False
        super().__init__(*args, **kwargs)
