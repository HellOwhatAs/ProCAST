from pathlib import Path
import json
import pandas as pd
from collections import defaultdict
import more_itertools
from functools import cached_property, reduce
from operator import or_
from abc import ABC, abstractmethod


class MyDataset(ABC):
    @abstractmethod
    def __init__(self, path: str): ...

    @abstractmethod
    def to_dataframe(self, use_columns: bool = True) -> pd.DataFrame: ...

    @abstractmethod
    def to_unique_dataframe(self, use_columns: bool = True) -> pd.DataFrame: ...

    @property
    @abstractmethod
    def freq(self) -> str: ...

    @property
    @abstractmethod
    def columns(self) -> list[str]: ...

    @property
    @abstractmethod
    def uniques(self) -> list[str]: ...

    @abstractmethod
    def mat(self, column: str, unique: str) -> bool: ...


class ECommerce(MyDataset):
    freq = "B"

    def __init__(self, path: str = "./datasets/e-commerce/data.csv"):
        self.path = Path(path)
        self.df = pd.read_csv(
            path,
            encoding="latin1",
            date_format=r"%m/%d/%Y %H:%M",
            parse_dates=["InvoiceDate"],
            converters={
                "InvoiceNo": str,
                "StockCode": str,
                "Description": str,
                "Quantity": int,
                "UnitPrice": float,
                "CustomerID": str,
                "Country": str,
            },
        )

        # filter df with Quantity > 0
        self.df = self.df[self.df["Quantity"] > 0]

    @cached_property
    def category(self):
        with open(self.path.parent.joinpath("category.json"), "rb") as f:
            category: dict[str, list[str]] = json.load(f)
        return category

    @cached_property
    def id2quant(self):
        id2quant: dict[str, int] = {
            tup[0]: tup[2] for tup in self.stock_code.itertuples(index=False)
        }
        return id2quant

    @cached_property
    def tag2quant(self):
        tag2quant: defaultdict[str, defaultdict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        for id, tags in self.category.items():
            for tag in tags:
                tag2quant[tag][id] += self.id2quant[id]

        rev_map = defaultdict(list)
        for tag, quants in tag2quant.items():
            rev_map[tuple(sorted(quants.keys()))].append(tag)

        for tags in rev_map.values():
            if len(tags) == 1:
                continue
            for tag in tags[1:]:
                for k, v in tag2quant.pop(tag).items():
                    tag2quant[tags[0]][k] += v

        return tag2quant

    @cached_property
    def structured_categories(self):
        tags = sorted(
            self.tag2quant.keys(),
            key=lambda k: sum(self.tag2quant[k].values()),
            reverse=True,
        )

        stocks = sorted(
            more_itertools.flatten([self.tag2quant[tag].items() for tag in tags]),
            key=lambda it: it[1],
            reverse=True,
        )

        stock2unique: dict[str, int] = {
            id: reduce(
                or_,
                (1 << j for j, tag in enumerate(tags) if id in self.tag2quant[tag]),
            )
            for id, _ in stocks
        }

        return tags, stocks, stock2unique

    @cached_property
    def stock_code(self):
        # build stock_code
        stock_code = self.df[["StockCode", "Description", "Quantity"]].copy()

        # get most used Description for each StockCode
        counter = defaultdict(lambda: defaultdict(int))
        for tup in stock_code.itertuples(index=False):
            counter[tup[0]][tup[1]] += tup[2]

        stock2desc = {
            k: max(v.items(), key=lambda item: item[1] if item[0] != "" else 0)[0]
            for k, v in counter.items()
        }

        # use Description for StockCode
        stock_code["Description"] = stock_code["StockCode"].apply(stock2desc.get)

        # aggregate
        stock_code = (
            stock_code.groupby(["StockCode", "Description"])
            .agg({"Quantity": "sum"})
            .reset_index()
        )

        return stock_code

    def to_dataframe(self, use_columns: bool = True) -> pd.DataFrame:
        tags, _, _ = self.structured_categories
        tags_set = set(tags)
        stock2tags = defaultdict(list)
        for tag, ids in self.tag2quant.items():
            if tag not in tags_set:
                continue
            for id in ids:
                stock2tags[id].append(tag)

        timeseries_df = self.__to_stock_dataframe(use_columns=False)
        timeseries_df["unique_id"] = timeseries_df["unique_id"].apply(stock2tags.get)
        timeseries_df = (
            timeseries_df.explode("unique_id")
            .groupby(["ds", "unique_id"])
            .agg({"y": "sum"})
            .reset_index()
        )
        if use_columns:
            return timeseries_df.pivot(index="ds", columns="unique_id", values="y")

        return timeseries_df

    def to_unique_dataframe(self, use_columns: bool = True) -> pd.DataFrame:
        _, _, stock2unique = self.structured_categories
        timeseries_df = self.__to_stock_dataframe(use_columns=False)
        timeseries_df["unique_id"] = (
            timeseries_df["unique_id"].apply(stock2unique.get).apply(str)
        )
        timeseries_df = (
            timeseries_df.explode("unique_id")
            .groupby(["ds", "unique_id"])
            .agg({"y": "sum"})
            .reset_index()
        )
        if use_columns:
            return timeseries_df.pivot(index="ds", columns="unique_id", values="y")

        return timeseries_df

    def __to_stock_dataframe(self, use_columns: bool = True):
        _, stocks, _ = self.structured_categories
        timeseries_df = self.df[
            ["InvoiceNo", "StockCode", "Quantity", "InvoiceDate"]
        ].copy()
        timeseries_df["InvoiceDate"] = timeseries_df["InvoiceDate"].dt.floor("D")

        timeseries_df = timeseries_df[
            timeseries_df["StockCode"].isin(set(i for i, _ in stocks))
        ]

        timeseries_df = (
            timeseries_df.groupby(["InvoiceDate", "StockCode"])
            .agg({"Quantity": "sum"})
            .reset_index()
        )

        timeseries_df = timeseries_df.pivot(
            index="InvoiceDate", columns="StockCode", values="Quantity"
        ).fillna(0)
        timeseries_df = timeseries_df.asfreq(self.freq, fill_value=0)

        if not use_columns:
            timeseries_df = (
                timeseries_df.melt(
                    var_name="unique_id", value_name="y", ignore_index=False
                )
                .reset_index()
                .rename(columns={"InvoiceDate": "ds"})
            )

        return timeseries_df

    def stock2desc_quant_limit(self, quant_limit: int = 1000):
        stock_code = self.stock_code[self.stock_code["Quantity"] >= quant_limit]
        return {
            tup[0]: str.lower(tup[1]).strip()
            for tup in stock_code.itertuples(index=False)
        }

    @cached_property
    def columns(self) -> list[str]:
        tags, _, _ = self.structured_categories
        return tags

    @cached_property
    def uniques(self) -> list[str]:
        _, _, stock2unique = self.structured_categories
        return list(map(str, set(stock2unique.values())))

    @cached_property
    def _cached_mat(self):
        _, _, stock2unique = self.structured_categories
        return set(
            (tag, str(stock2unique[id]))
            for tag in self.columns
            for id in self.tag2quant[tag]
        )

    def mat(self, column: str, unique: str) -> bool:
        return (column, unique) in self._cached_mat


class RH(MyDataset):
    freq = "D"

    def __init__(self, path: str = "./datasets/RH/data.pkl"):
        import pickle

        with open(path, "rb") as f:
            self.data = pickle.load(f)

    def to_dataframe(self, use_columns: bool = True) -> pd.DataFrame:
        return self.data["to_dataframe"][use_columns]

    def to_unique_dataframe(self, use_columns: bool = True) -> pd.DataFrame:
        raise NotImplementedError("not available")

    @cached_property
    def columns(self) -> list[str]:
        return self.data["columns"]

    @cached_property
    def uniques(self) -> list[str]:
        return self.data["uniques"]

    def mat(self, column: str, unique: str) -> bool:
        return self.data["mat"][(column, unique)]


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from darts import TimeSeries

    datasets: list[MyDataset] = [
        ECommerce(),
        RH(),
    ]
    for ec in datasets:
        ts = TimeSeries.from_dataframe(ec.to_dataframe())
        ts.plot(new_plot=True, max_nr_components=-1)

    plt.show()
