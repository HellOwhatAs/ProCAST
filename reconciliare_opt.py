from config import support_length
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from typing import Optional, Callable, Protocol
import pytntnn


type PinvCallback = Callable[[np.ndarray | None], None]


class ProjectFunc(Protocol):
    def __call__(
        self,
        mat: torch.Tensor | np.ndarray,
        df_pred: pd.DataFrame,
        df_true: pd.DataFrame = None,
        h: int = None,
        **kwargs,
    ) -> pd.DataFrame: ...


class _TorchPinv(torch.nn.Module):
    def __init__(
        self,
        mat: torch.Tensor,
        softmax_constraint: bool = False,
        masked: bool = True,
        callback: Optional[PinvCallback] = None,
        max_iter: int = 2000,
        patience: int = 100,
        eps: float = 1e-4,
    ) -> None:
        super().__init__()
        self.mat_original = torch.nn.Parameter(mat.T, requires_grad=False)
        self.eye_m = torch.nn.Parameter(torch.eye(mat.shape[1]), requires_grad=False)
        self.pinv_params = torch.nn.Parameter(torch.pinverse(self.mat_original))
        self.softmax_constraint = softmax_constraint
        self.masked = masked
        self.callback = callback
        self.max_iter = max_iter
        self.patience = patience
        self.eps = eps

    @property
    def mat(self) -> torch.Tensor:
        return (
            (self.mat_original / self.mat_original.sum(axis=0))
            if self.softmax_constraint
            else self.mat_original
        )

    @property
    def pinv(self) -> torch.Tensor:
        pinv = (
            torch.where(
                self.mat_original.T == 0,
                (-torch.inf if self.softmax_constraint else 0),
                self.pinv_params,
            )
            if self.masked
            else self.pinv_params
        )
        if self.softmax_constraint:
            return pinv.softmax(dim=1)
        return pinv

    def fit(
        self,
        y_hat: torch.Tensor,
        y_true: torch.Tensor,
        weights: torch.Tensor = None,
    ):
        y_hat = y_hat.unsqueeze(dim=1)
        y_true = y_true.unsqueeze(dim=1)
        optimizer = torch.optim.Adam([self.pinv_params], lr=1e-3, weight_decay=1e-2)
        min_loss, count, total_count = torch.inf, 0, 0
        for _ in range(self.max_iter):
            loss = (
                (((y_hat @ self.pinv).clamp(min=0) @ self.mat) - y_true)
                .square()
                .sum(dim=(1, 2))
                * (weights if weights is not None else 1)
            ).sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            delta_loss = 2 * (min_loss - loss.item()) / (loss.item() + min_loss)
            if delta_loss > self.eps:
                count = 0
            else:
                count += 1
            if count > self.patience:
                break
            min_loss: float = min(min_loss, loss.item())
            total_count += 1

        if self.callback is not None:
            self.callback(("count", total_count))

    def predict(self, products: torch.Tensor) -> torch.Tensor:
        if self.callback is not None:
            self.callback(("pinv", self.pinv.detach().cpu().numpy()))

        with torch.no_grad():
            return ((products @ self.pinv).clamp(min=0) @ self.mat).squeeze()


def torch_oblique(
    mat: torch.Tensor | np.ndarray,
    df_pred: pd.DataFrame,
    df_true: pd.DataFrame,
    h: int = support_length,
    softmax_constraint: bool = False,
    masked_pinv: bool = False,
    orthogonal_today: bool = True,
    pinv_callback: Optional[PinvCallback] = None,
    orthogonal_lambda: float = 0.5,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    dtype: torch.dtype = torch.float64,
) -> pd.DataFrame:
    model = _TorchPinv(
        mat=torch.tensor(mat),
        softmax_constraint=softmax_constraint,
        masked=masked_pinv,
        callback=pinv_callback,
        max_iter=10**5,
    ).to(device=device, dtype=dtype)
    x_hat = torch.from_numpy(df_pred.to_numpy()).to(device=device, dtype=dtype)
    x_true = torch.from_numpy(df_true.to_numpy()).to(device=device, dtype=dtype)

    x_hat_: list[torch.Tensor] = []
    for i in tqdm(range(h + 1, x_hat.shape[0])):
        if orthogonal_today:
            weights = torch.tensor(
                [(1 - orthogonal_lambda) / h] * h + [orthogonal_lambda],
                device=device,
                dtype=dtype,
            )
            model.fit(
                x_hat[i - h : i + 1],
                torch.cat((x_true[i - h : i], x_hat[i : i + 1])),
                weights=weights,
            )
        else:
            model.fit(x_hat[i - h : i], x_true[i - h : i])
        x_hat_.append(model.predict(x_hat[i : i + 1]))
    res = torch.stack(x_hat_)
    df_pred[-res.shape[0] :] = (
        res.detach().cpu().numpy().astype(df_pred.to_numpy().dtype)
    )
    if pinv_callback is not None:
        pinv_callback(None)
    return df_pred


def tntnn_orthogonal(
    mat: torch.Tensor | np.ndarray,
    df_pred: pd.DataFrame,
    df_true: pd.DataFrame = None,
    h: int = None,
):
    del df_true  # orthogonal projector does not need df_true
    assert h is None or h == 0, "tntnn does not support h > 0"

    if isinstance(mat, torch.Tensor):
        mat = mat.detach().cpu().numpy()
    for i in range(len(df_pred)):
        try:
            df_pred.iloc[i] = (pytntnn.tntnn(mat, df_pred.iloc[i]).x @ mat.T).astype(
                df_pred.iloc[i].dtype
            )
        except ValueError:
            df_pred.iloc[i] *= 0

    return df_pred
