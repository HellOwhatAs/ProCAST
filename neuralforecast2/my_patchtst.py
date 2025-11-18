from neuralforecast.common._base_model import BaseModel
from neuralforecast.models.patchtst import PatchTST_backbone
from neuralforecast.losses.pytorch import MAE
from typing import Optional
import torch


class PatchTST(BaseModel):
    # Class attributes
    EXOGENOUS_FUTR = False
    EXOGENOUS_HIST = False
    EXOGENOUS_STAT = False
    MULTIVARIATE = True  # If the model produces multivariate forecasts (True) or univariate (False)
    RECURRENT = (
        False  # If the model produces forecasts recursively (True) or direct (False)
    )

    def __init__(
        self,
        h,
        input_size,
        n_series,
        stat_exog_list=None,
        hist_exog_list=None,
        futr_exog_list=None,
        exclude_insample_y=False,
        encoder_layers: int = 3,
        n_heads: int = 16,
        hidden_size: int = 128,
        linear_hidden_size: int = 256,
        dropout: float = 0.2,
        fc_dropout: float = 0.2,
        head_dropout: float = 0.0,
        attn_dropout: float = 0.0,
        patch_len: int = 16,
        stride: int = 8,
        revin: bool = True,
        revin_affine: bool = False,
        revin_subtract_last: bool = True,
        activation: str = "gelu",
        res_attention: bool = True,
        batch_normalization: bool = False,
        learn_pos_embed: bool = True,
        loss=MAE(),
        valid_loss=None,
        max_steps: int = 5000,
        learning_rate: float = 1e-4,
        num_lr_decays: int = -1,
        early_stop_patience_steps: int = -1,
        val_check_steps: int = 100,
        batch_size: int = 32,
        valid_batch_size: Optional[int] = None,
        windows_batch_size=1024,
        inference_windows_batch_size: int = 1024,
        start_padding_enabled=False,
        step_size: int = 1,
        scaler_type: str = "identity",
        random_seed: int = 1,
        drop_last_loader: bool = False,
        alias: Optional[str] = None,
        optimizer=None,
        optimizer_kwargs=None,
        lr_scheduler=None,
        lr_scheduler_kwargs=None,
        dataloader_kwargs=None,
        **trainer_kwargs,
    ):
        super(PatchTST, self).__init__(
            h=h,
            input_size=input_size,
            n_series=n_series,
            stat_exog_list=stat_exog_list,
            hist_exog_list=hist_exog_list,
            futr_exog_list=futr_exog_list,
            exclude_insample_y=exclude_insample_y,
            loss=loss,
            valid_loss=valid_loss,
            max_steps=max_steps,
            learning_rate=learning_rate,
            num_lr_decays=num_lr_decays,
            early_stop_patience_steps=early_stop_patience_steps,
            val_check_steps=val_check_steps,
            batch_size=batch_size,
            valid_batch_size=valid_batch_size,
            windows_batch_size=windows_batch_size,
            inference_windows_batch_size=inference_windows_batch_size,
            start_padding_enabled=start_padding_enabled,
            step_size=step_size,
            scaler_type=scaler_type,
            random_seed=random_seed,
            drop_last_loader=drop_last_loader,
            alias=alias,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler=lr_scheduler,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            dataloader_kwargs=dataloader_kwargs,
            **trainer_kwargs,
        )

        # Enforce correct patch_len, regardless of user input
        patch_len = min(input_size + stride, patch_len)

        c_out = self.loss.outputsize_multiplier

        # Fixed hyperparameters
        c_in = n_series  # Always univariate
        padding_patch = "end"  # Padding at the end
        pretrain_head = False  # No pretrained head
        norm = "BatchNorm"  # Use BatchNorm (if batch_normalization is True)
        pe = "zeros"  # Initial zeros for positional encoding
        d_k = None  # Key dimension
        d_v = None  # Value dimension
        store_attn = False  # Store attention weights
        head_type = "flatten"  # Head type
        individual = False  # Separate heads for each time series
        max_seq_len = 1024  # Not used
        key_padding_mask = "auto"  # Not used
        padding_var = None  # Not used
        attn_mask = None  # Not used

        self.model = PatchTST_backbone(
            c_in=c_in,
            c_out=c_out,
            input_size=input_size,
            h=h,
            patch_len=patch_len,
            stride=stride,
            max_seq_len=max_seq_len,
            n_layers=encoder_layers,
            hidden_size=hidden_size,
            n_heads=n_heads,
            d_k=d_k,
            d_v=d_v,
            linear_hidden_size=linear_hidden_size,
            norm=norm,
            attn_dropout=attn_dropout,
            dropout=dropout,
            act=activation,
            key_padding_mask=key_padding_mask,
            padding_var=padding_var,
            attn_mask=attn_mask,
            res_attention=res_attention,
            pre_norm=batch_normalization,
            store_attn=store_attn,
            pe=pe,
            learn_pe=learn_pos_embed,
            fc_dropout=fc_dropout,
            head_dropout=head_dropout,
            padding_patch=padding_patch,
            pretrain_head=pretrain_head,
            head_type=head_type,
            individual=individual,
            revin=revin,
            affine=revin_affine,
            subtract_last=revin_subtract_last,
        )

    def forward_setp1(self, z: torch.Tensor):
        # norm
        if self.model.revin:
            z = z.permute(0, 2, 1)
            z: torch.Tensor = self.model.revin_layer(z, "norm")
            z = z.permute(0, 2, 1)

        # do patching
        if self.model.padding_patch == "end":
            z = self.model.padding_patch_layer(z)
        z = z.unfold(
            dimension=-1, size=self.model.patch_len, step=self.model.stride
        )  # z: [bs x nvars x patch_num x patch_len]
        z = z.permute(0, 1, 3, 2)  # z: [bs x nvars x patch_len x patch_num]

        # model
        z_hidden: torch.Tensor = self.model.backbone(
            z
        )  # z_hidden: [bs x nvars x hidden_size x patch_num]

        return z_hidden

    def forward_setp2(self, z_hidden: torch.Tensor):
        z: torch.Tensor = self.model.head(z_hidden)  # z: [bs x nvars x h]

        # denorm
        if self.model.revin:
            z = z.permute(0, 2, 1)
            z = self.model.revin_layer(z, "denorm")
            z = z.permute(0, 2, 1)

        return z

    def forward(self, windows_batch):
        x: torch.Tensor = windows_batch["insample_y"]

        x = x.permute(0, 2, 1)

        # x = self.model(x)
        ########################
        z_hidden = self.forward_setp1(x)
        x = self.forward_setp2(z_hidden)
        ########################

        forecast = x.permute(0, 2, 1)

        return forecast
