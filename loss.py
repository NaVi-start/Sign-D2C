from torch import nn

class RegLoss(nn.Module):
    def __init__(self, cfg, target_pad=0.0):
        super(RegLoss, self).__init__()

        self.loss = cfg["training"]["loss"].lower()

        if self.loss == "l1":
            self.criterion = nn.L1Loss()
        elif self.loss == "mse":
            self.criterion = nn.MSELoss()
        else:
            print("Loss not found - revert to default L1 loss")
            self.criterion = nn.L1Loss()

        model_cfg = cfg["model"]
        self.target_pad = target_pad
        self.loss_scale = model_cfg.get("loss_scale", 1.0)

    def forward(self, preds, targets, loss_mask=None):

        if loss_mask==None:
            loss_mask = (targets != self.target_pad)

        preds_masked = preds * loss_mask
        targets_masked = targets * loss_mask

        loss = self.criterion(preds_masked, targets_masked) 

        if self.loss_scale != 1.0:
            loss = loss * self.loss_scale

        return loss
