import torch.nn as nn
import torch.nn.functional as F


class MaskedBCEWithLogitsLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, targets):

        label_mask = (targets != -1)

        filtered_logits = logits[label_mask]
        filtered_targets = targets[label_mask]

        return F.binary_cross_entropy_with_logits(
            filtered_logits, filtered_targets.float()
        )
