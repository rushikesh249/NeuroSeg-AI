import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneralizedDiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        # logits: (B, C, D, H, W), target: (B, D, H, W) with values 0..C-1
        probs = F.softmax(logits, dim=1)
        C = logits.shape[1]
        target_onehot = F.one_hot(target.long(), num_classes=C).permute(0, 4, 1, 2, 3).float()

        # compute volumes per class
        w = 1.0 / (torch.sum(target_onehot, dim=(2, 3, 4)) ** 2 + self.eps)
        intersect = (probs * target_onehot).sum(dim=(2, 3, 4))
        denom = (probs + target_onehot).sum(dim=(2, 3, 4))

        # weighted dice per class per batch
        num = 2.0 * (w * intersect).sum(dim=1)
        den = (w * denom).sum(dim=1) + self.eps
        loss = 1.0 - (num / den).mean()
        return loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, ignore_index=None):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        # logits: (B, C, ...), target: (B, ...)
        if self.ignore_index is None:
            logpt = -F.cross_entropy(logits, target.long(), reduction='none')
        else:
            logpt = -F.cross_entropy(logits, target.long(), reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(logpt)
        loss = ((1 - pt) ** self.gamma) * (-logpt)
        # mask ignore_index
        if self.ignore_index is not None:
            mask = (target != self.ignore_index).float()
            loss = loss * mask
            return loss.sum() / (mask.sum() + 1e-6)
        else:
            return loss.mean()


class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0, ignore_index=None):
        super().__init__()
        self.alpha = alpha
        self.gdl = GeneralizedDiceLoss()
        self.focal = FocalLoss(gamma=gamma, ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        loss_gdl = self.gdl(logits, target)
        loss_focal = self.focal(logits, target)
        return self.alpha * loss_gdl + (1.0 - self.alpha) * loss_focal


if __name__ == '__main__':
    # smoke test
    l = CombinedLoss()
    logits = torch.randn(2, 4, 32, 64, 64)
    target = torch.randint(0, 4, (2, 32, 64, 64))
    print(l(logits, target).item())
