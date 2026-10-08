import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth=1.0, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1)
        den = torch.sum(predict, dim=1) + torch.sum(target, dim=1) + self.smooth

        dice_score = 2 * num / den
        loss_avg = 1 - dice_score.mean()
        return loss_avg


class DiceLoss4BraTS(nn.Module):
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss4BraTS, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index
        self.dice = BinaryDiceLoss(**self.kwargs)

    def forward(self, predict, target):
        assert predict.shape == target.shape, f'predict {predict.shape} & target {target.shape} shape do not match'
        total_loss = 0
        predict = torch.sigmoid(predict)

        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = self.dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], f'Expect weight shape [{target.shape[1]}], get [{self.weight.shape[0]}]'
                    dice_loss *= self.weight[i]
                total_loss += dice_loss

        divisor = target.shape[1] - 1 if self.ignore_index is not None else target.shape[1]
        return total_loss / divisor


class BCELoss4BraTS(nn.Module):
    def __init__(self, ignore_index=None, **kwargs):
        super(BCELoss4BraTS, self).__init__()
        self.kwargs = kwargs
        self.ignore_index = ignore_index
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        total_loss = 0
        for i in range(target.shape[1]):
            if i != self.ignore_index:
                bce_loss = self.criterion(predict[:, i], target[:, i])
                total_loss += bce_loss
        return total_loss.mean()


class DiceLoss4MOTS(nn.Module):
    def __init__(self, weight=None, ignore_index=None, num_classes=3, **kwargs):
        super(DiceLoss4MOTS, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.dice = BinaryDiceLoss(**self.kwargs)

    def forward(self, predict, target):
        total_loss = []
        predict = torch.sigmoid(predict)

        for i in range(self.num_classes):
            if i != self.ignore_index:
                dice_loss = self.dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == self.num_classes, f'Expect weight shape [{self.num_classes}], get [{self.weight.shape[0]}]'
                    dice_loss *= self.weight[i]
                total_loss.append(dice_loss)

        total_loss = torch.stack(total_loss)
        total_loss = total_loss[total_loss == total_loss]
        return total_loss.sum() / total_loss.shape[0]


class CELoss4MOTS(nn.Module):
    def __init__(self, ignore_index=None, num_classes=3, **kwargs):
        super(CELoss4MOTS, self).__init__()
        self.kwargs = kwargs
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        total_loss = []
        for i in range(self.num_classes):
            if i != self.ignore_index:
                ce_loss = self.criterion(predict[:, i], target[:, i])
                ce_loss = torch.mean(ce_loss, dim=[1, 2, 3])
                ce_loss_avg = ce_loss[target[:, i, 0, 0, 0] != -1].sum() / ce_loss[target[:, i, 0, 0, 0] != -1].shape[0]
                total_loss.append(ce_loss_avg)

        total_loss = torch.stack(total_loss)
        total_loss = total_loss[total_loss == total_loss]
        return total_loss.sum() / total_loss.shape[0]


class DomainBCELoss(nn.Module):
    def __init__(self):
        super(DomainBCELoss, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        total_loss = self.criterion(predict, target)
        return total_loss.sum() / total_loss.shape[0]


class KDLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=None, num_classes=3, **kwargs):
        super(KDLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index
        self.num_classes = num_classes

    def forward(self, source_logits, source_gt, target_logits, target_gt):
        temperature = 2.0
        loss = 0.0

        for i in range(self.num_classes):
            eps = 1e-6
            s_mask = source_gt[:, i, :, :, :].unsqueeze(1).repeat_interleave(repeats=self.num_classes, dim=1)
            s_logits_mask_out = source_logits * s_mask
            s_logits_avg = torch.sum(s_logits_mask_out, dim=(0, 2, 3, 4)) / (torch.sum(source_gt[:, i, :, :, :]) + eps)
            s_soft_prob = F.softmax(s_logits_avg / temperature, dim=0)

            t_mask = target_gt[:, i, :, :, :].unsqueeze(1).repeat_interleave(repeats=self.num_classes, dim=1)
            t_logits_mask_out = target_logits * t_mask
            t_logits_avg = torch.sum(t_logits_mask_out, dim=(0, 2, 3, 4)) / (torch.sum(target_gt[:, i, :, :, :]) + eps)
            t_soft_prob = F.softmax(t_logits_avg / temperature, dim=0)

            loss = (torch.sum(s_soft_prob * torch.log(s_soft_prob / (t_soft_prob + eps))) +
                    torch.sum(t_soft_prob * torch.log(t_soft_prob / (s_soft_prob + eps)))) / 2.0
        return loss


class DomainClsLoss(nn.Module):
    def __init__(self):
        super(DomainClsLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], 'predict & target shape do not match'
        return self.criterion(predict, target)
