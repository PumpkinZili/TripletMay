import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLossV2(nn.Module):
    """TripletLoss and Inner class Loss together
    """

    def __init__(self, margin):
        super(TripletLossV2, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=None):
        """Average
        Args:
            size_average: None, average on semi and hard,

        """
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)

        triplet_loss = F.relu(distance_positive - distance_negative + self.margin)

        if size_average == None:
            return triplet_loss.mean() if size_average else triplet_loss.sum()


# class InnerClassLoss(nn.Module):
#    """Inner class loss
#    """
#    def __init__(self):
#        super(InnerClassLoss, self).__nit__()
#
#    def forward(self, anchor, positive, size_average=None)
#        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
#        inner_class_loss = torch.log1p(distance_positive)
#
#        if size_average == None:
#            return inner_class_loss
#        elif size_average == 'mean':
#            return inner_class_loss.mean()
#        elif size_average == 'sum':
#            return inner_class_loss.sum()

class CenterLoss(nn.Module):

    def __init__(self, n_classes, mean_center=True):
        """
        if mean_center: return <inner class embeddeds - Mean(inner class embeddeds)>
        else: return <inner class embeddeds - the neareast embedded of Mean(inner class embeddeds)>
        """
        super(CenterLoss, self).__init__()
        self.n_classes = n_classes
        self.mean_center = mean_center

    def forward(self, embeddeds, target):
        losses = []
        for i in range(self.n_classes):
            clsi_id = torch.nonzero(target == i)
            if clsi_id.size(0) == 0:
                losses.append(0)
                continue
            embeddeds_i = embeddeds[clsi_id].squeeze(dim=1)
            mean_embeddeds = embeddeds_i.mean(dim=0)
            if self.mean_center:
                losses.append((embeddeds_i - mean_embeddeds).norm(p=2, dim=1).mean())
            else:
                distances = (embeddeds_i - mean_embeddeds).norm(p=2, dim=1)
                nearest = embeddeds_i[torch.argsort(distances)[0]]
                losses.append((embeddeds_i - nearest).norm(p=2, dim=1).mean())
        return sum(losses)








class AlteredTripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin, c=1, reduction='mean'):
        super(AlteredTripletLoss, self).__init__()
        self.margin = margin
        self.reduction = reduction
        self.c = c

    def forward(self, anchor, positive, negative):
        """Average
        Args:
            reduce: none, mean or sum,

        """
        distance_positive = (anchor - positive).norm(p=2, dim=1)  # .pow(.5)
        distance_negative = (anchor - negative).norm(p=2, dim=1)  # .pow(.5)
        losses = F.relu(self.c * distance_positive - distance_negative + self.margin)

        if self.reduction == 'none':
            return losses
        elif self.reduction == 'mean':
            return losses.mean()
        elif self.reduction == 'sum':
            return losses.sum()
        else:
            raise RuntimeError("reduce should be one of none, mean or sum")
