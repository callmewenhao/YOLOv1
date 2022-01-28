import torch
import torch.nn as nn
from utils.IOU import intersection_over_union


class YOLOLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super(YOLOLoss, self).__init__()
        self.mse = nn.MSELoss(reduction='sum')  # 论文中是对误差平方求和
        self.S = S
        self.B = B
        self.C = C
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        # reshape
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)
        iou_b1 = intersection_over_union(
            predictions[..., 21:25], target[..., 21:25]
        )
        iou_b2 = intersection_over_union(
            predictions[..., 26:30], target[..., 21:25]
        )
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
        iou_maxes, best_box = torch.max(ious, dim=0)  # 返回最大值和索引
        exist_box = target[..., 20].unsqueeze(3)  # Iobj_i
        # for box coordinates
        box_predictions = exist_box * (
            best_box * predictions[..., 26:30]
            + (1 - best_box) * predictions[..., 21:25]
        )
        box_target = exist_box * target[..., 21:25]

        box_predictions[..., 2:4] = torch.sign(  # 对长和宽开根号，并且保证开根号前后符号一致
            box_predictions[..., 2:4]
        ) * torch.sqrt(
            torch.abs(
                box_predictions[..., 2:4] + 1e-6
            )
        )
        box_target[..., 2:4] = torch.sqrt(box_target[..., 2:4])

        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),  # (N, S, S, 4)->(N*S*S, 4)
            torch.flatten(box_target, end_dim=-2)
        )

        # for object loss  只算最好的bb
        pred_obj = (
            best_box * predictions[..., 25:26]
            + (1 - best_box) * predictions[..., 20:21]
        )
        # (N*S*S)
        object_loss = self.mse(
            torch.flatten(exist_box * pred_obj),
            torch.flatten(exist_box * target[..., 20:21])
        )

        # for no object loss  两个bb都算
        no_object_loss = self.mse(
            torch.flatten((1 - exist_box) * predictions[..., 20:21], start_dim=1),
            torch.flatten((1 - exist_box) * target[..., 20:21], start_dim=1)
        )
        no_object_loss += self.mse(
            torch.flatten((1 - exist_box) * predictions[..., 25:26], start_dim=1),
            torch.flatten((1 - exist_box) * target[..., 20:21], start_dim=1)
        )

        # for class loss
        # (N, S, S, 20)->(N*S*S, 20)
        class_loss = self.mse(
            torch.flatten(exist_box * predictions[..., :20], end_dim=-2),
            torch.flatten(exist_box * target[..., :20], end_dim=-2)
        )

        # 总loss
        loss = (
            self.lambda_coord * box_loss  # first part of the paper loss
            + object_loss  # second part
            + self.lambda_noobj * no_object_loss  # third part
            + class_loss  # last part
        )
        return loss








