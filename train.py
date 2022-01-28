import torch
import torch.optim as optim
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import YOLOv1
from dataset import VOCDataset
from utils.utils import (
    non_max_suppression,
    mean_average_precision,
    get_bboxes,
    cellboxes_to_boxes,
    plot_image,
    save_checkpoint,
    load_checkpoint
)
from loss import YOLOLoss

# seed = 123
# torch.manual_seed(seed)

# Hyper Parameters etc.
learning_rate = 2e-5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 4
weight_decay = 0
epochs = 180
num_workers = 0
pin_memory = True
load_model = False
load_model_file = 'overfit.pth.tar'
img_dir = "G:\PascalVOC_YOLO\images"
label_dir = "G:\PascalVOC_YOLO\labels"
train_csv_path = "G:\PascalVOC_YOLO\\train.csv"

# transforms
transform = transforms.Compose([
    transforms.Resize([448, 448]),
    transforms.ToTensor(),
    # transforms.Normalize(),
])


def train_fn():
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update the progress bar
        loop.set_postfix(loss=loss.item())
    print(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")


# model
model = YOLOv1(3, split_size=7, num_boxes=2, num_class=20).to(device)

# optim
optimizer = optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=weight_decay,
)

# loss
loss_fn = YOLOLoss()

# load weight
if load_model:
    # model.load_state_dict(torch.load(load_model_file))
    load_checkpoint(torch.load(load_model_file), model, optimizer)

# dataset & dataloader
train_dataset = VOCDataset(
    "G:\PascalVOC_YOLO\\100examples.csv",
    img_dir,
    label_dir,
    transform=transform
)

test_dataset = VOCDataset(
    "G:\PascalVOC_YOLO\\100examples.csv",
    img_dir,
    label_dir,
    transform=transform
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=pin_memory,
    shuffle=True,
    drop_last=False,
)
test_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=pin_memory,
    shuffle=False,
    drop_last=False,
)


def train():
    for epoch in range(epochs):
        train_fn()
        pred_boxes, target_boxes = get_bboxes(
            train_loader, model, iou_threshold=0.5, threshold=0.4
        )
        mean_avg_prec = mean_average_precision(
            pred_boxes, target_boxes, iou_threshold=0.5, box_format='midpoint'
        )
        print(f"Train mAP: {mean_avg_prec}")

        # 保存参数
        # if mean_avg_prec > 0.9:
    checkpoint = {
       "state_dict": model.state_dict(),
       "optimizer": optimizer.state_dict(),
    }
    save_checkpoint(checkpoint, filename=load_model_file)



def eval():
    load_checkpoint(torch.load(load_model_file), model, optimizer)
    for x, y in train_loader:
        x = x.to(device)
        for idx in range(batch_size):
            bboxes = cellboxes_to_boxes(model(x))
            bboxes = non_max_suppression(bboxes[idx], iou_threshold=0.5, threshold=0.4, box_format="midpoint")
            plot_image(x[idx].permute(1, 2, 0).to("cpu"), bboxes)
        break

if __name__ == "__main__":
    # train()
    eval()


