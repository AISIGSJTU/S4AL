import argparse
import os
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as TF

from predictor.model import Predictor


class FidDataset(Dataset):
    def __init__(self, file_root, test=False, transforms=None, resize=True):
        if test:
            with open("{}/fid_test.txt".format(file_root), 'r+') as fr:
                dic = eval(fr.read())
                self.files = list(dic.items())
        else:
            with open("{}/fid_train.txt".format(file_root), 'r+') as fr:
                dic = eval(fr.read())
                self.files = list(dic.items())

        if transforms is None:
            if resize:
                self.transforms = TF.Compose([
                    TF.Resize(224),
                    TF.ToTensor(),
                ])
            else:
                self.transforms = TF.ToTensor()
        else:
            self.transforms = transforms

    def __getitem__(self, i):
        path, score = self.files[i]
        img = Image.open(path).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        return img, score

    def __len__(self):
        return len(self.files)


def evaluate_model(model, test_dataloader, device):
    model.eval()

    losses = []
    loss_function = nn.MSELoss()

    for i_batch, batch in enumerate(test_dataloader, 0):

        inputs, targets = batch
        inputs, targets = inputs.float().to(device), targets.float().to(device)
        targets = targets.reshape((targets.shape[0], 1))

        with torch.no_grad():
            outputs = model(inputs)

        loss = loss_function(outputs, targets)
        losses.append(loss.item())

    model.train()
    return sum(losses) / len(losses)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='The learning rate')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size to use')
    parser.add_argument('--num-workers', type=int, default=None,
                        help=('Number of processes to use for data loading. '
                              'Defaults to `min(8, num_cpus)`'))
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use. Like cuda, cuda:0 or cpu')
    args = parser.parse_args()

    if args.device is None:
        device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    else:
        device = torch.device(args.device)

    if args.num_workers is None:
        num_avail_cpus = len(os.sched_getaffinity(0))
        num_workers = min(num_avail_cpus, 8)
    else:
        num_workers = args.num_workers

    # hyper-parameters
    batch_size = args.batch_size
    i_epoch_start = 0
    num_epoch = 1000
    ckp_save_path = "logs/predictor/ckps"
    os.makedirs(ckp_save_path, exist_ok=True)

    # build the model
    model = Predictor(num_classes=1,
                      ckp_path="msm/backbone/mobilenet_v2-b0353104.pth").to(device)

    # loss function, optimizer, ect.
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # load the dataset
    train_fid_dataset = FidDataset("datasets/predictor/train", test=False)
    train_fid_dataloader = DataLoader(train_fid_dataset,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=num_workers)
    test_fid_dataset = FidDataset("datasets/predictor/test", test=True)
    test_fid_dataloader = DataLoader(test_fid_dataset,
                                     batch_size=batch_size,
                                     shuffle=False,
                                     num_workers=num_workers)

    for i_epoch in range(i_epoch_start, num_epoch):
        for i_batch, batch in enumerate(train_fid_dataloader, 0):

            inputs, targets = batch
            inputs, targets = inputs.float().to(device), targets.float().to(device)
            targets = targets.reshape((targets.shape[0], 1))

            outputs = model(inputs)

            optimizer.zero_grad()
            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()

            print("[{:4d}]-[{:4d}] loss {}".format(i_epoch, i_batch, loss.item()))

        print("\nTesting...")
        test_mse = evaluate_model(model, test_fid_dataloader, device)
        print("[{:4d}] MSE (test) {}\n".format(i_epoch, test_mse))

        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'i_epoch': i_epoch,
        }, "{}/ckp_{}.tar".format(ckp_save_path, i_epoch))


if __name__ == '__main__':
    main()
