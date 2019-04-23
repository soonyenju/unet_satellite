"""
to-do-list:
1. how to clear cache, remove redundant variables which have gradients
2. log train and test in txt files
"""

from dataset import Preparation, Dataset, ImageDataset
import pickle, argparse
from unet import Unet
import torch
from torch import autograd, optim
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import numpy as np

class Dispatcher(object):
    def __init__(self, device, data_path = 'data'):
        self.device = device
        self.data_path = data_path
        self.x_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        # mask只需要转换为tensor
        self.y_transforms = transforms.ToTensor()
        self.model = Unet(3, 1).to(device)
        self.batch_size = args.batch_size
        self.crop_size = args.crop_size
        self.num_epoch = args.num_epoch
        self.usr_info = args.usr_info
        self.src = args.src
        self.criterion = torch.nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters())

    def train(self, if_pics):
        if not if_pics:
            try:
                with open(Path.cwd().joinpath('data/train_set.pkl'), 'rb') as f:
                    self.train_set = pickle.load(f)
                # with open(Path.cwd().joinpath('data/test_set.pkl'), 'rb') as f:
                #     self.test_set = pickle.load(f)
            except Exception as e:
                print(e)
                prep = Preparation(self.data_path, size = self.crop_size, feature_range = (0, 1))
                prep.cut2pieces(0)
                self.train_set = prep.train_set
                # self.test_set = prep.test_set
            dataset = Dataset(self.train_set,transform=self.x_transforms,target_transform=self.y_transforms)
        else:
            prep = Preparation(self.data_path, size = self.crop_size)
            prep.cut2pieces(1)
            dataset = ImageDataset("data/train",transform=self.x_transforms,target_transform=self.y_transforms)

        dataloaders = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        self.train_model(self.model, self.criterion, self.optimizer, dataloaders, self.num_epoch)

    def train_model(self, model, criterion, optimizer, dataload, num_epochs):
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)
            dt_size = len(dataload.dataset)
            epoch_loss = 0
            step = 0
            for x, y in dataload:
                step += 1
                inputs = x.to(device)
                labels = y.to(device)
                # the dtype of weights is float, so the types of input data must match the weights.
                if inputs.dtype != torch.float32: inputs = inputs.float()
                if labels.dtype != torch.float32: labels = labels.float()
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                '''
                # a way to fix runtime warning, CUDA ran out of memory. However, it did not work.
                try:
                    outputs = model(inputs)
                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        print('| WARNING: ran out of memory')
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()
                    else:
                        raise e
                '''
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                print(f"{step}/{(dt_size - 1) // dataload.batch_size + 1: d}, train_loss:{loss.item(): .3f}")
            print(f"epoch {epoch} loss: {epoch_loss}")
        save_path = Path.cwd().joinpath("models")
        if not save_path.exists(): save_path.mkdir()
        torch.save(model.state_dict(), save_path.joinpath(f'weights_{str(self.batch_size).zfill(3)}batch{str(self.crop_size).zfill(3)}crop{str(self.num_epoch).zfill(3)}epoch{self.usr_info}.pth').as_posix())
        return model

    #显示模型的输出结果
    def test(self, if_pics, confidence = 0.6):
        model = Unet(3, 1)
        model.load_state_dict(torch.load(Path.cwd().joinpath("models").joinpath(args.ckp),map_location='cpu'))
        # evaluate the using recall and precision:
        # TP -> wheat, FN -> missing, FP -> error, TN -> not wheat
        TP, FN, FP, TN = 0, 0, 0, 0
        if not if_pics:
            try:
                with open(Path.cwd().joinpath('data/test_set.pkl'), 'rb') as f:
                    self.test_set = pickle.load(f)
            except Exception as e:
                print(e)
                prep = Preparation(self.data_path, size = self.crop_size, feature_range = (0, 1))
                prep.cut2pieces(0)
                self.test_set = prep.test_set
            dataset = Dataset(self.test_set,transform=self.x_transforms,target_transform=self.y_transforms)
        else:        
            dataset = ImageDataset("data/test", transform=self.x_transforms,target_transform=self.y_transforms)
        dataloaders = DataLoader(dataset, batch_size=1)
        test = model.eval()
        with torch.no_grad():
            print('-' * 10)
            for x, y_true in dataloaders:
                y_pred = model(x) 
                y_true = torch.squeeze(y_true).numpy() # the size of y_true is 4D: batch, band, height, width
                y_pred = torch.squeeze(y_pred).numpy() # its size similar to that of y_true

                y_pred = np.select([y_pred >= confidence, y_pred < confidence], [1, 0])
                diff = (y_true * 2 + 2 - y_pred).ravel().astype(np.int)
                counts = np.bincount(diff)
                FP = FP + counts[1]
                TN = TN + counts[2]
                TP = TP + counts[3]
                FN = FN + counts[4]
                # sum_val = FP + TN + TP + FN
                # print(f'The error is: {100*FP/(sum_val): .2f}%;') 
                # print(f'the non-wheat is: {100*TN/(sum_val): .2f}%;') 
                # print(f'the wheat is: {100*TP/(sum_val): .2f}%;')
                # print(f'the missing is: {100*FN/(sum_val): .2f}%') 
            print(f"Accuracy: {100*(TP)/(TP+FP): .2f}%")
            print(f"Recall: {100*(TP)/(TP+FN): .2f}%")
    
    def deploy(self):
        model = Unet(3, 1)
        model.load_state_dict(torch.load(Path.cwd().joinpath("models").joinpath(args.ckp),map_location='cpu'))
        prep = Preparation(self.data_path, size = self.crop_size)
        x = torch.from_numpy(prep.arr)
        y=model(x)
        print(y)

        


if __name__ == "__main__":
    #参数解析
    parse = argparse.ArgumentParser()
    parse.add_argument("action", type=str, help="train, test, apply")
    parse.add_argument("--batch_size", type=int, default=8)
    parse.add_argument("--crop_size", type=int, default=256)
    parse.add_argument("--num_epoch", type=int, default=20)
    parse.add_argument("--confi", type=float, default=0.8)
    parse.add_argument("--usr_info", type=str, default="")
    parse.add_argument("--device", type=str, default="not defined")
    parse.add_argument("--ckp", type=str, help="the path of model weight file")
    parse.add_argument("--src", type=str, help="the path of data to be applied")
    args = parse.parse_args()

    # 是否使用cuda
    if args.device == "cuda":
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dist = Dispatcher(device)
    if args.action == "train":
        print("training the arrays...")
        dist.train(False)
    elif args.action == "train_pics":
        print("training the pics in './data/train' folder...")
        dist.train(True)
    elif args.action=="test":
        print("testing the model using prepared testset...")
        dist.test(False)
    elif args.action=="test_pics":
        print("testing the pics in './data/test' folder....")
        dist.test(True, confidence=args.confi)
    elif args.action == "deploying":
        print("deploying the model to data...")
        dist.deploy()

    # python start.py train --crop_size=256 --num_epoch=1 --usr_info=zhusy
    # python start.py test_pics --ckp=weights_19.pth