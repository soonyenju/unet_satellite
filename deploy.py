# encoding: utf-8
"""
@author: syzhu 
@contact: soonyenju@foxmail.com
"""
from unet import Unet
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import rasterio as rio
import argparse, torch, pickle
from dataset import Preparation, Dataset
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

class Server(object):
    def __init__(self, device, path):
        path = Path(path)
        self.device = device
        self.x_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        # mask只需要转换为tensor
        self.y_transforms = transforms.ToTensor()

        self.prep = Preparation(path, size = args.crop_size, feature_range = (0, 1), deploy = True)
        self.prep.cut2pieces(0, deploy = True)
        Xs = self.prep.rgbs # ndarray
        ysdummy = np.empty((Xs.shape[0], 1, Xs.shape[2], Xs.shape[3]))
        ds = [Xs, ysdummy]
        self.ds = Dataset(ds,transform=self.x_transforms,target_transform=self.y_transforms)



    def run(self, confidence = 0.6):
        temp = []
        array_list = []
        model = Unet(3, 1).to(self.device)
        model.load_state_dict(torch.load(Path.cwd().joinpath("models").joinpath(args.ckp),map_location='cpu'))
        dataloaders = DataLoader(self.ds, batch_size=1)
        with torch.no_grad():
            print('-' * 10)
            for x, _ in dataloaders:
                y = model(x.to(self.device))
                y = torch.squeeze(y).cpu().numpy()
                y = np.select([y >= confidence, y < confidence], [1, 0])
                array_list.append(y)

        '''
        merge all ys into one larger matrix, stack along width-wise, then stack these matrices along height-wise:
        for example: 
            height == 5120
            width == 7424
            size == 256
        therefore the number of width-wise arrays is 7424/256->29 while number of height-wise array is  5120/256->20
        => after the first concatenate, we have 20:
        arr+arr+...+arr, 29 array conducted along axis 1;
        then concatenate these 29x25 arrays along axis 0
        '''
        for h in range(self.prep.height//self.prep.size):
            temp.append(np.concatenate(array_list[h*self.prep.width//self.prep.size: (h+1)*self.prep.width//self.prep.size], axis = 1))
        temp = np.concatenate(temp, axis = 0)
        # remove padding
        temp = temp[self.prep.height_add: self.prep.height - self.prep.height_add, 
                self.prep.width_add: self.prep.width - self.prep.width_add]

        self.prep.profile.update(
                dtype=rio.uint8
        )

        with rio.open(args.name + '.tif', 'w', **self.prep.profile) as dst:
            dst.write(temp.astype(rio.uint8), 1)

if __name__ == "__main__":
    # 是否使用cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #参数解析
    parse = argparse.ArgumentParser()
    parse.add_argument("--crop_size", type=int, default=256)
    parse.add_argument("--confi", type=float, default=0.8)
    parse.add_argument("--ckp", type=str, help="the path of model weight file")
    parse.add_argument("--src", type=str, help="the path of data to be applied")
    parse.add_argument("--name", type=str, default="prediction")
    args = parse.parse_args()

    server = Server(device, args.src)
    server.run(confidence = args.confi)

    # python deploy.py --src=./src --ckp=weights_19.pth