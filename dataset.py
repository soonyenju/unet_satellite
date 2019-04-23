from pathlib import Path
import rasterio as rio
import numpy as np
import cv2, pickle
from sklearn.preprocessing import MinMaxScaler
import torch.utils.data as data
from sklearn.model_selection import train_test_split
import PIL.Image as Image
import os
from utils import ImageFolderSplitter

class Preparation(object):
    def __init__(self, path, size = 512, feature_range = (0, 255), deploy = False):
        self.path = path
        self.size = size
        arr = []
        if deploy:
            path_ls = list(path.glob(r'*.tif'))
        else:
            Xs_path = Path(path).joinpath('tiff/Xs')
            ys_path = Path(path).joinpath('tiff/ys')
            path_ls = list(Xs_path.glob(r'*.tif')) + list(ys_path.glob(r'*.tif'))
        for p in path_ls:
            with rio.open(p) as src:
                if not "profile" in locals().keys(): self.profile = src.profile
                data = src.read() # 3 dimensions: (band, height, width)
                data = data.reshape([data.shape[1], data.shape[2]])
                min_max_scaler = MinMaxScaler(feature_range=feature_range)  # range (0, 255)

                col_mean = np.nanmean(data, axis=0) # mean value for each column, ignoring the nans
                inds = np.where(np.isnan(data))
                data[inds] = np.take(col_mean, inds[1]) # fill the nan with col_mean

                data = min_max_scaler.fit_transform(data) # apply the scaler
                if feature_range == (0, 255): data = np.around(data, 0) # to incorporate NDVI into RGB, the value should be integers.
                # data.dtype = 'uint8'
                height, width = data.shape
                # add new rows and columns to be exactly divided by the size
                self.height_add = int((self.size - (height%self.size))/2)
                self.width_add = int((self.size - (width%self.size))/2)
                data = np.pad(data, ((self.height_add, self.height_add), (self.width_add, self.width_add)), 'constant')
                arr.append(data)
                print(p.stem)
        
        self.arr = np.array(arr)

    def cut2pieces(self, if_pics, test_size = 0.1, deploy = False):
        height, width = self.arr.shape[1::] # arr is 3D array: band, height, width
        self.height, self.width = height, width

        if if_pics:
            # create the RGB pics and corresponding masks
            p_pics = Path(self.path).joinpath('pics')
            if not p_pics.exists(): p_pics.mkdir()
            p_msks = Path(self.path).joinpath('msks')
            if not p_msks.exists(): p_msks.mkdir()

            for h in range(height//self.size):
                for w in range(width//self.size):
                    R = self.arr[0, h*self.size: (h+1)*self.size, w*self.size: (w+1)*self.size]
                    G = self.arr[1, h*self.size: (h+1)*self.size, w*self.size: (w+1)*self.size]
                    B = self.arr[2, h*self.size: (h+1)*self.size, w*self.size: (w+1)*self.size]
                    M = self.arr[3, h*self.size: (h+1)*self.size, w*self.size: (w+1)*self.size]
                    merged = cv2.merge([B,G,R])#合并R、G、B分量
                    order = str(h*width//self.size + w).zfill(3) # fill numbers to 3 digits strings
                    cv2.imwrite(p_pics.joinpath(order + '.png').as_posix(), merged)
                    merged_mask = cv2.merge([M])#合并R、G、B分量
                    cv2.imwrite(p_msks.joinpath(order + '_mask.png').as_posix(), merged_mask)
            splitter = ImageFolderSplitter(self.path)
            splitter.split(test_size)
        else:
            if deploy:
                print("deploying...")
                # cut array into 512x512 pieces
                rgbs = []
                for h in range(height//self.size):
                    for w in range(width//self.size):
                        temp = self.arr[:, h*self.size: (h+1)*self.size, w*self.size: (w+1)*self.size]
                        rgbs.append(temp)
                self.rgbs = np.array(rgbs)

            else:
                # cut array into 512x512 pieces
                rgbs = []
                msks = []
                for h in range(height//self.size):
                    for w in range(width//self.size):
                        temp = self.arr[:, h*self.size: (h+1)*self.size, w*self.size: (w+1)*self.size]
                        rgbs.append(temp[0:-1, :, :])
                        msks.append(temp[-1, :, :][np.newaxis, :]) # add a new axis to match the dimension of rgbs.
                rgbs = np.array(rgbs)
                msks = np.array(msks)

                # split into train and test sets
                X_train, X_test, y_train, y_test = train_test_split(rgbs, msks, test_size=test_size, shuffle=True)
                self.train_set = [X_train, y_train]
                self.test_set = [X_test, y_test]
                with open(Path(self.path).joinpath('train_set.pkl'), 'wb+') as f:
                    pickle.dump(self.train_set, f)
                with open(Path(self.path).joinpath('test_set.pkl'), 'wb+') as f:
                    pickle.dump(self.test_set, f)

class Dataset(data.Dataset):
    def __init__(self, dataset, transform=None, target_transform=None):
        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        '''
        the torchvision.transform function will swap axes, like (3, 512, 512) -> (512, 3, 512)
        according to official guide: https://seba-1511.github.io/tutorials/beginner/data_loading_tutorial.html
        >> swap color axis because
        >> numpy image: H x W x C
        >> torch image: C X H X W
        Xs = self.dataset[0][index].swapaxes(0, 2) works as well
        '''
        Xs = self.dataset[0][index].transpose((2, 1, 0))
        ys = self.dataset[1][index].transpose((2, 1, 0))

        if self.transform is not None:
            Xs = self.transform(Xs)
        if self.target_transform is not None:
            ys = self.target_transform(ys)
        return Xs, ys

    def __len__(self):
        return len(self.dataset[0])

class ImageDataset(data.Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        imgs = self.__make__(root)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x_path, y_path = self.imgs[index]
        img_x = Image.open(x_path)
        img_y = Image.open(y_path)
        if self.transform is not None:
            img_x = self.transform(img_x)
        if self.target_transform is not None:
            img_y = self.target_transform(img_y)
        return img_x, img_y

    def __make__(self, root):
        imgs=[]
        n=len(os.listdir(root))//2
        for i in range(n):
            img=os.path.join(root,"%03d.png"%i)
            mask=os.path.join(root,"%03d_mask.png"%i)
            imgs.append((img,mask))
        return imgs

    def __len__(self):
        return len(self.imgs)