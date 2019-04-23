from pathlib import Path
import numpy as np
import shutil

class ImageFolderSplitter(object):
    # images should be placed in folders like:
    # --root
    # ----root\pics
    # --------pics\000.png
    # --------pics\001.png
    # ----root\msks
    # --------msks\000_mask.png
    # --------msks\001_mask.png
    # ----root\train
    # --------train\xxx.png
    # --------train\xxx_mask.png
    # ----root\test
    # --------test\xxx.png
    # --------test\xxx_mask.png

    def __init__(self, root):
        self.root = Path(root)
        self.train_path = self.root.joinpath("train")
        self.test_path = self.root.joinpath("test")
        if not self.train_path.exists(): self.train_path.mkdir()
        if not self.test_path.exists(): self.test_path.mkdir()

    def split(self, test_size, xpath = "pics", ypath = "msks"):
        xpaths = list(self.root.joinpath(xpath).glob(r"*"))
        ypaths = list(self.root.joinpath(ypath).glob(r"*"))
        if len(xpaths) != len(ypaths):
            raise(Exception("Unequal lengths!"))
        the_len = len(xpaths)
        '''
        numpy.random.rand(d0,d1,...)指定一个数组，并使用区间[0,1)区间随机数据填充
        numpy.random.randn(d0,d1,...)指定一个数组，并使用区间随机数据填充。与rand区别在于，返回数据符合标准正态分布
        numpy.random.randint(low, high, size, dtype)生成[low到heigh)随机整数，是半开半闭区间
        numpy.random.random_integers(low, high, size)生成[low, high]的np.int类型随机整数，两边都闭区间
        numpy.random.random_sample(size)在区间[0,1)内生成指定size随机浮点数
        numpy.random.choice(a, size, replace, p)将会给定1维数组生成随机数
        '''
        train_idxs = np.random.randint(0, np.int(the_len), np.int(np.round(the_len*(1 - test_size), 0)))
        test_idxs = np.random.randint(0, np.int(the_len), np.int(np.round(the_len*(test_size), 0)))
        self.__execute__(train_idxs, xpaths, self.train_path) # train pics
        self.__execute__(train_idxs, ypaths, self.train_path) # train masks
        self.__execute__(test_idxs, xpaths, self.test_path) # test pics
        self.__execute__(test_idxs, ypaths, self.test_path) # test masks
    
    def __execute__(self, idxs, paths, dest_folder):
        for idx, _ in enumerate(idxs):
            src = paths[idx].as_posix()
            name = paths[idx].name
            dest = dest_folder.joinpath(name).as_posix()
            shutil.copy(src, dest)

# splitter = ImageFolderSplitter(r"D:\workspace\unet_classification\data")
# splitter.split(0.1)