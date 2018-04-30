import random, os, scipy.misc
import numpy as np
from utils import mod_crop

class DataGenerator():
    def __init__(self, args):
        self.args = args
        self.load_data()
        print('Data is loaded!')
    
    # load all dataset (HR and LR used in an experiment)
    def load_data(self):
        # initialze dataset
        self.dataset = []

        # check scales or degradation level (Track 2, 3, and 4) used in an experiment 
        self.scale_list = list(map(lambda x: int(x), self.args.scale.split('+')))

        if self.args.is_test:
            # directories for LR (for scale used)
            if not self.args.degrade:
                dirs = [os.path.join(self.args.data_dir, "DIV2K_test_LR_bicubic/X%d" % scale) for scale in self.scale_list]
            elif int(self.args.degrade) == 1:
                dirs = [os.path.join(self.args.data_dir, "DIV2K_test_LR_mild")]
            elif int(self.args.degrade) == 2:
                dirs = [os.path.join(self.args.data_dir, "DIV2K_test_LR_difficult")]
            elif int(self.args.degrade) == 3:
                dirs = [os.path.join(self.args.data_dir, "DIV2K_test_LR_wild")]
            else:
                raise NotImplementedError

            # The list of file_names for each directory 
            self.file_names_for_dirs = [sorted([f for f in os.listdir(dir) if not f=='Thumbs.db']) for dir in dirs]

            # Load data for LR 
            for dir, file_names in zip(dirs, self.file_names_for_dirs):
                tmp = []
                for file_name in file_names:
                    tmp.append(scipy.misc.imread(dir + "/" + file_name, mode='RGB'))
                self.dataset.append(tmp)

        elif self.args.degrade:
            self.degra_list = list(map(lambda x: int(x), self.args.degrade.split('+')))
            # directories for LR (for scale used) and HR (ex. [x2, x4, x8, HR])
            dirs = [os.path.join(self.args.data_dir, "DIV2K_train_LR_degrade/%d" % degra) for degra in self.degra_list]
            hr_dirs = [os.path.join(self.args.data_dir, "DIV2K_train_HR_degrade/%d" % degra) for degra in self.degra_list]
            for dir in hr_dirs:
                dirs.append(dir)

            # The list of file_names for each directory 
            self.file_names_for_dirs = [sorted([f for f in os.listdir(dir) if not f=='Thumbs.db']) for dir in dirs]

            # Load data for LR and HR (ex. self.dataset = [[mild_LR], [difficult_LR], [wild_LR], [mild_HR], [difficult_HR], [wild_HR]])
            for dir, file_names in zip(dirs, self.file_names_for_dirs):
                tmp = []
                for file_name in file_names:
                    tmp.append(scipy.misc.imread(dir + "/" + file_name, mode='RGB'))
                self.dataset.append(tmp)    
        else:
            # directories for LR (for scale used) and HR (ex. [x2, x4, x8, HR])
            dirs = [os.path.join(self.args.data_dir, "DIV2K_train_LR_bicubic/X%d" % scale) for scale in self.scale_list]
            dirs.append(os.path.join(self.args.data_dir, "DIV2K_train_HR"))

            # The list of file_names for each directory 
            self.file_names_for_dirs = [sorted([f for f in os.listdir(dir) if not f=='Thumbs.db']) for dir in dirs]

            # Load data for LR and HR (ex. self.dataset = [[x2_data], [x4_data], [x8_data], [HR_data]])
            for dir, file_names in zip(dirs, self.file_names_for_dirs):
                tmp = []
                for file_name in file_names:
                    tmp.append(scipy.misc.imread(dir + "/" + file_name, mode='RGB'))
                self.dataset.append(tmp)

    # construct batch data for randomly selected scale or degradation level
    # only use this function during traing, not validation, not testing 
    def get_batch(self, batch_size, idx_scale, in_patch_size=32):
        # randomly selet scale
        scale = self.scale_list[idx_scale]

        # select dataset
        if self.args.degrade:          
            inputs = self.dataset[idx_scale]
            targets = self.dataset[idx_scale+len(self.scale_list)]
            scale = 4
        else:
            inputs = self.dataset[idx_scale]
            targets = self.dataset[-1]

        # adjust the size of target dataset corresponding to current scale using mod_crop
        tmp = []
        for img in targets:
            tmp.append(mod_crop(img, scale)) 
        targets = tmp

        # divide train/validation 
        inputs = inputs[:-self.args.num_valid]
        targets = targets[:-self.args.num_valid]

        # assgin target patch size
        tar_patch_size = in_patch_size * scale
        in_batch = []
        tar_batch = []
        for i in range(batch_size):
            # select image
            idx_img = random.randrange(len(inputs)) 
            in_img = inputs[idx_img]
            tar_img = targets[idx_img]

            # random crop
            y, x, _ = in_img.shape
            in_x = random.randint(0, x - in_patch_size)
            in_y = random.randint(0, y - in_patch_size) 
            tar_x = in_x * scale
            tar_y = in_y * scale 
            in_patch = in_img[in_y:in_y + in_patch_size, in_x:in_x + in_patch_size]  
            tar_patch = tar_img[tar_y:tar_y + tar_patch_size, tar_x:tar_x + tar_patch_size]  

            if not self.args.degrade:
                # random rotate
                rot_num = random.randint(1, 4)
                in_patch = np.rot90(in_patch, rot_num)
                tar_patch = np.rot90(tar_patch, rot_num)
                
                # random flip left-to-right
                flipflag = random.random() > 0.5
                if flipflag:
                    in_patch = np.fliplr(in_patch) 
                    tar_patch = np.fliplr(tar_patch)

            # construct mini-batch
            in_batch.append(in_patch)
            tar_batch.append(tar_patch)                              
        return in_batch, tar_batch
