import os
import torch


class Config():
    def __init__(self):
        # data path
        self.dataset_dir = './Dataset'
        
        ''' Train Dataset: CAMO-Train + COD10K-Train '''
        self.train_imgs = os.path.join(self.dataset_dir, 'TrainDataset', 'Imgs')
        self.train_masks = os.path.join(self.dataset_dir, 'TrainDataset', 'GT')
        self.train_sample_txt = os.path.join(self.dataset_dir, 'TrainDataset', 'sampled_images.txt')
        self.sam_labels = os.path.join(self.dataset_dir, 'SAMLabel', 'sampled_masks')
        
        ''' Test Dataset '''
        # CHAMELEON
        self.test_CHAMELEON_imgs = os.path.join(self.dataset_dir, 'TestDataset', 'CHAMELEON', 'Imgs')
        self.test_CHAMELEON_masks = os.path.join(self.dataset_dir, 'TestDataset', 'CHAMELEON', 'GT')
        # CAMO-Test
        self.test_CAMO_imgs = os.path.join(self.dataset_dir, 'TestDataset', 'CAMO', 'Imgs')
        self.test_CAMO_masks = os.path.join(self.dataset_dir, 'TestDataset', 'CAMO', 'GT')
        # COD10K-Test
        self.test_COD10K_imgs = os.path.join(self.dataset_dir, 'TestDataset', 'COD10K', 'Imgs')
        self.test_COD10K_masks = os.path.join(self.dataset_dir, 'TestDataset', 'COD10K', 'GT')
        # NC4K
        self.test_NC4K_imgs = os.path.join(self.dataset_dir, 'TestDataset', 'NC4K', 'Imgs')
        self.test_NC4K_masks = os.path.join(self.dataset_dir, 'TestDataset', 'NC4K', 'GT')


        self.num_workers = 8

        self.u_train_size = 392
        self.l_train_size = 392
        self.test_size = 392

        ''' Save Paths '''
        self.result_path = './results'
        self.dir_name = 'ts_model'
        self.save_dir = os.path.join(self.result_path, self.dir_name)

        self.CUDA = True if torch.cuda.is_available() else False
        self.device = torch.device('cuda' if self.CUDA else 'cpu')

        self.epochs = 15
        self.u_batch_size = 32
        self.l_batch_size = 32

        ''' Data Augmentation '''
        self.rVFlip = True
        self.rCrop = True
        self.rRotate = False
        self.colorEnhance = True
        self.rPeper = False

        ''' Optimizer '''
        self.weight_decay = 0

        ''' LR Scheduler '''
        self.learning_rate = 1e-4
        self.min_lr = 1e-7

        ''' logging '''
        self.log_interval = 50


if __name__ == '__main__':
    config = Config()
    print(config)
