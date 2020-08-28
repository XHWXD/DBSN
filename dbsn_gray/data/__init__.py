import importlib
import torch.utils.data
from data.base_dataset import BaseDataset
from .image_dataset import ImageDataset 

dataset_map = {
    'image': ['bsd68','set12','kodak24','set14','mcmaster','cbsd68','bsd500','wed4744','div2k','imagenet_val'],
}
dataset_map = {i:j for j in dataset_map.keys() for i in dataset_map[j]}

# def find_dataset_using_name(dataset_name, split='train'):
#     dataset_name = dataset_map[dataset_name]
#     dataset_filename = "data." + dataset_name + "_dataset"
#     datasetlib = importlib.import_module(dataset_filename)

#     dataset = None
#     target_dataset_name = dataset_name.replace('_', '') + 'dataset'
#     for name, cls in datasetlib.__dict__.items():
#         if name.lower() == target_dataset_name.lower() \
#            and issubclass(cls, BaseDataset):
#             dataset = cls

#     if dataset is None:
#         raise NotImplementedError("In %s.py, there should be a subclass of "
#                         "BaseDataset with class name that matches %s in "
#                         "lowercase." % (dataset_filename, target_dataset_name))
#     return dataset



def create_dataset(dataset_name, split, opt):
    data_loader = CustomDatasetDataLoader(dataset_name, split, opt)
    dataset = data_loader.load_data()
    return dataset


class CustomDatasetDataLoader():
    def __init__(self, dataset_name, split, opt):
        self.opt = opt
        noiseL = opt.train_noiseL if split=='train' else opt.val_noiseL
        print("using %s_noiseL = %s"
              % ('train' if split=='train' else 'val', str(noiseL)))
        #
        self.dataset = ImageDataset(opt, split, dataset_name, noiseL)
        self.imio = self.dataset.imio
        print("dataset [%s(%s)] created" % (dataset_name, split))
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size if split=='train' else 1,
            shuffle=opt.shuffle and split=='train',
            num_workers=int(opt.load_thread),
            drop_last=True)

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the dataset"""
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            if i >= self.opt.max_dataset_size:
                break
            yield data

