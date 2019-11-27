import importlib
from torch.utils.data import Dataset, DataLoader

def CreateDataLoader(opt):
    
    dataset_filename = "data." + opt.model_name + '_dataset'
    datasetllib = importlib.import_module(dataset_filename)
    dataset = None
    target_dataset_name = opt.model_name.replace('_', '') + 'dataset'
    for name, cls in datasetllib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
           and issubclass(cls, Dataset):
            dataset = cls
    if dataset is None:
        raise NotImplementedError("In %s.py, there should be a subclass of torch.utils.data.Dataset with class name that matches %s in lowercase." % (dataset_filename, target_dataset_name))
    data_loader = DataLoader(dataset=dataset(opt),
                             batch_size=opt.batch_size,
                             pin_memory=True,
                             shuffle=opt.is_train,
                             drop_last=True,
                             num_workers=opt.n_threads)
                             
    return data_loader
