from tensorboardX import SummaryWriter
from .data_preparation.load_dataset import DatasetInitializer , DatasetSplittingGenerator
from .data_preparation.transformations import ToIdx, transformation_helpers
import torch
from torchvision import transforms
from torch.utils.data import DataLoader



def run_experiment(args):
    # select main device to run against
    device = torch.device("cuda:" + args.gpu_device_id if not args.cpu_only else "cpu")

    """ LOGGERS END """
    train_tb_logger = SummaryWriter(args.log_dir + args.exp_name + '/train')
    valid_tb_logger = SummaryWriter(args.log_dir + args.exp_name + '/valid')
    test_tb_logger = SummaryWriter(args.log_dir + args.exp_name + '/test')
    manual_tb_test_logger = SummaryWriter(args.log_dir + args.exp_name + '/manual_test')
    """ LOGGERS END """



    """ Dataset & Loaders Initialization """
    jobs_full_data = DatasetInitializer (args)
    train_dataset = DatasetSplittingGenerator(jobs_full_data, args, 'train' ,transform=transforms.Compose([ToIdx(args)]))
    valid_dataset = DatasetSplittingGenerator(jobs_full_data, args, 'valid' ,transform=transforms.Compose([ToIdx(args)]))
    test_dataset  = DatasetSplittingGenerator(jobs_full_data, args, 'test'  ,transform=transforms.Compose([ToIdx(args)]))

    # Loaders Initialization
    kwargs = {'pin_memory': True} if not args.cpu_only else {}
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,  num_workers=args.num_workers, collate_fn=train_dataset.collate_fn, **kwargs)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=train_dataset.collate_fn, **kwargs)
    test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=train_dataset.collate_fn, **kwargs)




    """ Define Model, Optimization """
    embedding = transformation_helpers.generate_embedding(args)
