from tensorboardX import SummaryWriter
from .data_preparation.load_dataset import DatasetInitializer , DatasetSplittingGenerator
from .data_preparation.transformations import ToIdx, transformation_helpers
from .model.network import RNN
from .model.nnTrainer import nnTrainer

import torch
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
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
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=train_dataset.collate_fn, **kwargs)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=train_dataset.collate_fn, **kwargs)
    test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=train_dataset.collate_fn, **kwargs)



    """ Define Model, Optimization """
    embedding = transformation_helpers.generate_embedding(args)
    rnn = RNN(embedding=embedding,
              rnn_type=args.rnn_type,
              hidden_size=args.hidden_size,
              num_layers=args.num_layers,
              dropout=args.dropout,
              output_size=args.output_size,
              bidirectional=args.bidirectional).to(device)

    rnn_optimizer = optim.Adam([p for p in rnn.parameters() if p.requires_grad], lr=args.learning_rate,weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    operational_model = nnTrainer(rnn, args, transformation_helpers.vocab ,train_dataset, train_tb_logger, criterion, device)

    """ End Model Definition """

    """ ##############################   RUN EXPERIMENT   ############################## """
    """ Initialize metrics """
    min_error = 1e8
    """ START TRAINING PHASE """
    for epoch in range(args.start_epoch, args.epochs + args.start_epoch):
        train_error = operational_model.train_model(train_loader, rnn_optimizer, epoch, train_tb_logger, min_error)
        print("Training Epoch: {} Finished, error: {}".format(epoch, train_error))
        valid_error = operational_model.validate_model(valid_loader, epoch, valid_tb_logger)
        print("Validation For Epoch: {} Finished, error: {}".format(epoch, valid_error))
        # remember best loss and save checkpoint
        valid_error_with_respect_to_train = valid_error - abs(train_error - valid_error)
        is_best = valid_error_with_respect_to_train < min_error
        min_error = min(valid_error_with_respect_to_train, min_error)

    """ END TRAINING PHASE """

    """ Close Loggers """
    train_tb_logger.close()
    valid_tb_logger.close()
    test_tb_logger.close()
    manual_tb_test_logger.close()


