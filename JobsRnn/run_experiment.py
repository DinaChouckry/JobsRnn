from tensorboardX import SummaryWriter
from .data_preparation.load_dataset import DatasetInitializer , DatasetSplittingGenerator
from .data_preparation.transformations import ToIdx, transformation_helpers , dict_builder
from .model.network import RNN
from .model.nnTrainer import nnTrainer
from .utils.utils import save_checkpoint

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
    vocab = dict_builder(args).get_dict()
    embedding = transformation_helpers.generate_embedding(args)
    # TODO : get output size form data ,not as argument
    jobs_rnn = RNN(embedding=embedding,
                   rnn_type=args.rnn_type,
                   hidden_size=args.hidden_size,
                   num_layers=args.num_layers,
                   dropout=args.dropout,
                   output_size=args.output_size,
                   bidirectional=args.bidirectional).to(device)

    print('=' * 100)
    print('Model log:\n')
    print(RNN)
    print('- RNN input embedding requires_grad={}'.format(jobs_rnn.embedding.weight.requires_grad))
    print('=' * 100 + '\n')


    jobs_rnn_optimizer = optim.Adam([p for p in jobs_rnn.parameters() if p.requires_grad], lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    operational_model = nnTrainer(jobs_rnn, args, transformation_helpers.vocab, train_dataset, train_tb_logger, criterion, device)

    """ End Model Definition """

    """ ##############################   RUN EXPERIMENT   ############################## """
    """ Initialize metrics """
    min_error = 1e8
    """End initialize metrics """
    """ START TRAINING PHASE """
    for epoch in range(args.start_epoch, args.epochs + args.start_epoch):
        train_error, train_f1score = operational_model.train_model(train_loader, jobs_rnn_optimizer, epoch, train_tb_logger, min_error)
        print("Training Epoch: {} Finished, error: {}".format(epoch, train_error))
        print("Training F1Score: {}".format(train_f1score))
        valid_error, valid_f1score = operational_model.validate_model(valid_loader, epoch, valid_tb_logger)
        print("Validation For Epoch: {} Finished, error: {}".format(epoch, valid_error))
        print("Validation F1Score: {}".format(valid_f1score))

        # remember best loss
        valid_error_with_respect_to_train = valid_error - abs(train_error - valid_error)
        is_best = valid_error_with_respect_to_train < min_error
        min_error = min(valid_error_with_respect_to_train, min_error)
        save_checkpoint(args,
                        {
                            'epoch': epoch + 1,
                            'state_dict': jobs_rnn.state_dict(),
                            'best_metric': min_error,
                            'optimizer': jobs_rnn_optimizer.state_dict(),
                        } , is_best)

    """ END TRAINING PHASE """
    """" LOG Experiment parameters"""
    train_tb_logger.add_text('config', str(args))

    """ START TESTING PHASE """
    print("Start Testing")
    best_checkpoint_file = args.weights_dir + args.exp_name + '/model_best.pth.tar'
    print("=> loading checkpoint '{}'".format(best_checkpoint_file))
    rnn_checkpoint = torch.load(best_checkpoint_file)
    epoch = rnn_checkpoint['epoch'] - 1  #
    jobs_rnn.load_state_dict(rnn_checkpoint['state_dict'])
    operational_model.test_model(test_loader, epoch, test_tb_logger)
    """ END TESTING PHASE """

    """ ##############################   END RUN EXPERIMENT   ############################## """
    """ Close Loggers """
    train_tb_logger.close()
    valid_tb_logger.close()
    test_tb_logger.close()



