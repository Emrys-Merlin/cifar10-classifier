import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam, SGD
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms

import numpy as np

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Loss, Accuracy
from ignite.handlers import EarlyStopping, ModelCheckpoint, Timer

from ignite_extension import create_supervised_trainer_with_metrics

# from vgg import VGG
from adapted_vgg import vgg16_bn

import argparse

def get_loaders(bs=100, datadir='./cifar10', valid_split=5000):
    trafo = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    aug = transforms.Compose([
        # transforms.ColorJitter(brightness=.2,contrast=.2, saturation=.2, hue=.2),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(5/32, 5/32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_data =  torchvision.datasets.CIFAR10(datadir, train=True,
                                               download=True, transform=aug)
    valid_data = torchvision.datasets.CIFAR10(datadir, train=True,
                                              download=True, transform=trafo)
    indices = list(range(len(train_data)))
    np.random.shuffle(indices)
    train_idx, valid_idx = indices[valid_split:], indices[:valid_split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    valid_loader = torch.utils.data.DataLoader(valid_data,
                                               batch_size=bs,
                                               drop_last=False,
                                               sampler=valid_sampler)
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=bs,
                                               sampler=train_sampler)

    test_data =  torchvision.datasets.CIFAR10(datadir, train=False,
                                              download=True, transform=trafo)

    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=bs,
                                              shuffle=False,
                                              drop_last=False)

    return train_loader, valid_loader, test_loader

def print_floyd_metric(name, value, step=None):
    if step is None:
        s = f'{{"metric": "{name}", "value": {value:.5f}}}'
    else:
        s = f'{{"metric": "{name}", "value": {value:.5f}, "step": {step:d}}}'

    print(s)

def evaluate(engine, step=None, prefix=''):
    metrics = engine.state.metrics
    avg_accuracy = metrics['accuracy']
    avg_loss = metrics['loss']
    print_floyd_metric(prefix+'loss', avg_loss, step)
    print_floyd_metric(prefix+'accuracy', avg_accuracy, step)


def run(args):
    epochs = args.e
    patience = args.p
    lr_patience = args.lp
    lr = args.lr
    batch_size = args.b
    data_dir = args.dir
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_loader, valid_loader, test_loader = get_loaders(batch_size, data_dir)
    model = vgg16_bn(num_classes=10) #VGG('VGG16')
    if args.adam:
        print('Using adam optimizer.')
        optim = Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    else:
        print('Using SGD optimizer.')
        optim = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    trainer = create_supervised_trainer_with_metrics(model, optim, F.cross_entropy,
                                                     metrics={'accuracy': Accuracy(),
                                                              'loss': Loss(F.cross_entropy)},
                                                     device=device)
    evaluator = create_supervised_evaluator(model, metrics={'accuracy': Accuracy(),
                                                            'loss': Loss(F.cross_entropy)},
                                            device=device)

    evaluator.register_events('validation_completed')

    # Check early stopping conditions after validation is completed
    def score_function(engine):
        val_loss = engine.state.metrics['loss']
        return -val_loss

    handler = EarlyStopping(patience=patience, score_function=score_function, trainer=trainer)
    evaluator.add_event_handler('validation_completed', handler)

    # Save model checkpoints
    handler = ModelCheckpoint(args.dir, 'cifar10', save_interval=10)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, handler, {'model': model, 'optim': optim})

    handler = ModelCheckpoint(args.dir, 'cifar10_end', save_interval=1)
    # trainer.add_event_handler(Events.EXCEPTION_RAISED, handler, {'model': model, 'optim': optim})
    trainer.add_event_handler(Events.COMPLETED, handler, {'model': model, 'optim': optim})

    # Setup timer
    timer = Timer(average=True)
    timer.attach(trainer, step=Events.EPOCH_COMPLETED)

    epoch_timer = Timer(average=False)
    epoch_timer.attach(trainer, start=Events.EPOCH_STARTED, pause=Events.EPOCH_COMPLETED,)

    # Set up learning rate scheduling
    if lr_patience != 0:
        scheduler = ReduceLROnPlateau(optim, patience=lr_patience, verbose=True)
        @evaluator.on('validation_completed')
        def scheduler_step(engine):
            scheduler.step(engine.state.metrics['loss'])

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_validation_results(engine):
        step = engine.state.epoch
        evaluate(trainer, step=step)
        evaluator.run(valid_loader)
        evaluate(evaluator, step=step, prefix='valid_')
        evaluator.fire_event('validation_completed')
        print_floyd_metric('zz_time', epoch_timer.value(), step)

    @trainer.on(Events.COMPLETED)
    def log_test_results(engine):
        step = engine.state.epoch
        evaluator.run(test_loader)
        evaluate(evaluator, step=step, prefix='test_')


    trainer.run(train_loader, max_epochs=epochs)
    total_time = timer.total / 3600
    print(f'Total training time: {total_time:.2f}h')
    print(f'Average time per epoch: {timer.value()}s')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run training and evaluation of a CIFAR10 experiment')
    parser.add_argument('-lr', default=.1,
                        help='Initial learning rate for training.')
    parser.add_argument('-e', default=10000,
                        help='Maximum number of epochs.')
    parser.add_argument('-p', default=30,
                        help='Patience for how long to wait for the validation loss to decrease before the training ends.')
    parser.add_argument('-lp', default=15,
                        help='Learning rate patience after which the learning rate will be decreased. If set to 0 learning rate will be static throughout training.')
    parser.add_argument('-b', default=128,
                        help='Batch size')
    parser.add_argument('-dir', default='./cifar10',
                        help='Directory where to put the CIFAR10 data set.')
    parser.add_argument('-adam', action='store_true',
                        help='If present, use Adam optimizer else use SGD with 0.9 momentum')

    args = parser.parse_args()

    run(args)
