Cifar10 experiments
=======================

Training task for me to play around with pytorch and pytorch-ignite in particular on a computer vision task.

Prerequisites
----------------

- Python 3.6+
- pytorch 1.0+
- pytorch-ignite 0.2(+)
- (floyd-cli 0.11)

Training procedure
----------------------

The training can be started using `python train.py`. This script instantiates a VGG16 net (with batch normalization) slightly adapted to the CIFAR10 setting and performs a training that should converge in around 300 epochs. If desired, many hyperparameters of the training can be set using command line arguments. They can be displayed using `python train.py --help`. 

The learning rate will be decreased on plateaus of the validation loss. The same criterion with higher patience is used to terminate the training.

I was able to reach 93% test set accuracy. Interestingly, this was possible without any dropout layers.

I used [floydhub](floydhub.com) for my experiments and the output of my training script uses the format used for floydhub to automatically plot the loss and accuracy time series (which I find very convenient). This also explains the existence of the floyd_requirements.txt file. I was able to run the script on the service via `floyd run --gpu --env pytorch-1.0 "python train.py"`. Please note that you have to pip-install the floyd-cli package with the necessary setup for this to work.
