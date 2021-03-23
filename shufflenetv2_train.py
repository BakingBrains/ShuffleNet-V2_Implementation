'''
Name: Syed
Description: ShuffleNet V2
Organization: Sandlogic Technologies
E-mail: syed.abdul@sandlogic.com
Contact: 9902980465
'''
from IPython.core.interactiveshell import InteractiveShell
from torchvision import transforms, datasets, models
import torch
from torch import optim, cuda
from torch.utils.data import DataLoader, sampler
import torch.nn as nn

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import os

from PIL import Image
from timeit import default_timer as timer

InteractiveShell.ast_node_interactivity = 'all'


class Dataloader:
  def __init__(self,traindir,validdir):
    self.traindir = traindir
    self.validdir = validdir


  def data_loader(self):
    save_file_name = 'SL_shufflenet_v2_x1_0.pt'
    #checkpoint_path = 'shufflenet_v2_x1_0-Infer.pt'

    # Change to fit hardware
    batch_size = 32

    # Whether to train on a gpu
    use_gpu = cuda.is_available()
    print(f'Train on gpu: {use_gpu}')

        # Number of gpus
    if use_gpu:
      gpu_count = cuda.device_count()
      print(f'{gpu_count} gpus detected.')
      if gpu_count > 1:
        multi_gpu = True
      else:
        multi_gpu = False

    categories = []
    img_categories = []
    n_train = []
    n_valid = []
    hs = []
    ws = []

    # Iterate through each category
    for d in os.listdir(self.traindir):
      categories.append(d)

      # Number of each image
      train_imgs = os.listdir(self.traindir + d)
      valid_imgs = os.listdir(self.validdir + d)
      n_train.append(len(train_imgs))
      n_valid.append(len(valid_imgs))

      # Find stats for train images
      for i in train_imgs:
        img_categories.append(d)
        img = Image.open(self.traindir + d + '/' + i)
        img_array = np.array(img)
        # Shape
        hs.append(img_array.shape[0])
        ws.append(img_array.shape[1])

    # Dataframe of categories
    cat_df = pd.DataFrame({'category':categories,
                            'n_train': n_train,
                            'n_valid': n_valid}). \
    sort_values('category')

    # Dataframe of training images
    image_df = pd.DataFrame({
        'category': img_categories,
        'height': hs,
        'width': ws
        })

    cat_df.sort_values('n_train', ascending=False, inplace=True)
    cat_df.head()
    cat_df.tail()

    # Image transformations
    image_transforms = {
            # Train uses data augmentation
            'train': transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ]),
    'val': transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        }

    # Datasets from each folder
    self.data = {
          'train':
              datasets.ImageFolder(root=self.traindir, transform=image_transforms['train']),
          'val':
              datasets.ImageFolder(root=self.validdir, transform=image_transforms['val']),
        }

    # Dataloader iterators
    self.dataloaders = {
            'train': DataLoader(self.data['train'], batch_size=batch_size, shuffle=True),
            'val': DataLoader(self.data['val'], batch_size=batch_size, shuffle=True),
        }

    trainiter = iter(self.dataloaders['train'])
    features, labels = next(trainiter)

    n_classes = len(cat_df)
    print(f'There are {n_classes} different classes.')

    len(self.data['train'].classes)

    model = models.shufflenet_v2_x1_0(pretrained=True, progress=True) #shufflenet_v2_x0_5, shufflenet_v2_x1_5, shufflenet_v2_x1_5
    print(model)

    # Freeze early layers
    for param in model.parameters():
      param.requires_grad = False
    
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, n_classes)
    model.num_classes = n_classes
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
    print(categories)
    if use_gpu:
        model = model.cuda()

    model.class_to_idx = self.data['train'].class_to_idx
    model.idx_to_class = {
          idx: class_
          for class_, idx in model.class_to_idx.items()
        }

    list(model.idx_to_class.items())[:10]
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    return model,criterion,optimizer,self.dataloaders['train'],self.dataloaders['val'],save_file_name


class SL_shufflenetV2:
    def __init__(self,model, criterion,optimizer,train_loader,valid_loader,save_file_name,max_epochs_stop,n_epochs,print_every):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.save_file_name = save_file_name
        self.max_epochs_stop = max_epochs_stop
        self.n_epochs = n_epochs
        self.print_every = print_every


    def train(self):
        # Early stopping intialization
        self.epochs_no_improve = 0
        valid_loss_min = np.Inf

        valid_max_acc = 0
        history = []

        try:
            print(f'Model has been trained for: {self.model.epochs} epochs.\n')
        except:
            self.model.epochs = 0
            print(f'Starting Training from Scratch.\n')

        overall_start = timer()

        # Main loop
        for epoch in range(self.n_epochs):

            # keep track of training and validation loss each epoch
            train_loss = 0.0
            valid_loss = 0.0

            train_acc = 0
            valid_acc = 0

            # Set to training
            self.model.train()
            start = timer()

            # Training loop
            for ii, (data, target) in enumerate(self.train_loader):
                # Tensors to gpu
                #if use_gpu:
                data, target = data.cuda(), target.cuda()

                # Clear gradients
                self.optimizer.zero_grad()
                output = self.model(data)

                # Loss and backpropagation of gradients
                loss = self.criterion(output, target)
                loss.backward()

                # Update the parameters
                self.optimizer.step()

                train_loss += loss.item() * data.size(0)

                # Calculate accuracy by finding max log probability
                _, pred = torch.max(output, dim=1)
                correct_tensor = pred.eq(target.data.view_as(pred))
                # Need to convert correct tensor from int to float to average
                accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
                # Multiply average accuracy times the number of examples in batch
                train_acc += accuracy.item() * data.size(0)

                # Track training progress
                print(f'Epoch: {epoch}\t{100 * (ii + 1) / len(self.train_loader):.2f}% complete. {timer() - start:.2f} seconds elapsed in epoch.', end='\r')

            # After training loops ends, start validation
            else:
                self.model.epochs += 1

                # Don't need to keep track of gradients
                with torch.no_grad():
                    # Set to evaluation mode
                    self.model.eval()

                    # Validation loop
                    for data, target in self.valid_loader:
                        # Tensors to gpu
                        #if use_gpu:
                        data, target = data.cuda(), target.cuda()

                        # Forward pass
                        output = self.model(data)

                        # Validation loss
                        loss = self.criterion(output, target)
                        # Multiply average loss times the number of examples in batch
                        valid_loss += loss.item() * data.size(0)

                        # Calculate validation accuracy
                        _, pred = torch.max(output, dim=1)
                        correct_tensor = pred.eq(target.data.view_as(pred))
                        accuracy = torch.mean(
                            correct_tensor.type(torch.FloatTensor))
                        # Multiply average accuracy times the number of examples
                        valid_acc += accuracy.item() * data.size(0)

                    # Calculate average losses
                    train_loss = train_loss / len(self.train_loader.dataset)
                    valid_loss = valid_loss / len(self.valid_loader.dataset)

                    # Calculate average accuracy
                    train_acc = train_acc / len(self.train_loader.dataset)
                    valid_acc = valid_acc / len(self.valid_loader.dataset)

                    history.append([train_loss, valid_loss, train_acc, valid_acc])
                    # Print training and validation results
                    if (epoch + 1) % self.print_every == 0:
                        print(f'\nEpoch: {epoch} \tTraining Loss: {train_loss:.4f} \tValidation Loss: {valid_loss:.4f}')
                        print(f'\t\tTraining Accuracy: {100 * train_acc:.2f}%\t Validation Accuracy: {100 * valid_acc:.2f}%')
                        torch.save(self.model, "ShufflenetV2_model" + "_" + str(epoch) + ".pt")

                        # Save the model if validation loss decreases
                    if valid_loss < valid_loss_min:
                        # Save model
                        torch.save(self.model.state_dict(), self.save_file_name)
                        # Track improvement
                        epochs_no_improve = 0
                        valid_loss_min = valid_loss
                        valid_best_acc = valid_acc
                        best_epoch = epoch

                    # Otherwise increment count of epochs with no improvement
                    else:
                        epochs_no_improve += 1
                        # Trigger early stopping
                        if epochs_no_improve >= self.max_epochs_stop:
                            print(f'\nEarly Stopping! Total epochs: {epoch}. Best epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_acc:.2f}%')
                            total_time = timer() - overall_start
                            print(f'{total_time:.2f} total seconds elapsed. {total_time / (epoch + 1):.2f} seconds per epoch.')

                            # Load the best state dict
                            self.model.load_state_dict(torch.load(self.save_file_name))

                            # Attach the optimizer
                            self.model.optimizer = self.optimizer

                            # Format history
                            history = pd.DataFrame(
                                history,
                                columns=[
                                    'train_loss', 'valid_loss', 'train_acc',
                                    'valid_acc'
                                ])
                            return self.model, history

        # Attach the optimizer
        self.model.optimizer = self.optimizer
        # Record overall time and print out stats
        total_time = timer() - overall_start
        print(f'\nBest epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_acc:.2f}%')
        print(f'{total_time:.2f} total seconds elapsed. {total_time / (epoch):.2f} seconds per epoch.')
        # Format history
        history = pd.DataFrame(
            history,
            columns=['train_loss', 'valid_loss', 'train_acc', 'valid_acc'])
        return self.model, history

SL_model, SL_criterion, SL_optimizer, SL_train, SL_val, savefile=Dataloader('/content/data/hymenoptera_data/train/','/content/data/hymenoptera_data/val/').data_loader()
shuffleNEt = SL_shufflenetV2(SL_model, SL_criterion, SL_optimizer, SL_train, SL_val, save_file_name=savefile, max_epochs_stop=2, n_epochs=5, print_every=1).train()















