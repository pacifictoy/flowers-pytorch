from collections import OrderedDict

from torchvision import datasets, transforms, models
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from workspace_utils import keep_awake, active_session
from PIL import Image
import json


class MyModel:
    def __init__(self, category_name_filepath='cat_to_name.json', use_gpu=False, data_dir='flowers'):
        if use_gpu and torch.cuda.is_available():
            self.device = "cuda"
        else:
            if use_gpu and not torch.cuda.is_available():
                print('not using gpu despite instructed because gpu is not available')
            self.device = "cpu"
        self.cat_to_name = None
        self.class_to_idx = None
        self.train_transforms = None
        self.test_transforms = None
        self.train_dataloader = None
        self.validation_dataloader = None
        self.test_dataloader = None
        self.data_dir = data_dir
        self.init_data(category_name_filepath)

    def init_data(self, category_name_filepath):
        train_dir = self.data_dir + '/train'
        valid_dir = self.data_dir + '/valid'
        test_dir = self.data_dir + '/test'

        train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                               transforms.RandomResizedCrop(224),
                                               transforms.RandomHorizontalFlip(),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406],
                                                                    [0.229, 0.224, 0.225])])

        test_transforms = transforms.Compose([transforms.Resize(255),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406],
                                                                   [0.229, 0.224, 0.225])])

        # Load the datasets with ImageFolder
        train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
        validation_datasets = datasets.ImageFolder(valid_dir, transform=test_transforms)
        test_datasets = datasets.ImageFolder(test_dir, transform=test_transforms)

        # Using the image datasets and the trainforms, define the dataloaders
        self.train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)
        self.validation_dataloader = torch.utils.data.DataLoader(validation_datasets, batch_size=64)
        self.test_dataloader = torch.utils.data.DataLoader(test_datasets, batch_size=64)

        self.class_to_idx = train_datasets.class_to_idx

        with open(category_name_filepath, 'r') as f:
            self.cat_to_name = json.load(f)

        print('Model Init Complete')

    def create_model(self, base_model='vgg16', hidden_unit_num=4096):
        if base_model == 'vgg16':
            model = models.vgg16(pretrained=True)
        elif base_model == 'vgg11':
            model = models.vgg11(pretrained=True)

        for param in model.parameters():
            param.required_grad = False

        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(25088, hidden_unit_num)),
            ('relu1', nn.ReLU()),
            ('DO1', nn.Dropout(p=0.2)),
            ('fc2', nn.Linear(hidden_unit_num, hidden_unit_num)),
            ('relu2', nn.ReLU()),
            ('DO2', nn.Dropout(p=0.2)),
            ('fc3', nn.Linear(hidden_unit_num, 102)),
            ('output', nn.LogSoftmax(dim=1))
        ]))

        model.classifier = classifier

        model.to(self.device)

        print('model created')

        return model

    def create_optimizer(self, model, learning_rate=0.01):
        optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
        print('optimizer created')
        return optimizer

    def train_model(self, model, optimizer, epochs=5):
        running_loss = 0
        criterion = nn.NLLLoss()
        with active_session():
            for epoch in range(epochs):
                train_num = 0
                for inputs, labels in self.train_dataloader:

                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    optimizer.zero_grad()

                    logps = model.forward(inputs)
                    loss = criterion(logps, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    print("Training item number:", train_num)
                    train_num += 1
                else:
                    test_loss = 0
                    accuracy = 0
                    model.eval()
                    with torch.no_grad():
                        for inputs, labels in self.validation_dataloader:
                            inputs, labels = inputs.to(self.device), labels.to(self.device)
                            logps = model.forward(inputs)
                            batch_loss = criterion(logps, labels)
                            test_loss += batch_loss.item()

                            # calc accuracy
                            ps = torch.exp(logps)
                            top_prob, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor))

                    print(f"Epoch: {epoch + 1} / {epochs}"
                          f"Training Loss: {running_loss / len(self.train_dataloader):.3f}"
                          f"Validation Loss: {test_loss / len(self.validation_dataloader):.3f}"
                          f"Validation Accuracy: {accuracy / len(self.validation_dataloader):.3f}"
                          )
        print("======Done Training Model, YAY!=======")

    def test_model(self, model):
        accuracy = 0
        model.eval()
        with torch.no_grad():
            for inputs, labels in self.test_dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                logps = model.forward(inputs)

                ps = torch.exp(logps)

                # calc accuracy
                top_prob, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                acrcy = torch.mean(equals.type(torch.FloatTensor))
                print(acrcy)
                accuracy += acrcy

        print(f"Test Accuracy: {accuracy / len(self.test_dataloader):.3f}")

    def save_checkpoint(self, model, optimizer, dir='.'):
        model_dict = model.state_dict()

        checkpoint = {
            'model_dict': model_dict,
            'optimizer_dict': optimizer.state_dict(),
            'epochs': self.epochs,
            'class_to_idx': self.class_to_idx
        }
        filepath = dir+'/checkpoint.pth'
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath):
        if self.device == 'gpu':
            checkpoint = torch.load(filepath)
        else:
            checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
        model = self.create_model()
        model.load_state_dict(checkpoint['model_dict'])
        optimizer = self.create_optimizer(model)
        optimizer.load_state_dict(checkpoint['optimizer_dict'])
        self.class_to_idx = checkpoint['class_to_idx']
        print('loading checkpoint completed')
        return model, optimizer

    def process_image(self, image):
        ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
            returns an Numpy array
        '''
        my_transforms = transforms.Compose([transforms.Resize(255),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],
                                                                 [0.229, 0.224, 0.225])])
        im = Image.open(image)
        img = my_transforms(im)  # use transforms for test data to resize, normalize etc
        return img

    def predict(self, image_path, model, topk=5):
        ''' Predict the class (or classes) of an image using a trained deep learning model.
        '''
        img = self.process_image(image_path)
        imgc = img.to(self.device)
        ps = torch.exp(model(imgc.unsqueeze_(0)))
        top_pred, top_class = ps.topk(topk)

        # create dctionary for flowername:probability
        idx_to_class = dict([(value, key) for key, value in self.class_to_idx.items()])
        top_class_np = top_class.cpu().numpy().squeeze()
        class_ids = [idx_to_class[idx] for idx in top_class_np]
        class_names = [self.cat_to_name[idx] for idx in class_ids]
        top_pred_np = top_pred.detach().cpu().numpy().squeeze()
        answer = {class_names[i]:top_pred_np[i] for i in range(len(class_names))}
        return answer
