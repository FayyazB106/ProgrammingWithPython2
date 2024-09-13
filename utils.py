import matplotlib.pyplot as plt
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import numpy as np
from collections import OrderedDict
from PIL import Image

def sort_data():
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(15),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    
    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    # Load datasets
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    # DataLoaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
    
    return trainloader, validloader, testloader, train_data

def create_model(architecture, learning_rate, dropout, gpu, train_data):
    # Download selected model
    if architecture == 'vgg11':
        model = models.vgg11(pretrained=True)
    elif architecture == 'vgg13':
        model = models.vgg13(pretrained=True)
    elif architecture == 'vgg16':
        model = models.vgg16(pretrained=True)
    else:
        print('Sorry, requested architecture unavailable.')
    
    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace classifier
    model.classifier = nn.Sequential(OrderedDict([('input', nn.Linear(25088, 256)),
                                                  ('relu1', nn.ReLU()),
                                                  ('drop1', nn.Dropout(dropout)),
                                                  ('h1', nn.Linear(256, 128)),
                                                  ('relu2', nn.ReLU()),
                                                  ('drop2', nn.Dropout(dropout)),
                                                  ('h2', nn.Linear(128, 102)),
                                                  ('output', nn.LogSoftmax(dim=1))]))

    # Define loss function and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)
    
    # Turn on GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() and gpu else "cpu")
    model.to(device);
    
    # Map classes to indices
    model.class_to_idx = train_data.class_to_idx
    
    return model, criterion, optimizer, device

def train_model(epochs, model, criterion, optimizer, device, trainloader, validloader):
    # Hyperparameters
    epochs = epochs
    steps = 0
    train_loss = 0
    print_every = 5
    
    # Track training and validation losses
    train_losses, valid_losses = [], []
    
    # Training and validation
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Training
            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                
                with torch.no_grad():
                    for inputs, labels in validloader:
                        # Input validation data
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        valid_loss += batch_loss.item()
                        
                        # Validate
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                        
                # Track training and validation losses
                train_losses.append(train_loss/len(trainloader))
                valid_losses.append(valid_loss/len(validloader))
                
                # Print results for each step
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {train_loss/print_every:.3f}.. "
                      f"Valid loss: {valid_loss/len(validloader):.3f}.. "
                      f"Valid accuracy: {accuracy/len(validloader):.3f}")
                train_loss = 0
                model.train()
    
    return model, optimizer

def test_model(testloader, model, device):
    correct = 0
    total = 0
    model.eval()
    
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            _, predicted = torch.max(output.data.cpu(), 1)
            total += labels.size(0)
            correct += (predicted == labels.cpu()).sum().item()
    print("Test accuracy is: "+"{:.2%}".format(correct / total))    

def save_checkpoint(path, dropout, architecture, learning_rate, optimizer, model):
    # Saving checkpoint
    checkpoint = {'input_size': 25088,
                  'output_size': 102,
                  'hidden_layers': [each.out_features for each in model.classifier if isinstance(each, torch.nn.Linear)],
                  'state_dict': model.state_dict(),
                  'epochs': epochs,
                  'optimizer_state_dict': optimizer.state_dict(),
                  'class_to_idx': model.class_to_idx,
                  'loss': loss,
                  'drop': drop,
                  'lr': lr,
                  'arch': arch}
    
    torch.save(checkpoint, path)
    print('Model checkpoint saved')
    

def load_checkpoint(filepath):
    # Load checkpoint
    saved_checkpoint = torch.load(filepath)

    # Download selected model
    if saved_checkpoint['arch'] == 'vgg11':
        new_model = models.vgg11(pretrained=True)
    elif saved_checkpoint['arch'] == 'vgg13':
        new_model = models.vgg13(pretrained=True)
    elif saved_checkpoint['arch'] == 'vgg16':
        new_model = models.vgg16(pretrained=True)
    else:
        print('Sorry, requested architecture unavailable.')

    hidden_layers = {}
    for idx, layer in enumerate(saved_checkpoint['hidden_layers']):
        if(idx == 0):
            hidden_layers[f'Layer {idx+1}'] = nn.Linear(saved_checkpoint['input_size'], layer)
        else:
            hidden_layers[f'Layer {idx+1}'] = nn.Linear(saved_checkpoint['hidden_layers'][idx-1], layer)
        hidden_layers[f'relu{idx+1}'] = nn.ReLU()
        hidden_layers[f'dropout {idx+1}'] = nn.Dropout(p=0.2)

    # Output layer
    n_final_layer = len(saved_checkpoint['hidden_layers']) + 1
    hidden_layers[f'Layer {n_final_layer}'] = nn.Linear(saved_checkpoint['hidden_layers'][-1], saved_checkpoint['output_size'])
    hidden_layers[f'output'] = nn.LogSoftmax(dim=1)

    # Assemble model
    new_model.classifier = nn.Sequential(OrderedDict(hidden_layers))
    missing_keys, unexpected_keys = new_model.load_state_dict(saved_checkpoint['state_dict'], strict=False)
    new_model.class_to_idx = saved_checkpoint['class_to_idx']

    return new_model

def process_image(image):
    img = Image.open(image)
    img_transforms = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])

    return img_transforms(img)

def predict(image_path, model, topk, cat_to_name):
    image = process_image(image_path)
    image = image.unsqueeze_(0)
    image = image.float()
    model.eval()

    with torch.no_grad():
        # Make prediction
        output = model(image)
        prob, idxs = torch.topk(output, topk)

        # Map top 5 indices to classes
        idxs = np.array(idxs)
        idx_to_class = {val:key for key, val in model.class_to_idx.items()}
        classes = [idx_to_class[idx] for idx in idxs[0]]

        names = []
        for c in classes:
            names.append(cat_to_name[str(c)])

        return prob, names


