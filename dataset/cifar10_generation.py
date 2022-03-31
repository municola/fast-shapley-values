import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt     
import numpy as np
import argparse
import pandas as pd
from torchvision import models
from tqdm import tqdm

# Read in Arguments from Commandline
parser = argparse.ArgumentParser(description="")
parser.add_argument("--model_version", default="resnet50")
parser.add_argument("--show_images", default=False)
args = parser.parse_args()

# Reproducability
torch.manual_seed(1234)

# Constants
showImages = args.show_images
batch_size = 4

# In order to match the resnet input size we use Resizing with bilinear interpolation
# We normalize the images for the range [-1,1]
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize((224,224)),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data/images/cifar10', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=False, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data/images/cifar10', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


if showImages:
    # get some random training imagesimg
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    img = torchvision.utils.make_grid(images)

    # Bring back to [0,1]
    img = img*0.5 + 0.5

    # Print Image
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

    # Print labels    
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))


# Create Model Resnet model without top layer
# Since we want to extract the features of the second last layer
if args.model_version == 'resnet50':
    model = models.resnet50(pretrained=True)
elif args.model_version == 'resnet18':
    model = models.resnet18(pretrained=True)
else:
    raise ValueError("Specify modelVersion as resnet18 or resnet50")

newmodel = torch.nn.Sequential(*(list(model.children())[:-1]))
newmodel.eval()

# Create Dataframe
features = pd.DataFrame()
lables_df = pd.DataFrame()

# Create Features of Trainset
print("Starting Trainset Feature extraction..")
with torch.no_grad():
    with tqdm(trainloader, unit="batch") as tepoch:
        for images, labels in tepoch:
            # calculate encoding by running images through the network
            encoding = newmodel(images)
            encoding = encoding.squeeze(2)
            encoding = encoding.squeeze(2)

            for i in range (0,batch_size):
                # Append features
                rowdf = pd.DataFrame([encoding[i]])
                features = pd.concat([features,rowdf], ignore_index=True)

                # Append labels
                rowdf = pd.DataFrame([labels[i]])
                lables_df = pd.concat([lables_df, rowdf], ignore_index=True)

features.to_csv(r'data/features/cifar10/train_features.csv',index=False, header=True)
lables_df.to_csv(r'data/features/cifar10/train_labels.csv',index=False, header=True)


# Create Dataframe
features = pd.DataFrame()
lables_df = pd.DataFrame()

# Create Features of Testset
print("Starting Testset Feature extraction..")
with torch.no_grad():
    with tqdm(testloader, unit="batch") as tepoch:
        for images, labels in tepoch:
            # calculate encoding by running images through the network
            encoding = newmodel(images)
            encoding = encoding.squeeze(2)
            encoding = encoding.squeeze(2)

            for i in range (0,batch_size):
                # Append features
                rowdf = pd.DataFrame([encoding[i]])
                features = pd.concat([features,rowdf], ignore_index=True)

                # Append labels
                rowdf = pd.DataFrame([labels[i]])
                lables_df = pd.concat([lables_df, rowdf], ignore_index=True)

features.to_csv(r'data/features/cifar10/test_features.csv', index=False, header=True)
lables_df.to_csv(r'data/features/cifar10/test_labels.csv',index=False, header=True)
