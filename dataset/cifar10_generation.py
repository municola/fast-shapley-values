import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt     
import numpy as np
import argparse
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
batch_size = 32
print("Using batch_size: ", batch_size)

# In order to match the resnet input size we use Resizing with bilinear interpolation
# We normalize the images for the range [-1,1]
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize((224,224)),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data/images/cifar10', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=False, num_workers=12)

testset = torchvision.datasets.CIFAR10(root='./data/images/cifar10', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=12)

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

# Put of GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using: ", device)

# Create Model Resnet model without top layer
# Since we want to extract the features of the second last layer
if args.model_version == 'resnet50':
    model = models.resnet50(pretrained=True)
elif args.model_version == 'resnet18':
    model = models.resnet18(pretrained=True)
else:
    raise ValueError("Specify model Version as resnet18 or resnet50")

newmodel = torch.nn.Sequential(*(list(model.children())[:-1]))
newmodel.eval()
newmodel.to(device)

# Create Features of Trainset
print("Starting Trainset Feature extraction..")
features = []
labels = []
with torch.no_grad():
    with tqdm(trainloader, unit="batch") as tepoch:
        for images, y in tepoch:
            # calculate encoding by running images through the network
            encoding = newmodel(images.to(device))
            encoding = encoding.squeeze(2)
            encoding = encoding.squeeze(2)
            y = y.unsqueeze(1)

            features.extend(encoding.cpu().detach().numpy())
            labels.extend(y.cpu().detach().numpy())

# Saving numpy array
numpy_features = np.array(features)
numpy_features = numpy_features.astype(np.float64)
numpy_labels = np.array(labels)
numpy_labels = numpy_labels.astype(np.float64)
numpy_labels = numpy_labels

np.save(r'../data/features/cifar10/train_features.npy', numpy_features)
np.save(r'../data/features/cifar10/train_labels.npy', numpy_labels)
numpy_features.tofile(r'../data/features/cifar10/train_features.bin')
numpy_labels.tofile(r'../data/features/cifar10/train_labels.bin')
print('train_numpy_features: ', numpy_features[0,:10])
print('train_numpy_labels: ', numpy_labels[:10])
print('train_labels: ', labels[:10])


# Create Features of Testset
print("Starting Testset Feature extraction..")
features = []
labels = []
with torch.no_grad():
    with tqdm(testloader, unit="batch") as tepoch:
        for images, y in tepoch:
            # calculate encoding by running images through the network
            encoding = newmodel(images.to(device))
            encoding = encoding.squeeze(2)
            encoding = encoding.squeeze(2)
            y = y.unsqueeze(1)

            features.extend(encoding.cpu().detach().numpy())
            labels.extend(y.cpu().detach().numpy())

# Saving numpy array
numpy_features = np.array(features)
numpy_features = numpy_features.astype(np.float64)
numpy_labels = np.array(labels)
numpy_labels = numpy_labels.astype(np.float64)
numpy_labels = numpy_labels

np.save(r'../data/features/cifar10/test_features.npy', numpy_features)
np.save(r'../data/features/cifar10/test_labels.npy', numpy_labels)
numpy_features.tofile(r'../data/features/cifar10/test_features.bin')
numpy_labels.tofile(r'../data/features/cifar10/test_labels.bin')
print('test_numpy_features: ', numpy_features[0,:10])
print('test_numpy_labels: ', numpy_labels[:10])
print('test_labels: ', labels[:10])
