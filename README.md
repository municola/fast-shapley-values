# Advanced Systems Lab - Fast and efficient data valuation


## Datasets

|Name       | Dataset       | Architecture| #Features   | #Labels     |   #Train    |  #Test      |Images     | 
|-----------| ------------- |-------------|------------ |-----------  | ------------| ------------| ----------|
|Random     | Random        |     -       |   [0, ∞]    |   [0,  ∞]   |  [0,  ∞]    |  [0,  ∞]    |  ❌       |
|DogFish    | DogFish[1]    | Inceptionv3 |   2048      |   2         |  900        |  300        |  ❌       |
|Cifar-S    | Cifar10[2]    |  Resnet18   |   1024      |   10        |  50k        |  10k        |  ✅       |
|Cifar-L    | Cifar10[2]    |  Resnet50   |   2048      |   10        |  50k        |  10k        |  ✅       |
|Imagenet-S | Imagenet[3]   |  Resnet18   |   1024      |   1000      |  1M         |  50k        |  ✅       |
|Imagenet-L | Imagenet[3]   |  Resnet50   |   2048      |   1000      |  1M         |  50k        |  ✅       |

Original dataset
[1] [DogFish](https://worksheets.codalab.org/bundles/0x550cd344825049bdbb865b887381823c) <br>
[2] [Cifar10](https://www.cs.toronto.edu/~kriz/cifar.html)<br>
[3] [Imagenet](https://image-net.org/challenges/LSVRC/2010/2010-downloads.php) (Labels are in the Development Kit)

Already prepared
[2] [Cifar10](https://polybox.ethz.ch/index.php/s/flCES6dSsSL7LKD)

## Example
1. Run dataset/cifar10_generation.py
