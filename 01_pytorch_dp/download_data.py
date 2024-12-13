import torchvision
import warnings
warnings.simplefilter("ignore")

_ = torchvision.datasets.MNIST('data',
                               train=True,
                               transform=None,
                               target_transform=None,
                               download=True)
