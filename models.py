import torch.nn as nn

class EncoderNet(nn.Module):
    """
    Feature extractor for MNIST-like data
    """

    def __init__(self, num_channels=3, kernel_size=5):
        super(EncoderNet, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=kernel_size)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 64, kernel_size=kernel_size)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, 64 * 2, kernel_size=kernel_size)
        self.bn3 = nn.BatchNorm2d(64 * 2)
        self.sigmoid = nn.Sigmoid()
        self._out_features = 128

    def forward(self, input):
        x = self.bn1(self.conv1(input))
        x = self.relu1(self.pool1(x))
        x = self.bn2(self.conv2(x))
        x = self.relu2(self.pool2(x))
        x = self.sigmoid(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        return x

    def output_size(self):
        return self._out_features

class ClassifierNet(nn.Module):
    def __init__(self, input_size=128, n_class=10):
        super(ClassifierNet, self).__init__()
        self._n_classes = n_class
        self.fc1 = nn.Linear(input_size, 100)
        self.bn1 = nn.BatchNorm1d(100)
        self.relu1 = nn.ReLU()
        self.dp1 = nn.Dropout2d()
        self.fc2 = nn.Linear(100, 100)
        self.bn2 = nn.BatchNorm1d(100)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(100, n_class)

    def n_classes(self):
        return self._n_classes

    def forward(self, input):
        x = self.dp1(self.relu1(self.bn1(self.fc1(input))))
        x = self.relu2(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x