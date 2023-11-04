import torch

class Lambda(torch.nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)
    

class demoModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv1d(1, 20, kernel_size=2, stride=1, padding=1),
            torch.nn.ReLU(),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv1d(20, 20, kernel_size=2, stride=1, padding=1),
            torch.nn.BatchNorm1d(20),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=3, stride=1),
            torch.nn.Conv1d(20, 20, kernel_size=2, stride=1, padding=1),
            torch.nn.BatchNorm1d(20),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=3, stride=1),
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv1d(20, 30, kernel_size=2, stride=1, padding=1),
            torch.nn.BatchNorm1d(30),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=3, stride=1),
            torch.nn.Conv1d(30, 20, kernel_size=2, stride=1, padding=1),
            torch.nn.BatchNorm1d(20),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=3, stride=1),
        )
        self.dense = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool1d(200),
            Lambda(lambda x: x.view(x.size(0), -1)),
            torch.nn.Linear(20 * 200, 18),
            torch.nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv3(x)
        x = self.dense(x)
        return x
