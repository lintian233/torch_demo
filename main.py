import torch

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.optim import SGD
from torch.nn.functional import cross_entropy

from model.model_demo import demoModel
from framework.learn import learn
from utils.load_data import LightcurveDataset, WrappedDataLoader

dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def get_data(train_set, test_set, batch_size):
    train_batch_size = batch_size
    test_batch_size = batch_size * 2
    return (
        DataLoader(train_set, train_batch_size, shuffle=True),
        DataLoader(test_set, test_batch_size, shuffle=True),
    )

def preprocess(x, y):
    return x.view(-1, 1, 401).to(dev), y.to(dev)


def load_model(model, lr):
    model = model().to(dev)
    optimizer = SGD(model.parameters(), lr=lr)
    loss_func = cross_entropy
    return model, optimizer, loss_func


def simple_demo():
    DATSET_PATH = "./data/OCVS"
    TRAIN_SIZE = 0.8
    BATCH_SIZE = 30
    EPOCHS = 30
    RANDOM_STATE = 42
    LR = 0.001

    # LOAD DATA
    dataset = LightcurveDataset(DATSET_PATH, transform=True, target_transform=True)
    catgories = dataset.get_catgories()
    train_set, vaild_set = train_test_split(
        dataset, train_size=TRAIN_SIZE, random_state=RANDOM_STATE
    )
    train_dl, vaild_dl = get_data(train_set, vaild_set, batch_size=BATCH_SIZE)
    train_dl = WrappedDataLoader(train_dl, preprocess)
    vaild_dl = WrappedDataLoader(vaild_dl, preprocess)

    # LOAD MODEL
    model = load_model(demoModel, LR)

    # TRAIN MODEL
    learner = learn(model, train_dl, vaild_dl, catgories)
    # learner.lr_find()
    learner.fit(EPOCHS)
    learner.show_confusion_matrix()


if __name__ == "__main__":
    simple_demo()
