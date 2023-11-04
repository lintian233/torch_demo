import time
import os
import itertools

import torch
import numpy as np
import matplotlib.pyplot as plt

from torch_lr_finder import LRFinder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# from tqdm.autonotebook import tqdm

class learn:
    def __init__(self, model, train_dl, vaild_dl, catgories=None):
        self.model, self.opt, self.loss_func = model
        self.train_dl = train_dl
        self.vaild_dl = vaild_dl
        self.catgories = catgories


    def __loss_batch(self, model, loss_func, xb, yb, opt=None):
        loss = loss_func(model(xb), yb)
        if opt is not None:
            loss.backward()
            opt.step()
            opt.zero_grad()
        return loss.item(), len(xb)
    
    def predict(self, test_dl):
        self.model.eval()
        with torch.no_grad():
            true_labels = []
            predict_labels = []
            for xb, yb in test_dl:
                true_labels.append(yb)
                predict_labels.append(self.model(xb))

            true_labels = torch.cat(true_labels, dim=0).cpu().numpy()
            predict_labels = torch.cat(predict_labels, dim=0).cpu().numpy()
            true_labels = np.argmax(true_labels, axis=1)
            predict_labels = np.argmax(predict_labels, axis=1)

        return true_labels, predict_labels
    
    def fit(self, epochs):
        for epoch in range(epochs):
            start_time = time.time()

            self.model.train()
            for xb, yb in self.train_dl:
                self.__loss_batch(self.model, self.loss_func, xb, yb, self.opt)

            self.model.eval()
            with torch.no_grad():
                losses, nums = zip(
                    *[self.__loss_batch(self.model, self.loss_func, xb, yb) for xb, yb in self.vaild_dl]
                )

            val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

            time_format = "%H:%M:%S"
            print(
                f"Epoch: {epoch+1:02} | Val Loss: {val_loss:.3f} | Time: {time.strftime(time_format, time.gmtime(time.time() - start_time))}"
            )

    def lr_find(self):
        model = self.model
        optimizer = self.opt
        criterion = self.loss_func
        trainloader = self.train_dl

        lr_finder = LRFinder(model, optimizer, criterion)
        lr_finder.range_test(trainloader, end_lr=100, num_iter=100)
        lr_finder.plot()
        lr_finder.reset()

    def show_confusion_matrix(self):
        true_labels, predict_labels = self.predict(self.vaild_dl)
        catgories = self.catgories

        cm = confusion_matrix(true_labels, predict_labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=catgories)
        disp.plot()
        disp.figure_.savefig("./confusion_matrix.png")