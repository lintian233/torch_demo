import time

import torch
import numpy as np

from torch_lr_finder import LRFinder
# from tqdm.autonotebook import tqdm

class learn:
    def __init__(self, model, train_dl, vaild_dl):
        self.model, self.opt, self.loss_func = model
        self.train_dl = train_dl
        self.vaild_dl = vaild_dl

    def __loss_batch(self, model, loss_func, xb, yb, opt=None):
        loss = loss_func(model(xb), yb)
        if opt is not None:
            loss.backward()
            opt.step()
            opt.zero_grad()
        return loss.item(), len(xb)

    def fit(self, epochs):
        for epoch in range(epochs):
            start_time = time.time()

            self.model.train()
            for xb, yb in self.train_dl:
                self.__loss_batch(self.model, self.loss_func, xb, yb, self.opt)

            self.model.eval()
            with torch.no_grad():
                losses, nums = zip(
                    *[
                        self.__loss_batch(self.model, self.loss_func, xb, yb)
                        for xb, yb in self.vaild_dl
                    ]
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

        

    
