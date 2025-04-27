import pkbar
import torch
torch.cuda.empty_cache()
import torch.nn as nn
from sklearn.metrics import accuracy_score


class TrainModel:
    def __init__(self, model, epochs, optimizer, criterion, device, lossfn, train_loader, test_loader, num_class):
        self.lossfn = None
        self.model = model
        self.epochs = epochs
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.num_class = num_class

    def loss_function(self):
        if self.num_class == 2:
            self.lossfn = nn.BCEWithLogitsLoss()
        else:
            self.lossfn = nn.CrossEntropyLoss()

    def get_pred_labels(self, preds):
        if self.num_class == 2:
            return (torch.sigmoid(preds) > 0.5).float()
        else:
            return torch.argmax(preds, dim=1)

    def train_model(self):
        losses = torch.zeros(self.epochs)
        trainAcc = []
        testAcc = []

        for epoch in range(self.epochs):
            kbar = pkbar.Kbar(target=len(self.train_loader), epoch=epoch, num_epochs=self.epochs)
            epoch_preds, epoch_labels = [], []
            self.model.train()

            for (i, (x_train, y_train)) in enumerate(self.train_loader):
                x_train = x_train.to(self.device)
                y_train = y_train.float().unsqueeze(1).to(self.device)

                yHat = self.model(x_train)
                loss = self.lossfn(yHat, y_train)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                pred_labels = self.get_pred_labels(yHat)
                epoch_preds.append(pred_labels.cpu())
                epoch_labels.append(y_train.cpu())
                kbar.update(i, values=[("loss", loss.item())])  # update per batch with loss

            # Compute train accuracy
            pred_acc = torch.cat(epoch_preds)
            true_acc = torch.cat(epoch_labels)
            train_acc = 100 * accuracy_score(true_acc, pred_acc)
            trainAcc.append(train_acc)

            # Evaluation on test set
            self.model.eval()
            with torch.no_grad():
                X_test, y_test = next(iter(self.test_loader))
                X_test = X_test.to(self.device)
                y_test = y_test.float().unsqueeze(1).to(self.device)
                y_test_pred = self.model(X_test)
                test_labels = self.get_pred_labels(y_test_pred)
                test_acc = 100 * accuracy_score(y_test.cpu().numpy(), test_labels.cpu().numpy())
                testAcc.append(test_acc)

            # Show train and test accuracy at the end of the epoch
            kbar.add(1, values=[("train_acc", train_acc), ("test_acc", test_acc)])

            # if (epoch + 1) % 10 == 0:
            #     print(f'\nEpoch [{epoch+1}/{epochs}], Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')
            #     print('_' * 60)

        return trainAcc, testAcc
