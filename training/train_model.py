import pkbar
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from typing import Tuple, List, Optional, Any


class TrainModel:
    def __init__(
        self,
        model,
        epochs,
        optimizer,
        device,
        train_loader,
        test_loader,
        num_classes,
        scheduler,
        early_stopping_patience,
    ):

        self.model = model.to(device)
        self.epochs = epochs
        self.optimizer = optimizer
        self.criterion = self._get_default_loss_fn(num_classes)
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.num_classes = num_classes
        self.scheduler = scheduler
        self.early_stopping_patience = early_stopping_patience
        self.best_test_acc = 0.0
        self.epochs_without_improvement = 0

        
    def _get_default_loss_fn(self, num_classes: int) -> nn.Module:
        """Get default loss function based on number of classes."""
        return nn.BCEWithLogitsLoss() if num_classes == 2 else nn.CrossEntropyLoss()
    
    def _get_pred_labels(self, preds: torch.Tensor) -> torch.Tensor:
        """Convert model outputs to predicted labels."""
        if self.num_classes == 2:
            return (torch.sigmoid(preds) > 0.5).float()
        return torch.argmax(preds, dim = 1)

    def prepare_labels(self, y, num_classes, device):
        """Properly format labels based on classification type"""
        y = y.to(device)
        if num_classes == 2:  # Binary classification
            return y.float().unsqueeze(1)  # Shape [batch_size, 1]
        else:  # Multi-class classification
            return y.long()  # Shape [batch_size]
    
    # def _evaluate(self, data_loader: torch.utils.data.DataLoader) -> Tuple[float, float]:
    #     """Evaluate model on given data loader."""
    #     self.model.eval()
    #     torch.cuda.empty_cache()
    #     all_preds, all_labels = [], []
    #     total_loss = 0.0
        
    #     with torch.inference_mode():
    #         for x, y in data_loader:
    #             x = x.to(self.device)
    #             y = self.prepare_labels(y, num_classes=self.num_classes, device=self.device)
    #             outputs = self.model(x)
    #             loss = self.criterion(outputs, y)
    #             total_loss += loss.item()
                
    #             preds = self._get_pred_labels(outputs)
    #             all_preds.append(preds.cpu())
    #             all_labels.append(y.cpu())
        
    #     avg_loss = total_loss / len(data_loader)
    #     all_preds = torch.cat(all_preds)
    #     all_labels = torch.cat(all_labels)
    #     accuracy = 100 * accuracy_score(all_labels, all_preds)
    #     f1 = f1_score(all_labels, all_preds, average='macro' if self.num_classes > 2 else 'binary')
        
    #     return avg_loss, accuracy, f1
    def _evaluate(self, data_loader: torch.utils.data.DataLoader) -> Tuple[float, float]:
        """Evaluate model on given data loader with memory optimization."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        # Use inference_mode (stronger than no_grad) and autocast for FP16
        with torch.inference_mode():
            for x, y in data_loader:
                # Memory-efficient batch processing
                x = x.to(self.device, non_blocking=True)
                y = self.prepare_labels(y, self.num_classes, self.device)
                
                # Forward pass with manual memory management
                outputs = self.model(x)
                loss = self.criterion(outputs, y)
                total_loss += loss.item()
                
                # Process predictions on CPU immediately
                preds = self._get_pred_labels(outputs).cpu()
                all_preds.append(preds)
                all_labels.append(y.cpu())
                
                # Explicit cleanup
                del x, y, outputs, loss
                torch.cuda.empty_cache()  # Optional: clears cache between batches
                
        # Calculate metrics
        avg_loss = total_loss / len(data_loader)
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        accuracy = 100 * accuracy_score(all_labels.numpy(), all_preds.numpy())
        
        return avg_loss, accuracy
    
    def _check_early_stopping(self, test_acc: float) -> bool:
        """Check if early stopping criteria are met."""
        if self.early_stopping_patience is None:
            return False
            
        if test_acc > self.best_test_acc:
            self.best_test_acc = test_acc
            self.epochs_without_improvement = 0
            # Save best model
            torch.save(self.model.state_dict(), 'best_model.pth')
        else:
            self.epochs_without_improvement += 1
            
        return self.epochs_without_improvement >= self.early_stopping_patience
    
    def train_model(self) -> tuple[list[float], list[int | float], list[Any], list[Any], list[Any], list[Any]]:
        train_losses, train_accuracies, train_f1s = [], [], []
        test_losses, test_accuracies, test_f1s = [], [], []
        

        print('\nLoss Function:', self.criterion)
        
        for epoch in range(self.epochs):
            # Initialize progress bar
            kbar = pkbar.Kbar(target = len(self.train_loader), epoch = epoch, num_epochs = self.epochs, width = 20)
            
            # Training phase
            self.model.train()
            epoch_train_loss = 0.0
            all_train_preds, all_train_labels = [], []
            
            for i, (x_train, y_train) in enumerate(self.train_loader):
                x_train = x_train.to(self.device)
                y_train = self.prepare_labels(y_train, num_classes = self.num_classes, device = self.device)
                
                # Forward pass
                outputs = self.model(x_train)
                loss = self.criterion(outputs, y_train)
                epoch_train_loss += loss.item()
                
                # Backward pass and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Collect predictions and labels for metrics
                preds = self._get_pred_labels(outputs)
                all_train_preds.append(preds.cpu())
                all_train_labels.append(y_train.cpu())
                kbar.update(i, values=[("loss", loss.item())])

            
            # Calculate training metrics
            avg_train_loss = epoch_train_loss / len(self.train_loader)
            train_preds = torch.cat(all_train_preds)
            train_labels = torch.cat(all_train_labels)
            train_acc = 100 * accuracy_score(train_labels, train_preds)
            train_f1 = f1_score(train_labels, train_preds, average = 'macro' if self.num_classes > 2 else 'binary')
            
            train_losses.append(avg_train_loss)
            train_accuracies.append(train_acc)
            train_f1s.append(train_f1)
            
            #-----------------Evaluation phase--------------#
            # test_loss, test_acc, test_f1 = self._evaluate(self.test_loader)
            test_loss, test_acc = self._evaluate(self.test_loader)
            test_losses.append(test_loss)
            test_accuracies.append(test_acc)
            # test_f1s.append(test_f1)
            

            # Update progress bar with metrics
            kbar.add(1, values= [
                ("train_acc", round(train_acc, 2)),
                ("test_acc", round(test_acc, 2)),
            ])
            
            #----------// Lr scheduler //-----------------#
            if self.scheduler is not None:
                self.scheduler.step()
            
            #----------------// Check for early stopping //-------------------#
            if self._check_early_stopping(test_acc):
                print(f"\nEarly stopping at epoch {epoch + 1} as test accuracy didn't improve for {self.early_stopping_patience} epochs.")
                break
        
        return train_losses, train_accuracies, train_f1s, test_losses, test_accuracies, test_f1s