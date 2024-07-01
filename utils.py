import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from sklearn.manifold import TSNE
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torcheval.metrics import MulticlassAccuracy


class CIFAR10(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def __getitem__(self, index):
        image = self.dataset[index][0]
        image = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
        ])(image)
        target = self.dataset[index][1]
        return {'image': image, 'target': target}

    def __len__(self):
        return len(self.dataset)


class FeatureExtrator(nn.Module):
    def __init__(self, model, CONV=False):
        super().__init__()
        self.CONV = CONV
        if CONV:
            self.extractor = nn.Sequential(*list(model.children())[:-3])
        else:
            self.extractor = nn.Sequential(*list(model.children())[:-1])

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        features = self.extractor(x)
        if self.CONV:
            features_sliced = features[:, :10, :, :].clone()
            del features
            features_sliced = features_sliced.flatten(1, -1)
            return features_sliced
        features = features.flatten(1, -1)
        return features


def extract_features(model, device, dataloader_train, dataloader_val, CONV):
    extractor = FeatureExtrator(model, CONV).to(device)
    extractor.eval()
    all_features = []
    all_targets = []
    with torch.no_grad():
        for dataloader in [dataloader_train, dataloader_val]:
            for batch in tqdm(dataloader, desc='Extract features'):
                images = batch['image'].to(device)
                targets = batch['target'].to(device)
                feature = extractor(images)
                all_features.append(feature)
                all_targets.append(targets)

    all_features = torch.concatenate(all_features)
    all_targets = torch.concatenate(all_targets)
    print(all_features.shape, all_targets.shape)
    all_features = all_features.cpu().numpy()
    all_targets = all_targets.cpu().numpy()
    return all_features, all_targets


def visualize_tsne(all_features, all_targets):
    print('Implementing TSNE...')
    tsne = TSNE(n_components=2, perplexity=50)
    tsne_features = tsne.fit_transform(all_features)
    print('TSNE is done.')

    plt.figure(figsize=(5, 4))
    scatter = plt.scatter(tsne_features[:, 0], tsne_features[:, 1], c=all_targets, cmap='tab10')
    plt.legend(handles=scatter.legend_elements()[0], labels=list(range(10)), title="Classes")
    plt.title("t-SNE Visualization of CIFAR-10 Features")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.show()

    return tsne_features


def calculate_intra_class_variance(features, labels):
    unique_labels = range(10)
    class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    intra_class_variances = []
    for label in unique_labels:
        class_features = features[labels == label]
        class_mean = np.mean(class_features, axis=0)
        variance = np.mean(np.linalg.norm(class_features - class_mean, axis=1) ** 2)
        intra_class_variances.append(variance)
    plt.figure(figsize=(10, 6))
    plt.bar(class_labels, intra_class_variances, color='skyblue')
    plt.title('Intra-Class Variance')
    plt.xlabel('Class')
    plt.ylabel('Variance')
    plt.xticks(rotation=45)
    plt.show()
    return intra_class_variances


def calculate_inter_class_variance(features, labels):
    unique_labels = range(10)
    class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    class_means = []
    for label in unique_labels:
        class_features = features[labels == label]
        class_mean = np.mean(class_features, axis=0)
        class_means.append(class_mean)
    class_means = np.array(class_means)
    inter_class_variances = []

    for i in range(len(class_means)):
        for j in range(i + 1, len(class_means)):
            variance = np.linalg.norm(class_means[i] - class_means[j]) ** 2
            inter_class_variances.append(variance)

    inter_class_matrix = np.zeros((10, 10))
    k = 0
    for i in range(10):
        for j in range(i + 1, 10):
            inter_class_matrix[i, j] = inter_class_variances[k]
            inter_class_matrix[j, i] = inter_class_variances[k]
            k += 1

    plt.figure(figsize=(10, 8))
    sns.heatmap(inter_class_matrix, annot=True, fmt=".2f", xticklabels=class_labels, yticklabels=class_labels,
                cmap="YlGnBu")
    plt.title('Inter-Class Variance Heatmap')
    plt.xlabel('Class')
    plt.ylabel('Class')
    plt.show()

    return inter_class_variances


class Trainer:
    def __init__(self, model: nn.Module, train_data: DataLoader, val_data: DataLoader, optimizer: torch.optim.Optimizer,
                 scheduler, device: str):
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.loss_train_log = []
        self.loss_val_log = []
        self.accuracy_train_log = []
        self.accuracy_val_log = []
        self.metric_train = MulticlassAccuracy()
        self.metric_val = MulticlassAccuracy()
        self.log_count = 0

    def _run_batch(self, image, target, train=True):
        logits = self.model(image)
        loss = F.cross_entropy(logits, target)
        if train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        probs = torch.softmax(logits, dim=-1)
        preds = torch.max(probs, dim=-1)[1]

        if train:
            self.metric_train.update(preds, target)
            accuracy = self.metric_train.compute()
            self.loss_train_log.append(loss.item())
        else:
            self.metric_val.update(preds, target)
            accuracy = self.metric_val.compute()
            self.loss_val_log.append(loss.item())

        if self.log_count % 250 == 0:
            print(f'loss: {loss.item():4f}, accuracy: {accuracy.item():4f}')
        self.log_count += 1

    def _run_epoch(self):
        self.model.train()
        self.metric_train.reset()
        self.log_count = 0
        for batch in tqdm(self.train_data, desc='Training model'):
            image, target = batch['image'].to(self.device), batch['target'].to(self.device)
            self._run_batch(image, target, train=True)
        self.scheduler.step()
        self.accuracy_train_log.append(self.metric_train.compute().item())

        self.model.eval()
        self.metric_val.reset()
        self.log_count = 0
        with torch.no_grad():
            for batch in tqdm(self.val_data, desc='Evaluating model'):
                image, target = batch['image'].to(self.device), batch['target'].to(self.device)
                self._run_batch(image, target, train=False)
        self.accuracy_val_log.append(self.metric_val.compute().item())

    def train(self, num_epochs: int):
        for epoch in tqdm(range(num_epochs)):
            self._run_epoch()

    def draw(self):
        figs, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(range(len(self.loss_train_log)), self.loss_train_log, label='training')
        axes[0].plot(range(len(self.loss_val_log)), self.loss_val_log, label='validation')
        axes[0].legend()
        axes[0].grid(True)
        axes[0].set_xlabel('batches')
        axes[0].set_ylabel('loss')
        axes[1].plot(range(len(self.accuracy_train_log)), self.accuracy_train_log, label='training')
        axes[1].plot(range(len(self.accuracy_val_log)), self.accuracy_val_log, label='validation')
        axes[1].legend()
        axes[1].grid(True)
        axes[1].set_xlabel('batches')
        axes[1].set_ylabel('accuracy')
        plt.show()
