import torchvision
from torch.optim import AdamW, lr_scheduler
import argparse
from utils import *


def run(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device: ', device)
    model_path = r'model_state_dict.pth'
    data_path = r'/CIFAR10'

    load_model = args.load_model
    model_type = args.model_type
    CONV = args.conv
    batch_size = 8
    learning_rate = 3e-4
    num_epochs = 3

    dataset_train = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True)
    dataset_val = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True)
    dataset_train = CIFAR10(dataset_train)
    dataset_val = CIFAR10(dataset_val)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
    num_train, num_val = len(dataset_train), len(dataset_val)
    print('number of images in training set: ', num_train, '; number of images in validation set: ', num_val)

    model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
    for p in model.parameters():
        p.requires_grad = False

    if model_type == 'finetuned':
        # Using the fine-tuned model to extract features
        model.fc = nn.Linear(2048, 10, bias=True)
        model = model.to(device)
        torch.set_float32_matmul_precision('high')
        if load_model:
            model.load_state_dict(torch.load(model_path))
            for p in model.parameters():
                p.requires_grad = False
            print('Model is loaded successfully.')
        else:
            for p in model.parameters():
                p.requires_grad = True
            optimizer = AdamW(model.parameters(), lr=learning_rate)
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
            trainer = Trainer(model, dataloader_train, dataloader_val, optimizer, scheduler, device)
            trainer.train(num_epochs)
            torch.save(model.state_dict(), model_path)
            trainer.draw()
    # Using the pretrained model to extract features
    all_features, all_targets = extract_features(model, device, dataloader_train, dataloader_val, CONV)
    tsne_features = visualize_tsne(all_features, all_targets)
    # intra_class_variances = calculate_intra_class_variance(all_features, all_targets)
    # inter_class_variances = calculate_inter_class_variance(all_features, all_targets)
    intra_class_variances = calculate_intra_class_variance(tsne_features, all_targets)
    inter_class_variances = calculate_inter_class_variance(tsne_features, all_targets)
    print("Intra-class Variances:", intra_class_variances)
    print("Mean Intra-class Variance:", np.mean(intra_class_variances))
    print("Inter-class Variances:", inter_class_variances)
    print("Mean Inter-class Variance:", np.mean(inter_class_variances))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Model Argument Parser")
    parser.add_argument(
        'model_type',
        type=str,
        choices=['pretrained', 'finetuned'],
        help="Type of the model. Valid inputs are 'pretrained' and 'finetuned'."
    )
    parser.add_argument(
        '--load_model',
        action='store_true',
        help="Flag to load the model. Default is False."
    )
    parser.add_argument(
        '--conv',
        action='store_true',
        help="Flag to use convolutional layer to extract features. Default is False."
    )
    args = parser.parse_args()
    run(args)
