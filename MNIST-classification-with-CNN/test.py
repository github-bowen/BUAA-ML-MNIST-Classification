import os

import torch

from cnn import CNN
from dataset import load_mnist


def test(model, test_loader, device):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('\nAccuracy on test set: {:.2f} %'.format(100 * correct / total))


def lubohong_test(img_path, model_path, device):
    # load model
    model = CNN()
    model.load_state_dict(torch.load(model_path + 'model.pth', map_location=torch.device(device)))

    # load test set
    test_images, test_labels = load_mnist(img_path, kind='t10k')
    test_images = test_images.reshape(-1, 1, 28, 28)  # reshape the size to match the input of CNN

    # numpy to tensor
    test_images = torch.from_numpy(test_images.copy()).float()
    test_labels = torch.from_numpy(test_labels.copy()).long()

    # create dataset objects
    test_dataset = torch.utils.data.TensorDataset(test_images, test_labels)

    # create dataset loaders
    batch_size = 128
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("Model details:")
    print(model)
    test(model, test_loader, device)


if __name__ == "__main__":
    # check if CUDA is available and set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}\n')

    lubohong_test(
        img_path="./data",
        model_path="./",
        device=device
    )