import torch
import torch.optim as optim
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler

from cnn import CNN
from dataset import load_mnist


def train(train_loader, num_epochs, optimizer, scheduler, device):
    total_step = len(train_loader)
    model.train()
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # forward prop
            outputs = model(images)
            loss = criterion(outputs, labels)

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update learning rate
            scheduler.step()

            # print results every 100 steps
            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Learning Rate: {:.10f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(), scheduler.get_last_lr()[0]))


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


if __name__ == "__main__":
    data_path = './data'

    # set random seed to make the result repeatable
    torch.manual_seed(0)

    # hyper parameters
    batch_size = 128
    num_epochs = 13
    learning_rate0 = 0.001

    # check if CUDA is available and set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # load train set
    train_images, train_labels = load_mnist(data_path, kind='train')
    train_images = train_images.reshape(-1, 1, 28, 28)  # reshape the size to match the input of CNN

    # load test set
    test_images, test_labels = load_mnist(data_path, kind='t10k')
    test_images = test_images.reshape(-1, 1, 28, 28)  # reshape the size to match the input of CNN

    # numpy to tensor
    train_images = torch.from_numpy(train_images.copy()).float()
    train_labels = torch.from_numpy(train_labels.copy()).long()
    test_images = torch.from_numpy(test_images.copy()).float()
    test_labels = torch.from_numpy(test_labels.copy()).long()

    # create dataset objects
    train_dataset = torch.utils.data.TensorDataset(train_images, train_labels)
    test_dataset = torch.utils.data.TensorDataset(test_images, test_labels)

    # create dataset loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # initialize model and loss fn
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate0)

    # learning rate scheduler
    scheduler = lr_scheduler.StepLR(optimizer, step_size=469 * 2, gamma=0.1)

    # train
    print("\nTraining: ")
    train(
        train_loader=train_loader,
        num_epochs=num_epochs,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler
    )
    torch.save(model.state_dict(), 'model.pth')

    # test
    # test(
    #     test_loader=test_loader,
    #     model=model,
    #     device=device
    # )
