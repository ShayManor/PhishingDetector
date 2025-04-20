import torch
from torch import nn
from torch.utils.data import DataLoader

from train import PhishingDataset, NeuralNet


def check_correct(pred, exp):
    return int(pred[0] * 2) == int(exp[0] * 2)


if __name__ == '__main__':
    dataset = PhishingDataset('test.csv')
    dataloader = DataLoader(batch_size=5, dataset=dataset, shuffle=True)
    model = NeuralNet()
    state = torch.load('weights.pt')
    model.load_state_dict(state)
    model.eval()
    criterion = nn.MSELoss()
    correct_counter, total = 0, 0
    test_loss = 0

    for data, label in dataloader:
        output = model(data)
        pred = output.data
        correct = check_correct(pred[0].tolist(), label.tolist())
        if not correct:
            print(f'Incorrect case. Expected: {label[0]}, Received: {pred[0]}')
        correct_counter += int(correct)
        total += label.size(0)
        loss = criterion(output, label)
        test_loss += loss.item() * data.size(0)
    print(f'Testing Loss:{test_loss / len(dataloader)}')
    print(f'Correct Predictions: {correct_counter}/{total}')
