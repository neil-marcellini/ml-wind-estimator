# Neil Marcellini
# COMP 499
# 1-30-21

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gpxpy
import gpxpy.gpx
import torch.optim as optim

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class WindDataset(Dataset):
    def __init__(self):
        points = self.import_data()
        samples = self.segment_data(points)
        wind_estimates = self.fake_wind_data(samples)
        self.samples = samples
        self.wind_estimates = wind_estimates

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample, wind_estimate = self.samples[idx], self.wind_estimates[idx]
        return sample, wind_estimate

    def import_data(self):
        points = []
        gpx_file = open('Crissy-8-13.gpx', 'r')
        gpx = gpxpy.parse(gpx_file)
        for track in gpx.tracks:
            for segment in track.segments:
                for point in segment.points:
                    points.append([point.latitude, point.longitude])
        return points

    def fake_wind_data(self, points):
        # oscillate 5 degrees every size seconds
        size = 60
        counter = 1
        alternator = 1
        wind_dir = 255
        wind_estimates = []
        for point in points:
            if counter == size:
                counter = 1
                shift = 5 * alternator
                wind_dir += shift
                alternator *= -1
            wind_estimates.append(wind_dir)
            counter += 1
        return wind_estimates

    def segment_data(self, data_set):
        size = 60
        segments = []
        for i in range(size, len(data_set) - 1):
            sample = np.array(data_set[i-size: i])
            sample = sample.flatten()
            segments.append(sample)
        return segments



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(120, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, 256)
        self.output = nn.Linear(256, 360)

    # x represents our data
    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        x = F.relu(x)
        x = self.output(x)
        output = F.log_softmax(x, dim=1)
        return output


def train():
    dataset = WindDataset()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    net = Net()
    net = net.float()
    learning_rate = 0.001
    epochs = 5
    optimizer = optim.SGD(net.parameters(), lr=0.01)
    for epoch in range(epochs):
        for i, data in enumerate(dataloader):
            sample, wind_estimate = data
            sample = sample.float()
            wind_estimate = wind_estimate.long()
            net.zero_grad()
            # prediction = forward pass
            prediction = net(sample)
            # loss
            l = F.nll_loss(prediction, wind_estimate)
            # gradients, backward pass
            l.backward()
            # update weights
            optimizer.step()
        print(l)

train()
