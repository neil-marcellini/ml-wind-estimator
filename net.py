# Neil Marcellini
# COMP 499
# 1-30-21

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gpxpy
import gpxpy.gpx


def import_data():
    points = []
    gpx_file = open('Crissy-8-13.gpx', 'r')
    gpx = gpxpy.parse(gpx_file)
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                points.append([point.latitude, point.longitude])
    return points

def fake_wind_data(points):
    # oscillate 5 degrees every 10 mins
    # 10 mins = 600 points
    counter = 1
    alternator = 1
    wind_dir = 255
    wind_estimates = []
    for point in points:
        if counter == 600:
            counter = 1
            shift = 5 * alternator
            wind_dir += shift
            alternator *= -1
        wind_estimates.append(wind_dir)
        counter += 1
    return wind_estimates
        

points = import_data()
wind_estimates = fake_wind_data(points)
print(wind_estimates)
