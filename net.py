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


print(import_data())
