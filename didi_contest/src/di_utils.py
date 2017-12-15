# -*- coding:utf-8 -*-
'''
xxxxxxxxx

'''
import random
from matplotlib import pyplot as plt
import numpy as np
import os,sys
import math

X_OFS = 1000
Y_OFS = 1000

def track_stats(di_track_file):
	f_track = open(di_track_file, 'r')
	lines = f_track.readlines(60000)
	min_x = sys.float_info.max
	max_x = 0
	min_y = sys.float_info.max
	max_y = 0
	for line in lines[1:]:
		line = line.split(',')
		vehicle_id = line[0]
		timestamp = line[1]
		x_coordinate = line[2]
		y_coordinate = line[3]
		speed = line[4]
		category = line[5]

		x_coord = float(x_coordinate)
		y_coord = float(y_coordinate)

		if x_coord < min_x:
			min_x = x_coord
		elif x_coord > max_x:
			max_x = x_coord

		if y_coord < min_y:
			min_y = y_coord
		elif y_coord > max_y:
			max_y = y_coord

	print(min_x)
	print(max_x)
	print(min_y)
	print(max_y)

coords = []

def onclick(event):
    global ix, iy
    ix, iy = event.xdata, event.ydata
    print 'x = %d, y = %d'%(ix, iy)
    global coords
    coords.append((ix, iy))

    if len(coords) == 2:
        fig.canvas.mpl_disconnect(cid)

    return coords

def draw_veihcle_track(di_track_file, plot_all=True):
	f_track = open(di_track_file, 'r')
	line_coords = []
	vehicle_tracks = {}
	lines = f_track.readlines()
	for line in lines[1:]:
		line = line.split(',')
		vehicle_id = line[0]
		timestamp = line[1]
		x_coordinate = line[2]
		y_coordinate = line[3]
		speed = line[4]
		category = line[5]
		if vehicle_id in vehicle_tracks.keys():
			last_timestamp, _, _ = vehicle_tracks[vehicle_id][-1][-1]
			if float(timestamp) - float(last_timestamp) > 4:
				vehicle_tracks[vehicle_id].append([])
				vehicle_tracks[vehicle_id][-1].append([float(timestamp), float(x_coordinate), float(y_coordinate)])
			else:
				vehicle_tracks[vehicle_id][-1].append([float(timestamp), float(x_coordinate), float(y_coordinate)])
		else:
			vehicle_tracks[vehicle_id] = []
			vehicle_tracks[vehicle_id].append([])
			vehicle_tracks[vehicle_id][-1].append([float(timestamp), float(x_coordinate), float(y_coordinate)])

	if plot_all:
		all_coords = np.array([[1493859164, 521696.473915, 55061.506951]])
		for vehicle_id in vehicle_tracks:
			for vehicle_track in vehicle_tracks[vehicle_id]:
				per_vehicle_track = np.array(vehicle_track)
				all_coords = np.concatenate((all_coords, per_vehicle_track), axis=0)
		timestamp, x, y = all_coords.T
		plt.xlim(521018.721067-X_OFS, 521726.523312+X_OFS)
		plt.ylim(53888.244714-Y_OFS, 58147.347356+Y_OFS)
		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.scatter(x, y)
		cid = fig.canvas.mpl_connect('button_press_event', onclick)
		#plt.scatter(x, y)
		plt.show()

	else:
		for vehicle_id in vehicle_tracks:
			print(vehicle_id)
			for vehicle_track in vehicle_tracks[vehicle_id]:
				per_vehicle_track = np.array(vehicle_track)
				timestamp, x, y = per_vehicle_track.T
				plt.xlim(521018.721067, 521726.523312)
				plt.ylim(53888.244714, 58147.347356)
				plt.scatter(x, y)
				plt.show()

def calc_distance():
	node_coords = [521780, 55117]
	vehicle_coords_depart = [521696.473915, 55061.506951]
	vehicle_coords_arrive = [521680.892788, 55051.685121]

	print(math.sqrt((node_coords[0] - vehicle_coords_depart[0])**2 + (node_coords[1] - vehicle_coords_depart[1])**2))
	print(math.sqrt((node_coords[0] - vehicle_coords_arrive[0])**2 + (node_coords[1] - vehicle_coords_arrive[1])**2))

if __name__ == '__main__':
    random.seed(200)

    calc_distance()
    #track_stats(di_track_file='/home/nlp/bigsur/data/diditech/vehicle_track.txt')
    #draw_veihcle_track(di_track_file='/home/nlp/bigsur/data/diditech/vehicle_track.txt')
