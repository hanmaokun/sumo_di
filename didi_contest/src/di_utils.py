# -*- coding:utf-8 -*-
'''
xxxxxxxxx

'''
import random
from matplotlib import pyplot as plt
import numpy as np
import os,sys
import math
from numpy.linalg import norm
import xml.etree.cElementTree as ET
import copy

X_OFS = 1000
Y_OFS = 1000

MAX_NORMAL_TIMELAP = 10
MAX_ABNORMAL_IN_CONTINUOUSE_ROUTE_ALLOWANCE = 3

TIMESTAMP_BASE = 1493852410.0
TIMESTAMP_MAX  = 1494982799.0

def track_stats(di_track_file):
	f_track = open(di_track_file, 'r')
	lines = f_track.readlines()
	min_x = sys.float_info.max
	max_x = 0
	min_y = sys.float_info.max
	max_y = 0
	min_t = sys.float_info.max
	max_t = 0
	min_sod = sys.float_info.max
	max_spd = 0
	for line in lines[1:]:
		line = line.split(',')
		vehicle_id = line[0]
		timestamp = line[1]
		x_coordinate = line[2]
		y_coordinate = line[3]
		speed = float(line[4])
		category = line[5]

		if float(timestamp) < min_t:
			min_t = float(timestamp)
		elif float(timestamp) > max_t:
			max_t = float(timestamp)

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

		if speed < min_sod:
			min_sod = speed
		elif speed > max_spd:
			max_spd = speed

	print(min_x)
	print(max_x)
	print(min_y)
	print(max_y)
	print(max_t)
	print(min_t)
	print(max_spd)
	print(min_sod)

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

def distance(p0, p1):
    return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)

def is_continuous_track(last_t, last_x, last_y, last_spd, t, x, y, spd):
	timelap = t - last_t

	if timelap < 0:
		return False

	if timelap > MAX_NORMAL_TIMELAP:
		return False

	last_pos = [last_x, last_y]
	cur_pos = [x, y]
	delta_dist = distance(last_pos, cur_pos)

	max_spd = spd if spd > last_spd else last_spd
	exp_dist = max_spd * timelap

	if delta_dist > exp_dist + 10:
		print("real dist: " + str(delta_dist) + ", expect dist: " + str(exp_dist))
		return False

	return True

def draw_veihcle_track(di_track_file, plot_all=False):
	f_track = open(di_track_file, 'r')
	line_coords = []
	vehicle_tracks = {}
	lines = f_track.readlines()
	abnormal_ctr = 0
	for line in lines:
		line = line.split(',')
		vehicle_id = line[0]
		timestamp = line[1]
		x_coordinate = line[2]
		y_coordinate = line[3]
		speed = line[4]
		#category = line[5]
		if vehicle_id in vehicle_tracks.keys():
			last_timestamp, last_x, last_y, last_speed = vehicle_tracks[vehicle_id][-1][-1]
			if is_continuous_track(last_timestamp, last_x, last_y, last_speed, float(timestamp), float(x_coordinate), float(y_coordinate), float(speed)):
				vehicle_tracks[vehicle_id][-1].append([float(timestamp), float(x_coordinate), float(y_coordinate), float(speed)])
			else:
				abnormal_ctr += 1
				if abnormal_ctr == MAX_ABNORMAL_IN_CONTINUOUSE_ROUTE_ALLOWANCE:
					abnormal_ctr = 0
					vehicle_tracks[vehicle_id].append([])
					vehicle_tracks[vehicle_id][-1].append([float(timestamp), float(x_coordinate), float(y_coordinate), float(speed)])
				else:
					vehicle_tracks[vehicle_id][-1].append([last_timestamp, last_x, last_y, last_speed])
		else:
			abnormal_ctr = 0
			vehicle_tracks[vehicle_id] = []
			vehicle_tracks[vehicle_id].append([])
			vehicle_tracks[vehicle_id][-1].append([float(timestamp), float(x_coordinate), float(y_coordinate), float(speed)])

	if plot_all:
		all_coords = np.array([[1493859164, 521696.473915, 55061.506951, 6.6]])
		for vehicle_id in vehicle_tracks:
			for vehicle_track in vehicle_tracks[vehicle_id]:
				per_vehicle_track = np.array(vehicle_track)
				all_coords = np.concatenate((all_coords, per_vehicle_track), axis=0)
		timestamp, x, y, spd = all_coords.T
		plt.xlim(520955.288814-X_OFS, 521920.177507+X_OFS)
		plt.ylim(53380.188236-Y_OFS, 58715.227708+Y_OFS)
		fig = plt.figure()
		ax = fig.add_subplot(111)
		#ax.scatter(x, y)
		ax.plot(x, y)
		cid = fig.canvas.mpl_connect('button_press_event', onclick)
		#plt.scatter(x, y)
		plt.show()

	else:
		plt.xlim(520955.288814, 521920.177507)
		plt.ylim(53380.188236, 58715.227708)
		for vehicle_id in vehicle_tracks:
			print(vehicle_id)
			for vehicle_track in vehicle_tracks[vehicle_id]:
				if len(vehicle_track) < 10:
					continue
				per_vehicle_track = np.array(vehicle_track)
				timestamp, x, y, spd = per_vehicle_track.T
				plt.plot(x, y)
		plt.show()

def calc_distance():
	node_coords = [521411, 54822]
	vehicle_coords_depart = [521400, 53998]
	vehicle_coords_arrive = [521509.502259, 54947.216566]

	print(math.sqrt((node_coords[0] - vehicle_coords_depart[0])**2 + (node_coords[1] - vehicle_coords_depart[1])**2))
	#print(math.sqrt((node_coords[0] - vehicle_coords_arrive[0])**2 + (node_coords[1] - vehicle_coords_arrive[1])**2))

routes_nodirection = ['1#2', '2#3', '3#4', \
					  	'4#5', '5#6', '6#7', \
						'1_0#1', '1_1#1', '1_2#1', \
						'2_0#2', \
						'3_0#3', '3_1#3', \
						'4_0#4', '4_1#4', \
						'5_0#5', '5_1#5', \
						'6_0#6', '6_1#6', \
						'7_0#7', '7_1#7', '7_2#7' \
					]
routes_coords = [[[521677, 58109], [521580,57466]], [[521580, 57466], [521520,57059]], [[521520, 57059], [521452,56668]], \
				 [[521452, 56668], [521433,55855]], [[521433, 55855], [521411,54822]], [[521411, 54822], [521400,53998]], \
				 [[521296, 58295], [521677,58109]], [[521729, 58722], [521677,58109]], [[521769, 58107], [521677,58109]], \
				 [[521542, 57515], [521580,57466]], \
				 [[521387, 57093], [521520,57059]], [[521732, 57032], [521520,57059]], \
				 [[520955, 56693], [521452,56668]], [[521920, 56605], [521452,56668]], \
				 [[521378, 55918], [521433,55855]], [[521660, 55437], [521433,55855]], \
				 [[520987, 54810], [521411,54822]], [[521780, 55117], [521411,54822]], \
				 [[521000, 53960], [521400,53998]], [[521417, 53371], [521400,53998]], [[521856, 54023], [521400,53998]] \
				]

def p_distance(p1, p2, p3):
	x1, y1 = p1
	x2, y2 = p2
	x3, y3 = p3
	px = x2-x1 
	py = y2-y1
	dAB = px*px + py*py
	u = ((x3 - x1) * px + (y3 - y1) * py) / dAB
	x4 = x1 + u * px
	y4 = y1 + u * py

	p4 = (x4, y4)
	return distance(p1, p4)

def list_mean(lst):
	return reduce(lambda x, y: x + y, lst) / len(lst)

def gen_route_coords(net_xml_file):
	SUMO_X_OFS = 520955
	SUMO_Y_OFS = 53371
	net_tree = ET.ElementTree(file=net_xml_file)
	net_tree_root = net_tree.getroot()
	edges = net_tree_root.findall('edge')

	routes_coords_ = []

	for route_nodir in routes_nodirection:
		route_coord_ = []
		routes = route_nodir.split('#')
		edge_a_name = 'edgeL-' + routes[0] + '-' + routes[1]
		edge_b_name = 'edgeL-' + routes[1] + '-' + routes[0]
		start_x = []
		start_y = []
		stop_x = []
		stop_y = []
		for edge in edges:
			cur_edge_id = edge.attrib['id']
			lane = edge.find('lane')
			shape_str = lane.attrib['shape']
			shapes_str = shape_str.split(' ')
			start_coord_str = shapes_str[0].split(',')
			stop_coord_str = shapes_str[1].split(',')
			if (cur_edge_id == edge_a_name):
				start_x.append(float(start_coord_str[0]))
				start_y.append(float(start_coord_str[1]))
				stop_x.append(float(stop_coord_str[0]))
				stop_y.append(float(stop_coord_str[1]))
			if (cur_edge_id == edge_b_name):
				stop_x.append(float(start_coord_str[0]))
				stop_y.append(float(start_coord_str[1]))
				start_x.append(float(stop_coord_str[0]))
				start_y.append(float(stop_coord_str[1]))

		start_x_center = list_mean(start_x) + SUMO_X_OFS
		start_y_center = list_mean(start_y) + SUMO_Y_OFS
		stop_x_center = list_mean(stop_x) + SUMO_X_OFS
		stop_y_center = list_mean(stop_y) + SUMO_Y_OFS
		route_coord_.append([start_x_center, start_y_center])
		route_coord_.append([stop_x_center, stop_y_center])
		routes_coords_.append(route_coord_)

	return routes_coords_

def is_on_route(vehicle_coord, line_end_a, line_end_b, route_width):
	dist = norm(np.cross(line_end_b-line_end_a, line_end_a-vehicle_coord))/norm(line_end_b-line_end_a)
	if dist > route_width*3:
		return False, sys.float_info.max

	# 'https://stackoverflow.com/questions/1811549/perpendicular-on-a-line-from-a-given-point'
	#  :calc the perpendicular cross point to a line from given point.
	# x3, y3 = vehicle_coord
	# x1, y1 = line_end_a
	# x2, y2 = line_end_b
	# k = ((y2-y1) * (x3-x1) - (x2-x1) * (y3-y1)) / (math.sqrt(abs(y2-y1)) + math.sqrt(abs(x2-x1)))
	# x4 = x3 - k * (y2-y1)
	# y4 = y3 + k * (x2-x1)

	# https://stackoverflow.com/questions/10301001/perpendicular-on-a-line-segment-from-a-given-point
	x1, y1 = line_end_a
	x2, y2 = line_end_b
	x3, y3 = vehicle_coord
	px = x2-x1 
	py = y2-y1
	dAB = px*px + py*py
	u = ((x3 - x1) * px + (y3 - y1) * py) / dAB
	x4 = x1 + u * px
	y4 = y1 + u * py

	if (((x4 <= x2) and (x4 >= x1)) or ((x4 >= x2) and (x4 <= x1))) and \
		(((y4 <= y2) and (y4 >= y1)) or ((y4 >= y2) and (y4 <= y1))):
		return True, 0
	else:
		p1 = [x1, y1]
		p2 = [x2, y2]
		p4 = [x4, y4]
		dist_a = distance(p1, p4)
		dist_b = distance(p2, p4)
		return False, min(dist_a, dist_b)

def find_route(vehicle_track, routes_coords):
	ROUTE_WIDTH = 20
	_, start_x, start_y, _ = vehicle_track[0]
	start_idx = 0
	start_is_on_route = False

	idx2dist = {}

	for k, route_coord in enumerate(routes_coords):
		P1 = np.array(route_coord[0])
		P2 = np.array(route_coord[1])

		if not start_is_on_route:
			P3 = np.array([start_x, start_y])
			start_is_on_route, dist = is_on_route(P3, P1, P2, ROUTE_WIDTH)
			idx2dist[k] = dist
			if start_is_on_route:
				start_idx = k
				break

	if not start_is_on_route:
		min_dist = sys.float_info.max
		for k_ in idx2dist.keys():
			per_dist = idx2dist[k_]
			if per_dist < min_dist:
				min_dist = per_dist
				start_idx = k_

	_, stop_x, stop_y, _  = vehicle_track[-1]
	stop_idx = 0
	stop_is_on_route = False

	idx2dist = {}

	for k, route_coord in enumerate(routes_coords):
		P1 = np.array(route_coord[0])
		P2 = np.array(route_coord[1])
		
		if not stop_is_on_route:
			P3 = np.array([stop_x, stop_y])
			stop_is_on_route, dist = is_on_route(P3, P1, P2, ROUTE_WIDTH)
			idx2dist[k] = dist
			if stop_is_on_route:
				stop_idx = k
				break

	if not stop_is_on_route:
		min_dist = sys.float_info.max
		for k_ in idx2dist.keys():
			per_dist = idx2dist[k_]
			if per_dist < min_dist:
				min_dist = per_dist
				stop_idx = k_

	return start_idx, stop_idx, vehicle_track

def link_nodes(start_node, stop_nodes, routes_dict, node_lists, parent_node):
	next_nodes = routes_dict[start_node]
	if parent_node in next_nodes:
		idx = next_nodes.index(parent_node)
		next_nodes.pop(idx)
	if len(next_nodes) == 0:
		return False

	for next_node in next_nodes:
		node_lists.append(next_node)
		if next_node in stop_nodes:
			if next_node == stop_nodes[0]:
				node_lists.append(stop_nodes[1])
			else:
				node_lists.append(stop_nodes[0])
			return True
		else:
			found = link_nodes(next_node, stop_nodes, routes_dict, node_lists, start_node)
			if not found:
				node_lists.pop(-1)
			else:
				return True

def gen_routes_dict():
	routes_dict = {}
	for route in routes_nodirection:
		nodes = route.split('#')
		if not nodes[0] in routes_dict.keys():
			routes_dict[nodes[0]] = []
			routes_dict[nodes[0]].append(nodes[1])
		elif not nodes[1] in routes_dict[nodes[0]]:
			routes_dict[nodes[0]].append(nodes[1])

		if not nodes[1] in routes_dict.keys():
			routes_dict[nodes[1]] = []
			routes_dict[nodes[1]].append(nodes[0])
		elif not nodes[0] in routes_dict[nodes[1]]:
			routes_dict[nodes[1]].append(nodes[0])

	return routes_dict

def find_route_name(start_idx, stop_idx, vehicle_track, routes_coords):
	routes_dict = gen_routes_dict()
	if start_idx != stop_idx:
		start_route_nodes = routes_nodirection[start_idx].split('#')
		stop_route_nodes = routes_nodirection[stop_idx].split('#')
		for k_start, node in enumerate(start_route_nodes):
			if node in stop_route_nodes:
				k_stop = stop_route_nodes.index(node)
				start_route_name = 'edgeL-' + start_route_nodes[1-k_start] + '-' + start_route_nodes[k_start]
				stop_route_name = 'edgeL-' + stop_route_nodes[k_stop] + '-' + stop_route_nodes[1-k_stop]
				return [start_route_name, stop_route_name]
		node_lists = []
		node_lists.append(start_route_nodes[0])
		link_nodes(start_route_nodes[0], stop_route_nodes, routes_dict, node_lists, '0')
		if not node_lists[1] in start_route_nodes:
			node_lists.insert(0, start_route_nodes[1])
		routes_list = []
		for k, node in enumerate(node_lists[:-1]):
			routes_list.append('edgeL-' + node_lists[k] + '-' + node_lists[k+1])
		return routes_list
	else:
		start_x, start_y = vehicle_track[0]
		stop_x, stop_y   = vehicle_track[-1]
		name_route_nodes = routes_nodirection[start_idx].split('#')
		scale_route_nodes = routes_coords[start_idx]
		dist_start = distance(scale_route_nodes[0], [start_x, start_y])
		dist_stop  = distance(scale_route_nodes[0], [stop_x, stop_y])
		if dist_start < dist_stop:
			route_name = 'edgeL-' + name_route_nodes[0] + '-' + name_route_nodes[1]
		else:
			route_name = 'edgeL-' + name_route_nodes[1] + '-' + name_route_nodes[0]
		return [route_name]

def get_rel_pos(route, start_pos, routes_coords, is_depart=True):
	depart_nodes = route.split('-')
	route_nodir = depart_nodes[1] + '#' + depart_nodes[2]
	if route_nodir in routes_nodirection:
		idx = routes_nodirection.index(route_nodir)
		nodes = routes_coords[idx]
		node_start_coord = nodes[0]
		node_end_coord = nodes[1]
	else:
		route_nodir = depart_nodes[2] + '#' + depart_nodes[1]
		idx = routes_nodirection.index(route_nodir)
		nodes = routes_coords[idx]
		node_start_coord = nodes[1]
		node_end_coord = nodes[0]
	rel_pos = p_distance(node_start_coord, node_end_coord, start_pos)

	if is_depart and rel_pos > 10:
		rel_pos -= 10

	if (not is_depart) and (rel_pos > distance(node_start_coord, node_end_coord)):
		rel_pos = 1.3141592653589793238462

	return rel_pos

def get_pos(routes, track, routes_coords):
	depart_route = routes[0]
	arrival_route = routes[-1]
	depart_position = get_rel_pos(depart_route, track[0], routes_coords, True)
	arrival_position = get_rel_pos(arrival_route, track[-1], routes_coords, False)

	return depart_position, arrival_position

'''
<node id="node1_1" x="521729" y="58722" type="priority"/>

<node id="node1" x="521676" y="58126" type="traffic_light"/> 
<node id="node2" x="521581" y="57500" type="traffic_light"/> 
<node id="node3" x="521516" y="57070" type="traffic_light"/> 
<node id="node4" x="521454" y="56672" type="traffic_light"/> 
<node id="node5" x="521427" y="55870" type="traffic_light"/> 
<node id="node6" x="521418" y="54881" type="traffic_light"/> 
<node id="node7" x="521406" y="53986" type="traffic_light"/> 

<node id="node7_1" x="521417" y="53371" type="priority"/>
'''

def plot_didi_map():
	nodes = [[521729, 58722], [521676, 58126], [521581, 57500], [521516, 57070], [521454, 56672], \
			[521427, 55870], [521418, 54881], [521406, 53986], [521417, 53371]]
	#plt.xlim(521400 - 200, 521769 + 200)
	#plt.ylim(53371 - 200, 58722 + 200)
	route_nodes = np.array(nodes)
	x, y = route_nodes.T
	plt.scatter(x, y)

def pick_from_abnormal(tracks, abnormal_points):
	valid_pnts = []
	anchor = abnormal_points[-1]
	valid_pnts.append(anchor)
	k = len(abnormal_points)
	for i in range(k-1):
		j = k-2-i
		cur_pnt = abnormal_points[j]
		last_pnt = valid_pnts[-1]
		last_timestamp, last_x, last_y, last_speed = cur_pnt
		timestamp, x, y, speed = last_pnt
		if is_continuous_track(last_timestamp, last_x, last_y, last_speed, float(timestamp), float(x), float(y), float(speed)):
			valid_pnts.append(cur_pnt)

	valid_pnts.reverse()
	for valid_pnt in valid_pnts:
		tracks[-1].append(valid_pnt)

def gen_routes(di_track_file, debug):
	next_route_dict = {
		'edgeL-1_0-1': 'edgeL-1-1_2',
		'edgeL-1_1-1': 'edgeL-1-2',
		'edgeL-2-1': 'edgeL-1-1_1',
		'edgeL-1_2-1': 'edgeL-1-1_0',
		'edgeL-1-2': 'edgeL-2-3',
		'edgeL-3-2': 'edgeL-2-1',
		'edgeL-2_0-2': 'edgeL-2-2_1',
		'edgeL-2_1-2': 'edgeL-2-2_0',
		'edgeL-2-3': 'edgeL-3-4',
		'edgeL-4-3': 'edgeL-3-2',
		'edgeL-3_0-3': 'edgeL-3-3_1',
		'edgeL-3_1-3': 'edgeL-3-3_0',
		'edgeL-3-4': 'edgeL-4-5',
		'edgeL-5-4': 'edgeL-4-3',
		'edgeL-4_0-4': 'edgeL-4-4_1',
		'edgeL-4_1-4': 'edgeL-4-4_0',
		'edgeL-4-5': 'edgeL-5-6',
		'edgeL-6-5': 'edgeL-5-4',
		'edgeL-5_0-5': 'edgeL-5-5_1',
		'edgeL-5_1-5': 'edgeL-5-5_0',
		'edgeL-5-6': 'edgeL-6-7',
		'edgeL-7-6': 'edgeL-6-5',
		'edgeL-6_0-6': 'edgeL-6-6_1',
		'edgeL-6_1-6': 'edgeL-6-6_0',
		'edgeL-6-7': 'edgeL-7-7_1',
		'edgeL-7_1-7': 'edgeL-7-6',
		'edgeL-7_0-7': 'edgeL-7-7_2',
		'edgeL-7_2-7': 'edgeL-7-7_0'
	}
	TIMESTAMP_INTERVAL_MAX = 10
	f_auto_gen_routes_xml = open('/home/nlp/bigsur/devel/didi/sumo/didi_contest/di-auto-28.rou.xml', 'w')
	f_track = open(di_track_file, 'r')
	line_coords = []
	vehicle_tracks = {}
	lines = f_track.readlines()
	abnormal_ctr = 0
	abnormal_points = []
	for line in lines:
		line = line.split(',')
		vehicle_id = line[0]
		timestamp = line[1]
		x_coordinate = line[2]
		y_coordinate = line[3]
		speed = line[4]
		category = line[5]
		if vehicle_id in vehicle_tracks.keys():
			last_timestamp, last_x, last_y, last_speed = vehicle_tracks[vehicle_id][-1][-1]
			if is_continuous_track(last_timestamp, last_x, last_y, last_speed, float(timestamp), float(x_coordinate), float(y_coordinate), float(speed)):
				vehicle_tracks[vehicle_id][-1].append([float(timestamp), float(x_coordinate), float(y_coordinate), float(speed)])
			else:
				abnormal_ctr += 1
				abnormal_points.append([float(timestamp), float(x_coordinate), float(y_coordinate), float(speed)])
				if abnormal_ctr == MAX_ABNORMAL_IN_CONTINUOUSE_ROUTE_ALLOWANCE:
					abnormal_ctr = 0
					vehicle_tracks[vehicle_id].append([])
					pick_from_abnormal(vehicle_tracks[vehicle_id], abnormal_points)
					#vehicle_tracks[vehicle_id][-1].append([float(timestamp), float(x_coordinate), float(y_coordinate), float(speed)])
					abnormal_points = []
				else:
					vehicle_tracks[vehicle_id][-1].append([last_timestamp, last_x, last_y, last_speed])
		else:
			abnormal_ctr = 0
			vehicle_tracks[vehicle_id] = []
			vehicle_tracks[vehicle_id].append([])
			vehicle_tracks[vehicle_id][-1].append([float(timestamp), float(x_coordinate), float(y_coordinate), float(speed)])

	route_lists = []
	invalid_tracks_ctr = 0
	for vehicle_id in vehicle_tracks:
		for k, vehicle_track in enumerate(vehicle_tracks[vehicle_id]):
			if len(vehicle_track) < 10:
				continue

			cur_vehicle_id = vehicle_id + '_' + str(k)

			fined_route_coords = gen_route_coords('/home/nlp/bigsur/devel/didi/sumo/didi_contest/di.kun.net.xml')		

			vehicle_track_ori = copy.deepcopy(vehicle_track)
			start_idx, stop_idx, fine_vehicle_track = find_route(vehicle_track, fined_route_coords)
			print(str(start_idx) + ',' + str(stop_idx))
			if start_idx == -1 and stop_idx == -1:
				invalid_tracks_ctr += 1
				print("invalid vehicle track len: " + str(len(fine_vehicle_track)))
				continue

			timestamp_start, start_pos_x, start_pos_y, depart_spd = vehicle_track_ori[0]
			timestamp_stop, stop_pos_x, stop_pos_y, arrival_spd = vehicle_track_ori[-1]
			timestamp_start -= TIMESTAMP_BASE
			timestamp_stop  -= TIMESTAMP_BASE

			vehicle_pos = []
			vehicle_pos.append([start_pos_x, start_pos_y])
			vehicle_pos.append([stop_pos_x, stop_pos_y])	

			routes_list = find_route_name(start_idx, stop_idx, vehicle_pos, fined_route_coords)
			#for route in routes_list:
			#	print(route)

			depart_pos, arrival_pos = get_pos(routes_list, vehicle_pos, fined_route_coords)

			# special return result indicating adding last route.
			if arrival_pos == 1.3141592653589793238462:
				routes_list.append(next_route_dict[routes_list[-1]])
				arrival_pos = 5.0

			routes_str = ' '.join(routes_list)

			route_lists.append([cur_vehicle_id, timestamp_start, depart_pos, depart_spd, arrival_pos, arrival_spd, routes_str])

			if debug:
				print(routes_str + ' ' + str(depart_pos) + ' ' + str(arrival_pos) + ' ' + vehicle_id)
				#if routes_str == "edgeL-6_0-6 edgeL-6-5" or routes_str == "edgeL-6_1-6 edgeL-6-5":
				plt.xlim(521400 - 200, 521769 + 200)
				plt.ylim(53371 - 200, 58722 + 200)
				plot_didi_map()
				per_vehicle_track = np.array(vehicle_track_ori)
				timestamp, x, y, spd = per_vehicle_track.T
				plt.scatter(x[0], y[0], color='r')
				plt.plot(x, y, color = 'r')
				plt.show()

	print('invalid routes:' + str(invalid_tracks_ctr))

	if not debug:
		f_auto_gen_routes_xml.write('<routes>\n')
		f_auto_gen_routes_xml.write('   <vType id="type1" accel="0.8" decel="4.5" sigma="0.5" length="5" maxSpeed="102"/>\n')

		sorted_route_lists = sorted(route_lists,key=lambda x: x[1])

		for route_info in sorted_route_lists:
			cur_vehicle_id, timestamp_start, depart_pos, depart_spd, arrival_pos, arrival_spd, routes_str = route_info

			vehicle_content_str = '   <vehicle id="%s" type="type1" depart="%s" color="1,1,0" departPos="%s" departSpeed="%s" \
				arrivalPos="%s">\n' % (\
				cur_vehicle_id, str(timestamp_start), str(depart_pos), str(depart_spd), str(arrival_pos))

			route_content_str = '      <route edges="%s"/>\n' % (routes_str)

			f_auto_gen_routes_xml.write(vehicle_content_str)
			f_auto_gen_routes_xml.write(route_content_str)
			f_auto_gen_routes_xml.write('   </vehicle>\n')

		f_auto_gen_routes_xml.write('</routes>\n')
		f_auto_gen_routes_xml.close()

def split_routes(routes_file):
	net_tree = ET.ElementTree(file=routes_file)
	net_tree_root = net_tree.getroot()
	vehicles = net_tree_root.findall('vehicle')
	departs = []
	y = []
	for vehicle in vehicles:
		depart_time = float(vehicle.attrib['depart'])
		departs.append(depart_time)
		y.append(1)
	plt.xlim(0, 1130281)
	plt.ylim(0, 2)
	plt.scatter(departs, y)
	plt.show()

def finetune_routes():
	for x in range(0, 10):
		route_file = "di-auto.rou." + str(x) + ".xml"
		net_tree = ET.ElementTree(file=route_file)
		net_tree_root = net_tree.getroot()
		vehicles = net_tree_root.findall('vehicle')
		base_depart = True
		base_depart_time = 0.0
		for vehicle in vehicles:
			depart_time = float(vehicle.attrib['depart'])
			if base_depart:
				base_depart = False
				base_depart_time = depart_time
				vehicle.attrib['depart'] = "0.0"
			else:
				depart_time -= base_depart_time
				vehicle.attrib['depart'] = str(depart_time)

		net_tree.write(route_file)

def check_routes():
	route_file = "/home/nlp/bigsur/devel/didi/sumo/didi_contest/di-auto-28.rou.xml"
	net_tree = ET.ElementTree(file=route_file)
	net_tree_root = net_tree.getroot()
	vehicles = net_tree_root.findall('vehicle')
	for vehicle in vehicles:
		depart_pos = float(vehicle.attrib['departPos'])
		arrival_pos = float(vehicle.attrib['arrivalPos'])
		route = vehicle.find('route')
		edges = route.attrib['edges']
		edge_list = edges.split(' ')
		if len(edge_list) == 1:
			if depart_pos > arrival_pos:
				print(vehicle.attrib['id'])

def finetune_results():
	output_str = ""
	ori_output = "145 122 5 8 32 14 8 170 5 36 189 14 240 32 123 13 88 30 5 32 69 7 143 32 51 7 33 7 39 58 7 78 43"
	oris = ori_output.split(' ')
	num_tl = 7
	ori_ctr = 0
	tl_phases = [4, 4, 3, 5, 3, 4, 3]
	for tl_idx in range(num_tl):
		ofs = oris[ori_ctr]
		num_phase = tl_phases[tl_idx]

		cycle = 0
		for x in range(num_phase):
			if not (tl_idx == 1 and x == 0):
				oris[ori_ctr+x+1] = str(int(oris[ori_ctr+x+1]) + 3)
			cycle += int(oris[ori_ctr+x+1])
		tune_ofs = int(ofs)%cycle
		
		output_str += str(tune_ofs)
		ori_ctr += 1
		for phase in range(num_phase):
			phase_dur = oris[ori_ctr]
			output_str += ',' + phase_dur
			ori_ctr += 1
		output_str += ';'

	print(output_str[:-1])

def sort_didi_csv(di_track_file='/home/nlp/bigsur/data/diditech/vehicle_track.txt'):
	import csv
	import operator

	with open(di_track_file) as f:
		reader = csv.reader(f)
		sorted_by_vehicle_timestamp = sorted(reader, key=lambda col: (col[0], int(col[1])), reverse=False)

	new_csv_file = open('/home/nlp/bigsur/data/diditech/vehicle_track_sorted.txt', 'w')
	for line in sorted_by_vehicle_timestamp:
		new_csv_file.write("%s,%s,%s,%s,%s, %s\n" % (line[0], line[1], line[2], line[3], line[4], line[5]))

	new_csv_file.close()

if __name__ == '__main__':
    random.seed(200)

    #plot_didi_map()
    #sort_didi_csv()
    #calc_distance()
    #track_stats(di_track_file='/home/nlp/bigsur/data/diditech/vehicle_track.txt')
    #draw_veihcle_track(di_track_file='/home/nlp/bigsur/data/diditech/vehicle_track.txt')
    gen_routes(di_track_file='/home/nlp/bigsur/data/diditech/vehicle_track_sorted.txt', debug=False)
    #split_routes(routes_file='/home/nlp/bigsur/devel/didi/sumo/didi_contest/di-auto.rou.xml')
    #finetune_routes()
    #check_routes()

    #finetune_results()
