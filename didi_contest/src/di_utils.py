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

X_OFS = 1000
Y_OFS = 1000

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
	for line in lines[1:]:
		line = line.split(',')
		vehicle_id = line[0]
		timestamp = line[1]
		x_coordinate = line[2]
		y_coordinate = line[3]
		speed = line[4]
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

	print(min_x)
	print(max_x)
	print(min_y)
	print(max_y)
	print(max_t)
	print(min_t)

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

def distance(p0, p1):
    return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)

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
	distance = norm(np.cross(line_end_b-line_end_a, line_end_a-vehicle_coord))/norm(line_end_b-line_end_a)
	if distance > route_width:
		return False

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
		return True
	else:
		return False

def find_route(vehicle_track, routes_coords):
	ROUTE_WIDTH = 20
	_, start_x, start_y, _ = vehicle_track[0]
	start_idx = 0
	start_is_on_route = False

	while not start_is_on_route:
		for k, route_coord in enumerate(routes_coords):
			P1 = np.array(route_coord[0])
			P2 = np.array(route_coord[1])

			if not start_is_on_route:
				P3 = np.array([start_x, start_y])
				start_is_on_route = is_on_route(P3, P1, P2, ROUTE_WIDTH)
				if start_is_on_route:
					start_idx = k
					break
		if start_is_on_route:
			break

		vehicle_track.pop(0)
		if len(vehicle_track) == 0:
			break

		_, start_x, start_y, _ = vehicle_track[0]


	if len(vehicle_track) < 2:
		return -1, -1, vehicle_track

	_, stop_x, stop_y, _  = vehicle_track[-1]
	stop_idx = 0
	stop_is_on_route = False

	while not stop_is_on_route:
		for k, route_coord in enumerate(routes_coords):
			P1 = np.array(route_coord[0])
			P2 = np.array(route_coord[1])
			
			if not stop_is_on_route:
				P3 = np.array([stop_x, stop_y])
				stop_is_on_route = is_on_route(P3, P1, P2, ROUTE_WIDTH)
				if stop_is_on_route:
					stop_idx = k
					break
		if stop_is_on_route:
			break

		vehicle_track.pop(-1)
		if len(vehicle_track) < 2:
			break

		_, stop_x, stop_y, _ = vehicle_track[-1]

	if start_is_on_route and stop_is_on_route:
		return start_idx, stop_idx, vehicle_track
	else:
		return -1, -1, vehicle_track

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

def get_rel_pos(route, start_pos, routes_coords):
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
	depart_pos = p_distance(node_start_coord, node_end_coord, start_pos)
	return depart_pos

def get_pos(routes, track, routes_coords):
	depart_route = routes[0]
	arrival_route = routes[-1]
	depart_position = get_rel_pos(depart_route, track[0], routes_coords)
	arrival_position = get_rel_pos(arrival_route, track[-1], routes_coords)

	return depart_position, arrival_position

def gen_routes(di_track_file):
	TIMESTAMP_INTERVAL_MAX = 10
	f_auto_gen_routes_xml = open('/home/nlp/bigsur/devel/didi/sumo/didi_contest/di-auto.rou.xml', 'w')
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
			last_timestamp, _, _, _ = vehicle_tracks[vehicle_id][-1][-1]
			if float(timestamp) - float(last_timestamp) > TIMESTAMP_INTERVAL_MAX:
				vehicle_tracks[vehicle_id].append([])
				vehicle_tracks[vehicle_id][-1].append([float(timestamp), float(x_coordinate), float(y_coordinate), float(speed)])
			else:
				vehicle_tracks[vehicle_id][-1].append([float(timestamp), float(x_coordinate), float(y_coordinate), float(speed)])
		else:
			vehicle_tracks[vehicle_id] = []
			vehicle_tracks[vehicle_id].append([])
			vehicle_tracks[vehicle_id][-1].append([float(timestamp), float(x_coordinate), float(y_coordinate), float(speed)])

	route_lists = []
	invalid_tracks_ctr = 0
	for vehicle_id in vehicle_tracks:
		for k, vehicle_track in enumerate(vehicle_tracks[vehicle_id]):
			cur_vehicle_id = vehicle_id + '_' + str(k)

			fined_route_coords = gen_route_coords('/home/nlp/bigsur/devel/didi/sumo/didi_contest/di.han.net.xml')		

			start_idx, stop_idx, fine_vehicle_track = find_route(vehicle_track, fined_route_coords)
			print(str(start_idx) + ',' + str(stop_idx))
			if start_idx == -1 and stop_idx == -1:
				invalid_tracks_ctr += 1
				print("invalid vehicle track len: " + str(len(fine_vehicle_track)))
				continue

			timestamp_start, start_pos_x, start_pos_y, depart_spd = fine_vehicle_track[0]
			timestamp_stop, stop_pos_x, stop_pos_y, arrival_spd = fine_vehicle_track[-1]
			timestamp_start -= TIMESTAMP_BASE
			timestamp_stop  -= TIMESTAMP_BASE

			vehicle_pos = []
			vehicle_pos.append([start_pos_x, start_pos_y])
			vehicle_pos.append([stop_pos_x, stop_pos_y])	

			routes_list = find_route_name(start_idx, stop_idx, vehicle_pos, fined_route_coords)
			#for route in routes_list:
			#	print(route)
			routes_str = ' '.join(routes_list)

			depart_pos, arrival_pos = get_pos(routes_list, vehicle_pos, fined_route_coords)

			route_lists.append([cur_vehicle_id, timestamp_start, depart_pos, depart_spd, arrival_pos, arrival_spd, routes_str])

	print('invalid routes:' + str(invalid_tracks_ctr))

	f_auto_gen_routes_xml.write('<routes>\n')
	f_auto_gen_routes_xml.write('   <vType id="type1" accel="0.8" decel="4.5" sigma="0.5" length="5" maxSpeed="70"/>\n')

	sorted_route_lists = sorted(route_lists,key=lambda x: x[1])

	for route_info in sorted_route_lists:
		cur_vehicle_id, timestamp_start, depart_pos, depart_spd, arrival_pos, arrival_spd, routes_str = route_info

		vehicle_content_str = '   <vehicle id="%s" type="type1" depart="%s" color="1,1,0" departPos="%s" \
			departSpeed="%s"  arrivalPos="%s" arrivalSpeed="%s">\n' % (\
			cur_vehicle_id, str(timestamp_start), str(depart_pos), str(depart_spd), str(arrival_pos), str(arrival_spd))

		route_content_str = '      <route edges="%s"/>\n' % (routes_str)

		f_auto_gen_routes_xml.write(vehicle_content_str)
		f_auto_gen_routes_xml.write(route_content_str)
		f_auto_gen_routes_xml.write('   </vehicle>\n')

	f_auto_gen_routes_xml.write('</routes>\n')
	f_auto_gen_routes_xml.close()

if __name__ == '__main__':
    random.seed(200)

    #calc_distance()
    #track_stats(di_track_file='/home/nlp/bigsur/data/diditech/vehicle_track.txt')
    #draw_veihcle_track(di_track_file='/home/nlp/bigsur/data/diditech/vehicle_track.txt')

    gen_routes(di_track_file='/home/nlp/bigsur/data/diditech/vehicle_track.txt')
