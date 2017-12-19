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

routes_nodirection = ['1#2', '2#3', '3#4', '4#5', '5#6', '6#7', \
						'1_0#1', '1_1#1', '1_2#1', \
						'2_0#2', \
						'3_0#3', '3_1#3', \
						'4_0#4', '4_1#4', \
						'5_0#5', '5_1#5', \
						'6_0#6', '6_1#6', \
						'7_0#7', '7_1#7', '7_3#7' \
					]
routes_coords = [[[521677, 58109], [521580,57466]], [[521580, 57466], [521520,57059]], [[521520, 57059], [521452,56668]], \
				 [[521452, 56668], [521433,55855]], [[521433, 55855], [521411,54822]], [[521411, 54822], [521400,53998]], \
				 [[521296, 58295], [521677,58109]], [[521729, 58722], [521677,58109]], [[521769, 58107], [521677,58109]], \
				 [[521542, 57515], [521580,57466]], \
				 [[521387, 57093], [521520,57059]], [[521732, 57032], [521520,57059]], \
				 [[520955, 56693], [521452,56668]], [[521920, 56605], [521452,56668]], \
				 [[521378, 55918], [521433,55855]], [[521660, 55437], [521433,55855]], \
				 [[520987, 54810], [521411,54822]], [[521780, 55117], [521411,54822]], \
				 [[521000, 53960], [521400,53998], [521417, 53371], [521400,53998], [521856, 54023], [521400,53998]] \
				]

def distance(p0, p1):
    return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)

def find_route(vehicle_track):
	x_ofs = 10
	y_ofs = 0
	_, start_x, start_y, _ = vehicle_track[0]
	_, stop_x, stop_y, _  = vehicle_track[-1]
	dist_start_min =  sys.float_info.max
	dist_stop_min =  sys.float_info.max
	start_idx = 0
	stop_idx = 0
	for k, route_coord in enumerate(routes_coords):
		P1 = np.array(route_coord[0])
		P2 = np.array(route_coord[1])

		#P1 = p1 if p1[0] < p2[0] else p2
		#P2 = p1 if p1[0] > p2[0] else p2

		P3 = np.array([start_x, start_y])
		dist_start = norm(np.cross(P2-P1, P1-P3))/norm(P2-P1)
		if dist_start < dist_start_min:
			if P3[0] > (min(P1[0], P2[0])-x_ofs) and P3[0] < (max(P1[0], P2[0])+x_ofs):
				#and P3[1] > min(P1[1], P2[1])-y_ofs and P3[1] < max(P1[1], P2[1])+y_ofs:
				if distance(P3, P1) < distance(P1, P2):
					dist_start_min = dist_start
					start_idx = k
		
		P3 = np.array([stop_x, stop_y])
		dist_stop = norm(np.cross(P2-P1, P1-P3))/norm(P2-P1)
		if dist_stop < dist_stop_min:
			if P3[0] > min(P1[0], P2[0])-x_ofs and P3[0] < max(P1[0], P2[0])+x_ofs:
				#and P3[1] > min(P1[1], P2[1])-y_ofs and P3[1] < max(P1[1], P2[1])+y_ofs:
				if distance(P3, P1) < distance(P1, P2):
					dist_stop_min = dist_stop
					stop_idx = k

	return start_idx, stop_idx

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

def find_route_name(start_idx, stop_idx, vehicle_track):
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

def get_rel_pos(route, start_pos):
	depart_nodes = route.split('-')
	route_nodir = depart_nodes[1] + '#' + depart_nodes[2]
	if route_nodir in routes_nodirection:
		idx = routes_nodirection.index(route_nodir)
		nodes = routes_coords[idx]
		start_node_coord = nodes[0]
	else:
		route_nodir = depart_nodes[2] + '#' + depart_nodes[1]
		idx = routes_nodirection.index(route_nodir)
		nodes = routes_coords[idx]
		start_node_coord = nodes[1]
	depart_pos = distance(start_node_coord, start_pos)
	return depart_pos

def get_pos(routes, track):
	depart_route = routes[0]
	arrival_route = routes[-1]
	depart_position = get_rel_pos(depart_route, track[0])
	arrival_position = get_rel_pos(arrival_route, track[-1])

	return depart_position, arrival_position

def gen_routes(di_track_file):
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
			if float(timestamp) - float(last_timestamp) > 4:
				vehicle_tracks[vehicle_id].append([])
				vehicle_tracks[vehicle_id][-1].append([float(timestamp), float(x_coordinate), float(y_coordinate), float(speed)])
			else:
				vehicle_tracks[vehicle_id][-1].append([float(timestamp), float(x_coordinate), float(y_coordinate), float(speed)])
		else:
			vehicle_tracks[vehicle_id] = []
			vehicle_tracks[vehicle_id].append([])
			vehicle_tracks[vehicle_id][-1].append([float(timestamp), float(x_coordinate), float(y_coordinate), float(speed)])

	route_lists = []
	for vehicle_id in vehicle_tracks:
		for k, vehicle_track in enumerate(vehicle_tracks[vehicle_id]):
			cur_vehicle_id = vehicle_id + '_' + str(k)

			if cur_vehicle_id == "0e83daf2710839ed7c68d5626558e695_5":
				print(cur_vehicle_id)

			timestamp_start, start_pos_x, start_pos_y, depart_spd = vehicle_track[0]
			timestamp_stop, stop_pos_x, stop_pos_y, arrival_spd = vehicle_track[-1]
			timestamp_start -= TIMESTAMP_BASE
			timestamp_stop  -= TIMESTAMP_BASE

			vehicle_pos = []
			vehicle_pos.append([start_pos_x, start_pos_y])
			vehicle_pos.append([stop_pos_x, stop_pos_y])			

			start_idx, stop_idx = find_route(vehicle_track)
			print(str(start_idx) + ',' + str(stop_idx))
			routes_list = find_route_name(start_idx, stop_idx, vehicle_pos)
			#for route in routes_list:
			#	print(route)
			routes_str = ' '.join(routes_list)

			depart_pos, arrival_pos = get_pos(routes_list, vehicle_pos)

			route_lists.append([cur_vehicle_id, timestamp_start, depart_pos, depart_spd, arrival_pos, arrival_spd, routes_str])

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
