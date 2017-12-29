# -*- coding:utf-8 -*-
'''
xxxxxxxxx

'''
import random
import argparse
import xml.etree.cElementTree as ET

if __name__ == '__main__':
	file_srx = open("/home/nlp/bigsur/devel/didi/sumo/didi_contest/di-auto.rou.xml")
	lines = file_srx.readlines()
	num_lines=len(lines)-3
	for i in range(10):
		file_object_log = open('/home/nlp/bigsur/devel/didi/sumo/didi_contest/di-auto.rou' + str(i)+'.xml', 'w')
		file_object_log.write('<routes>\n')
		file_object_log.write('   <vType id="type1" accel="0.8" decel="4.5" sigma="0.5" length="5" maxSpeed="102"/>\n')
		if i==9:
	                for j in range(2+i*num_lines/10/3*3,len(lines)-2):
        	                file_object_log.write(lines[j])
	        	file_object_log.write('</routes>\n')
			break;
		for j in range(2+i*num_lines/10/3*3,2+(i+1)*num_lines/10/3*3):
			file_object_log.write(lines[j])
		file_object_log.write('</routes>\n')
