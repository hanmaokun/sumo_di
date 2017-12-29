import os
splittime=[0,50000,200000,380000,480000,570000,650000,800000,990000,1050000,10e100000000000000]

if __name__ == '__main__':
	ori_xml=open('di-auto-28.rou.xml')
	lines=ori_xml.readlines()
	num_line=len(lines)-3
	for i in range(10):
		dst_xml=open('di-auto.rou.'+str(i)+'.xml','w')
		dst_xml.write('<routes>\n')
		dst_xml.write('   <vType id="type1" accel="0.8" decel="4.5" sigma="0.5" length="5" maxSpeed="102"/>\n')
		for j in range(0,num_line/3):
			time_=float(lines[2+j*3].split('"')[5])
			#print time_split
			if time_>=splittime[i] and time_<splittime[i+1]:
				for k in range(3):
					dst_xml.write(lines[2+j*3+k])
		dst_xml.write('</routes>\n')
		dst_xml.close()