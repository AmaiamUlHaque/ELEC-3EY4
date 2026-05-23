#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import sys
import cv2
import time
import rospy
import tf2_ros
import math


from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, OccupancyGrid
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive


class OccupancyGridMap:
    def __init__(self):

        self.x = 0
        self.y = 0;
        self.yaw = 0;
        self.has_odom = False

        # Topics & Subs, Pubs

        self.t_prev = rospy.get_time()

        # ---------------------------------------------------------------------------
        # Read parameters from params.yaml 
        odom_topic              = rospy.get_param('~odom_topic')
        lidarscan_topic         = rospy.get_param('~scan_topic')
        map_topic               = rospy.get_param('~map_topic')
        self.occ_map_topic      = rospy.get_param('~occ_map_topic')
        self.odom_frame         = rospy.get_param('~odom_frame')
        self.max_lidar_range    = rospy.get_param('~scan_range')
        self.scan_beams         = rospy.get_param('~scan_beams')
        self.scan_disToBL       = rospy.get_param('~scan_distance_to_base_link')

        # Laser offset transformation (from base_link to laser)
        self.laser_offset_x = self.scan_disToBL  
        self.laser_offset_y = 0.0
        self.laser_offset_yaw = math.pi

        # ---------------------------------------------------------------------------
        # Read the map parameters from *.yaml file 
        self.map_res        = rospy.get_param('~map_res')
        self.map_width      = rospy.get_param('~map_width')
        self.map_height     = rospy.get_param('~map_height')
        self.object_size    = rospy.get_param('~object_size')
        
        # Probability parameters for occupancy grid mapping
        self.p_unknown  = 0.5
        self.p_occ      = rospy.get_param('~p_occ')
        self.p_free     = rospy.get_param('~p_free')
 
        # Log odds values for inverse sensor model
        self.LOG_ODDS_MAX   = 100.0 # Log-odds clamp limit to prevent overflow
        self.l_unknown      = self.probToLogOdds(self.p_unknown)
        self.l_occ          = self.probToLogOdds(self.p_occ)
        self.l_free         = self.probToLogOdds(self.p_free)
        self.l_prior        = self.probToLogOdds(self.p_unknown)

        
        # ---------------------------------------------------------------------------
        # Initialize the map meta info in the Occupancy Grid Message 
        # e.g., frame_id, stamp, resolution, width, height, etc.
        self.map_occ_grid_msg = OccupancyGrid()
        self.map_occ_grid_msg.header.frame_id = self.odom_frame
        self.map_occ_grid_msg.header.stamp = rospy.Time.now()
        self.map_occ_grid_msg.info.width = self.map_width
        self.map_occ_grid_msg.info.height = self.map_height
        self.map_occ_grid_msg.info.resolution = self.map_res
        self.map_occ_grid_msg.info.origin.position.x = -(self.map_width*self.map_res/2)
        self.map_occ_grid_msg.info.origin.position.y = -(self.map_height*self.map_res/2)
        self.map_occ_grid_msg.info.origin.orientation.x=0
        self.map_occ_grid_msg.info.origin.orientation.y=0
        self.map_occ_grid_msg.info.origin.orientation.z=0
        self.map_occ_grid_msg.info.origin.orientation.w=1

        # ---------------------------------------------------------------------------
        # Initialize the cell occupancy probabilites to 0.5 (unknown) with all cell data in Occupancy Grid Message set to unknown 
        self.map_occ_grid_msg.data =[-1] * (self.map_width *self.map_height)
        self.log_odds = [ [self.l_prior for _ in range(self.map_width)] for _ in range(self.map_height)]


        # ---------------------------------------------------------------------------
        # Subscribe to Lidar scan and odomery topics with corresponding lidar_callback() and odometry_callback() functions 
        self.scan_sub = rospy.Subscriber(lidarscan_topic, LaserScan, self.lidar_callback, queue_size=1)
        self.odom_sub = rospy.Subscriber(odom_topic, Odometry, self.odom_callback, queue_size=1)
        
        # ---------------------------------------------------------------------------
        # Create a publisher for the Occupancy Grid Map 
        self.occupancy_grid_map_pub = rospy.Publisher(self.occ_map_topic, OccupancyGrid, queue_size=1)
        self.occupancy_grid_map_pub = rospy.Publisher(self.occ_map_topic, OccupancyGrid, queue_size=1)

        # Print map params and info
        print('\nOccupancy Grid Map node initialised.')
        print('Occupancy Grid Map initialized with map size: ', self.map_width, 'x' , self.map_height, 'resolution: ', self.map_res)
        print('Occupancy thresholds: p_occ=', self.p_occ, ', p_free=', self.p_free, ', p_unknown=', self.p_unknown)
        print('Occupancy thresholds: l_occ=', self.l_occ, ', l_free=', self.l_free, ', l_unknown=', self.l_unknown)
        print('Laser offset: x=', self.laser_offset_x, ' y=', self.laser_offset_y, ' yaw=', self.laser_offset_yaw)
        

    # ---------------------------------------------------------------------------
    # lidar_callback () uses the current LiDAR scan and Wheel Odometry data to uddate and publish the Grid Occupancy map 
    def lidar_callback(self, data):
        
        print('\nlidar_callback start')
        
        # Check if we have odometry data
        if not self.has_odom:
            print('Waiting for odometry data...')
            return
        
        ranges = data.ranges  # get the range data from the lidar scan message
        
        # Shift the position of the lidar from the base_link to get the actual position of the lidar in the odom frame
        self.lidar_x = self.x + self.laser_offset_x * math.cos(self.yaw)  # shift the position of the lidar from the base_link
        self.lidar_y = self.y + self.laser_offset_x * math.sin(self.yaw)
        self.lidar_yaw = (self.yaw + math.pi)  # shift the orientation, add pi because lidar faces backwards
        
        # Process each cell in the occupancy grid
        for j in range(0, self.map_height):
            for i in range(0, self.map_width): 
                
                x = (-self.map_width/2 + 0.5 + i) * self.map_res
                y = (-self.map_height/2 + 0.5 + j) * self.map_res

                r = math.sqrt((x - self.lidar_x)**2 + (y - self.lidar_y)**2)
                angle_of_point = (math.atan2(y - self.lidar_y, x - self.lidar_x) - self.lidar_yaw)
                beam_index = int(round((angle_of_point - data.angle_min) / data.angle_increment)) % len(data.ranges)
                beam_data = data.ranges[beam_index]
                
                delta = self.object_size * (math.sqrt(2) / 2)
                
                # Ignore invalid measurement
                if beam_data < data.range_min or beam_data > data.range_max:
                    continue        
                # Ignore cells beyond measurement
                if r > min(self.max_lidar_range, beam_data + delta):
                    continue
                elif abs(r - beam_data) <= delta:
                    # Cell is occupied (beam endpoint falls in this cell)
                    self.log_odds[i][j] = min(self.log_odds[i][j] + self.l_occ, self.LOG_ODDS_MAX)
                elif r < beam_data:
                # Cell is free (between robot and obstacle)
                    self.log_odds[i][j] = max(self.log_odds[i][j] + self.l_free, -self.LOG_ODDS_MAX)
                
                # Convert log-odds to prob
                prob = self.logOddsToProb(self.log_odds[i][j])
                
                
                # Update occupancy grid msg
                print('updating map_occ_grid_msg.data for cell above^')

                # CHECK WITH PROB
                if prob >= self.p_occ:
                    self.map_occ_grid_msg.data[j * self.map_width + i] = 100  # occupied
                elif prob <= self.p_free:
                    self.map_occ_grid_msg.data[j * self.map_width + i] = 0  # free
                else:
                    self.map_occ_grid_msg.data[j * self.map_width + i] = -1  # unknown

                # # CHECK WITH LOGODD
                # if self.log_odds[i][j] >= self.l_occ:
                #     self.map_occ_grid_msg.data[j * self.map_width + i] = 100  # occupied
                # elif self.log_odds[i][j] <= self.l_free:
                #     self.map_occ_grid_msg.data[j * self.map_width + i] = 0  # free
                # else:
                #     self.map_occ_grid_msg.data[j * self.map_width + i] = -1  # unknown



                print('end of [i][j]')

                # Print prob and logodd of each cell -->idk if this good idea...oopsies...gonna find out
                print('cell [',i,'][',j,'] prob=', prob, '& logodd=', self.log_odds[i][j], 'data=', (self.map_occ_grid_msg.data[j*self.map_width+i]))


                # Publish to map topic
                # self.map_occ_grid_msg.header.stamp = rospy.Time.now()
                # self.occupancy_grid_map_pub.publish(self.map_occ_grid_msg)


            # # Publish to map topic
            self.map_occ_grid_msg.header.stamp = rospy.Time.now()
            self.occupancy_grid_map_pub.publish(self.map_occ_grid_msg)
            print('Map updated with scan containing', len(data.ranges), 'beams')

        # Publish to map topic
        # self.map_occ_grid_msg.header.stamp = rospy.Time.now()
        # self.occupancy_grid_map_pub.publish(self.map_occ_grid_msg)
        # print('Map updated with scan containing', len(data.ranges), 'beams') 
        print('lidar_callback end\n')
        
        

    # ---------------------------------------------------------------------------
    # odom_callback() retrives the wheel odometry data from the publsihed odom_msg
    def odom_callback(self, odom_msg):
        self.x = odom_msg.pose.pose.position.x
        self.y = odom_msg.pose.pose.position.y
        self.z = odom_msg.pose.pose.orientation.z
        self.w = odom_msg.pose.pose.orientation.w
        self.yaw = 2* math.atan2(self.z, self.w)
        self.has_odom = True
        print('Odometry updated: x=', self.x, ' y=', self.y, ' yaw=', self.yaw)



    # ---------------------------------------------------------------------------
    # HELPER FUNCTIONS

    def probToLogOdds(self, prob):
        if (prob <= 0.0): 
            return -self.LOG_ODDS_MAX
        if (prob >= 1.0): 
            return self.LOG_ODDS_MAX
        return math.log(prob / (1.0 - prob))


    def logOddsToProb(self, val):
        return 1.0 - 1.0 / (1.0 + math.exp(val))


    def logOddsToOccupancy(self, val):
        prob = self.logOddsToProb(val)

        if (prob > self.p_occ):
            return 100 #occupied
        if (prob < self.p_free):
            return 0 #free
        return -1


    def gridToIndex(self, row, col):
        return (row*self.map_width + col)


    # Convert world coordinates to grid cell indices
    def worldToGrid(self, x, y):
        # Get origin coordinates
        origin_x = -(self.map_width * self.map_res / 2)
        origin_y = -(self.map_height * self.map_res / 2)
        
        col = int((x - origin_x) / self.map_res)
        row = int((y - origin_y) / self.map_res)
        
        # Check bounds
        if row < 0 or row >= self.map_height or col < 0 or col >= self.map_width:
            return (-1, -1)
        return (row, col)


    def laserToOdom(self, laser_x, laser_y):
        # Transform from laser to base_link
        base_x = laser_x * math.cos(self.laser_offset_yaw) - laser_y * math.sin(self.laser_offset_yaw) + self.laser_offset_x
        base_y = laser_x * math.sin(self.laser_offset_yaw) + laser_y * math.cos(self.laser_offset_yaw) + self.laser_offset_y
        
        # Transform from base_link to odom using current robot pose
        odom_x = self.x + base_x * math.cos(self.yaw) - base_y * math.sin(self.yaw)
        odom_y = self.y + base_x * math.sin(self.yaw) + base_y * math.cos(self.yaw)
        
        return (odom_x, odom_y)


    # Calculate the range from laser origin to a cell center
    def calculateCellRange(self, cell_x, cell_y):
        # Transform cell center from odom to base_link
        dx_odom = cell_x - self.x
        dy_odom = cell_y - self.y
        base_x = dx_odom * math.cos(self.yaw) + dy_odom * math.sin(self.yaw)
        base_y = -dx_odom * math.sin(self.yaw) + dy_odom * math.cos(self.yaw)
        
        # Transform from base_link to laser
        laser_x = (base_x - self.laser_offset_x) * math.cos(self.laser_offset_yaw) + (base_y - self.laser_offset_y) * math.sin(self.laser_offset_yaw)
        laser_y = -(base_x - self.laser_offset_x) * math.sin(self.laser_offset_yaw) + (base_y - self.laser_offset_y) * math.cos(self.laser_offset_yaw)
        
        return math.sqrt(laser_x * laser_x + laser_y * laser_y)


    # Calculate the angle of a cell relative to laser frame
    def calculateCellAngle(self, cell_x, cell_y):
        # Transform cell center from odom to base_link
        dx_odom = cell_x - self.x
        dy_odom = cell_y - self.y
        base_x = dx_odom * math.cos(self.yaw) + dy_odom * math.sin(self.yaw)
        base_y = -dx_odom * math.sin(self.yaw) + dy_odom * math.cos(self.yaw)
        
        # Transform from base_link to laser
        laser_x = (base_x - self.laser_offset_x) * math.cos(self.laser_offset_yaw) + (base_y - self.laser_offset_y) * math.sin(self.laser_offset_yaw)
        laser_y = -(base_x - self.laser_offset_x) * math.sin(self.laser_offset_yaw) + (base_y - self.laser_offset_y) * math.cos(self.laser_offset_yaw)
        
        return math.atan2(laser_y, laser_x)





def main(args):
    rospy.init_node('occupancygridmap', anonymous=True)
    OccupancyGridMap()
    rospy.sleep(0.1)
    rospy.spin()

if __name__=='__main__':
	main(sys.argv)
