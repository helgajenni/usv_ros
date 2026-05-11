#!/usr/bin/env python3
import rospy
import tf
import math
import traceback
import numpy as np
from geometry_msgs.msg import Pose2D, Point
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Float64MultiArray, Int32
from nav_msgs.msg import Path

class RVizVisualizer:
    def __init__(self):
        rospy.init_node('rviz_visualizer_node')

        try:
            # Map parameters (sesuai MATLAB mapSize=[33,50])
            map_size = rospy.get_param('/map/size', [33, 50])
            self.y_max = float(map_size[0])   # 33 m
            self.x_max = float(map_size[1])   # 50 m

            self.obs = rospy.get_param('/map/obstacles', [])
            self.start    = rospy.get_param('/mission/start',    [1.0,  8.0])
            self.waypoint = rospy.get_param('/mission/waypoint', [25.0, 20.0])
            self.goal     = rospy.get_param('/mission/goal',     [48.0, 13.0])

            self.safety_margin  = rospy.get_param('/safety_margin',  2.0)
            self.display_margin = rospy.get_param('/display_margin', 0.5)

            usv_p = rospy.get_param('/usv', {})
            self.Ls = usv_p.get('L', 1.6)
            self.Bs = usv_p.get('B', 0.4)

            self.trail_points = []
            self.seg1_len = 0

            # Publishers
            self.pub_static = rospy.Publisher('/usv/static_markers', MarkerArray, queue_size=1, latch=True)
            self.pub_boat  = rospy.Publisher('/usv/boat_marker',  Marker, queue_size=10)
            self.pub_trail = rospy.Publisher('/usv/trail_marker', Marker, queue_size=10)
            self.pub_path  = rospy.Publisher('/usv/planned_path_marker', Marker, queue_size=1, latch=True)

            # Subscribers
            rospy.Subscriber('/usv/pose',          Pose2D, self.pose_cb)
            rospy.Subscriber('/usv/wp_junction_idx', Int32, self.junc_cb)
            rospy.Subscriber('/usv/planned_path',  Path, self.path_cb)

            self.tf_broadcaster = tf.TransformBroadcaster()
            self.publish_static_environment()
        except Exception as e:
            rospy.logerr(f"Error di __init__: {e}")
            rospy.logerr(traceback.format_exc())

    def junc_cb(self, msg):
        self.seg1_len = msg.data

    def path_cb(self, msg):
        try:
            m_path = Marker()
            m_path.header.frame_id = "map"
            m_path.header.stamp = rospy.Time.now()
            m_path.ns = "g2cbs_path"
            m_path.id = 0
            m_path.type = Marker.LINE_STRIP
            m_path.action = Marker.ADD
            m_path.pose.orientation.w = 1.0
            m_path.scale.x = 0.15 
            
            m_path.color.r = 0.2
            m_path.color.g = 0.7
            m_path.color.b = 0.2
            m_path.color.a = 1.0

            for pose_stamped in msg.poses:
                p = Point()
                p.x = pose_stamped.pose.position.x
                p.y = pose_stamped.pose.position.y
                p.z = 0.0
                m_path.points.append(p)

            self.pub_path.publish(m_path)
        except Exception as e:
            rospy.logerr(f"Error di path_cb: {e}")
            rospy.logerr(traceback.format_exc())

    def publish_static_environment(self):
        try:
            marker_array = MarkerArray()

            # 0. Map boundary box
            m_border = Marker()
            m_border.header.frame_id = "map"
            m_border.ns  = "map_boundary"
            m_border.id  = 0
            m_border.type   = Marker.LINE_STRIP
            m_border.action = Marker.ADD
            m_border.pose.orientation.w = 1.0
            m_border.scale.x = 0.15
            m_border.color.r = 0.7; m_border.color.g = 0.7; m_border.color.b = 0.7; m_border.color.a = 0.8
            for cx, cy in [(0, 0), (self.x_max, 0), (self.x_max, self.y_max), (0, self.y_max), (0, 0)]:
                p = Point(); p.x = cx; p.y = cy; p.z = 0.0
                m_border.points.append(p)
            marker_array.markers.append(m_border)

            # 1. Obstacles
            obs_colors = [[1.0, 0.3, 0.3], [1.0, 0.5, 0.2], [1.0, 0.7, 0.2], [0.8, 0.3, 1.0], [0.3, 0.8, 1.0], [0.3, 1.0, 0.5]]
            for i, o in enumerate(self.obs):
                oc = obs_colors[i % len(obs_colors)]
                ox, oy, orad = float(o[0]), float(o[1]), float(o[2])

                m_safe = Marker()
                m_safe.header.frame_id = "map"
                m_safe.ns = "obstacle_safety"; m_safe.id = i
                m_safe.type = Marker.CYLINDER; m_safe.action = Marker.ADD
                m_safe.pose.position.x = ox; m_safe.pose.position.y = oy; m_safe.pose.position.z = -0.02
                m_safe.pose.orientation.w = 1.0
                r_safe = orad + self.safety_margin
                m_safe.scale.x = r_safe * 2; m_safe.scale.y = r_safe * 2; m_safe.scale.z = 0.02
                m_safe.color.r, m_safe.color.g, m_safe.color.b = oc[0], oc[1], oc[2]
                m_safe.color.a = 0.08
                marker_array.markers.append(m_safe)

                m_disp = Marker()
                m_disp.header.frame_id = "map"
                m_disp.ns = "obstacle_display"; m_disp.id = i
                m_disp.type = Marker.CYLINDER; m_disp.action = Marker.ADD
                m_disp.pose.position.x = ox; m_disp.pose.position.y = oy; m_disp.pose.position.z = -0.01
                m_disp.pose.orientation.w = 1.0
                r_disp = orad + self.display_margin
                m_disp.scale.x = r_disp * 2; m_disp.scale.y = r_disp * 2; m_disp.scale.z = 0.02
                m_disp.color.r, m_disp.color.g, m_disp.color.b = oc[0], oc[1], oc[2]
                m_disp.color.a = 0.22
                marker_array.markers.append(m_disp)

                m_obs = Marker()
                m_obs.header.frame_id = "map"
                m_obs.ns = "obstacles"; m_obs.id = i
                m_obs.type = Marker.CYLINDER; m_obs.action = Marker.ADD
                m_obs.pose.position.x = ox; m_obs.pose.position.y = oy; m_obs.pose.position.z = 0.0
                m_obs.pose.orientation.w = 1.0
                m_obs.scale.x = orad * 2; m_obs.scale.y = orad * 2; m_obs.scale.z = 0.4
                m_obs.color.r, m_obs.color.g, m_obs.color.b = oc[0], oc[1], oc[2]
                m_obs.color.a = 0.85
                marker_array.markers.append(m_obs)

                m_lbl = Marker()
                m_lbl.header.frame_id = "map"
                m_lbl.ns = "obstacle_labels"; m_lbl.id = i
                m_lbl.type = Marker.TEXT_VIEW_FACING; m_lbl.action = Marker.ADD
                m_lbl.pose.position.x = ox; m_lbl.pose.position.y = oy; m_lbl.pose.position.z = 0.6
                m_lbl.pose.orientation.w = 1.0
                m_lbl.scale.z = 0.6
                m_lbl.color.r = 1.0; m_lbl.color.g = 1.0; m_lbl.color.b = 1.0; m_lbl.color.a = 1.0
                m_lbl.text = "O%d" % (i + 1)
                marker_array.markers.append(m_lbl)

            pts    = [self.start,       self.waypoint,    self.goal]
            colors = [[0.0, 1.0, 0.0], [0.2, 0.4, 1.0], [1.0, 0.1, 0.1]]
            labels = ["Start",         "WP",              "Goal"]
            for i, (pt, col, lbl) in enumerate(zip(pts, colors, labels)):
                m = Marker()
                m.header.frame_id = "map"
                m.ns = "mission_points"; m.id = i
                m.type = Marker.SPHERE; m.action = Marker.ADD
                m.pose.position.x = float(pt[0]); m.pose.position.y = float(pt[1]); m.pose.position.z = 0.0
                m.pose.orientation.w = 1.0
                m.scale.x = m.scale.y = m.scale.z = 0.9
                m.color.r, m.color.g, m.color.b = col[0], col[1], col[2]
                m.color.a = 1.0
                marker_array.markers.append(m)

                mt = Marker()
                mt.header.frame_id = "map"
                mt.ns = "mission_labels"; mt.id = i
                mt.type = Marker.TEXT_VIEW_FACING; mt.action = Marker.ADD
                mt.pose.position.x = float(pt[0]); mt.pose.position.y = float(pt[1]) + 1.2; mt.pose.position.z = 0.6
                mt.pose.orientation.w = 1.0
                mt.scale.z = 0.8
                mt.color.r, mt.color.g, mt.color.b = col[0], col[1], col[2]
                mt.color.a = 1.0
                mt.text = lbl
                marker_array.markers.append(mt)

            self.pub_static.publish(marker_array)
        except Exception as e:
            rospy.logerr(f"Error di publish_static_environment: {e}")
            rospy.logerr(traceback.format_exc())

    def get_boat_marker(self):
        m = Marker()
        m.header.frame_id = "usv_base_link"
        m.ns = "usv_shape"; m.id = 0
        m.type = Marker.LINE_STRIP; m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = 0.12
        m.color.r = 0.2; m.color.g = 0.6; m.color.b = 1.0; m.color.a = 1.0

        hull_x = [ 0.50*self.Ls,  0.28*self.Ls, -0.50*self.Ls, -0.50*self.Ls,  0.28*self.Ls,  0.50*self.Ls]
        hull_y = [ 0.0,           0.50*self.Bs,  0.40*self.Bs, -0.40*self.Bs, -0.50*self.Bs,  0.0]
        for x, y in zip(hull_x, hull_y):
            p = Point(); p.x, p.y, p.z = x, y, 0.0
            m.points.append(p)
        return m

    def get_trail_marker(self):
        m = Marker()
        m.header.frame_id = "map"
        m.ns = "usv_trail"; m.id = 0
        m.type = Marker.LINE_STRIP; m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = 0.08
        m.color.r = 0.4; m.color.g = 0.85; m.color.b = 1.0; m.color.a = 0.9
        for pt in self.trail_points:
            p = Point(); p.x, p.y, p.z = pt[0], pt[1], 0.0
            m.points.append(p)
        return m

    def pose_cb(self, msg):
        try:
            now = rospy.Time.now()
            quaternion = tf.transformations.quaternion_from_euler(0, 0, msg.theta)
            self.tf_broadcaster.sendTransform(
                (msg.x, msg.y, 0), quaternion, now,
                "usv_base_link", "map"
            )
            
            boat_marker = self.get_boat_marker()
            boat_marker.header.stamp = now
            self.pub_boat.publish(boat_marker)

            # Reset trail jika posisi melompat jauh (simulasi baru dimulai)
            if self.trail_points and math.hypot(msg.x - self.trail_points[-1][0],
                                                msg.y - self.trail_points[-1][1]) > 5.0:
                self.trail_points = []
            self.trail_points.append([msg.x, msg.y])
            if len(self.trail_points) > 1:
                trail_marker = self.get_trail_marker()
                trail_marker.header.stamp = now
                self.pub_trail.publish(trail_marker)
        except Exception as e:
            rospy.logerr(f"Error di pose_cb: {e}")
            rospy.logerr(traceback.format_exc())

if __name__ == '__main__':
    node = RVizVisualizer()
    rospy.spin()