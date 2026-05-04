#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float32MultiArray
import numpy as np

"""
USV 3-DOF Model
State: [u, v, r, x, y, psi]
u   = surge speed
v   = sway speed
r   = yaw rate
x,y = posisi
psi = heading (rad)

Input:
T  = thrust (gaya ke depan)
Te = yaw moment (gaya belok)
"""

class USV3DOF:
    def __init__(self):
        # ---- Params USV ----
        self.m = 11.8
        self.Xud = -29.63
        self.Yvd = -40.90
        self.Nrd = -17.69

        # State awal [u, v, r, x, y, psi]
        self.state = np.array([0.0, 0.0, 0.0, 
                               0.0, 0.0, 0.0]) 

        self.dt = 0.05  # 20 Hz

    def update(self, T, Te):
        u, v, r, x, y, psi = self.state

        # Persamaan gerak linear 3 DOF
        du = (T + self.Xud*u) / self.m
        dv = (self.Yvd*v) / self.m
        dr = (Te + self.Nrd*r) / (self.m*0.5)

        # Integrasi kecepatan
        u  += du * self.dt
        v  += dv * self.dt
        r  += dr * self.dt

        # Kinematic (earth-fixed)
        x  += (u*np.cos(psi) - v*np.sin(psi)) * self.dt
        y  += (u*np.sin(psi) + v*np.cos(psi)) * self.dt
        psi += r * self.dt

        # Normalisasi psi
        psi = (psi + np.pi) % (2*np.pi) - np.pi

        self.state = np.array([u, v, r, x, y, psi])
        return self.state


class USVNode:
    def __init__(self):
        rospy.init_node("usv_3dof")

        self.model = USV3DOF()

        # Subscriber: input T & Te
        self.sub = rospy.Subscriber("/usv/control", Float32MultiArray, self.control_callback)

        # Publisher: state [u, v, r, x, y, psi]
        self.pub = rospy.Publisher("/usv/state", Float32MultiArray, queue_size=10)

        self.T = 0.0
        self.Te = 0.0

    def control_callback(self, msg):
        # msg.data = [T, Te]
        try:
            self.T  = msg.data[0]
            self.Te = msg.data[1]
        except:
            rospy.logwarn("Format control harus [T, Te]")

    def run(self):
        rate = rospy.Rate(20)  # 20Hz
        while not rospy.is_shutdown():
            state = self.model.update(self.T, self.Te)

            msg = Float32MultiArray()
            msg.data = state.tolist()
            self.pub.publish(msg)

            rate.sleep()


if __name__ == "__main__":
    try:
        node = USVNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
