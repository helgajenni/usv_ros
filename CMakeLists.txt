cmake_minimum_required(VERSION 3.0.2)
project(my_usv_pkg)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  nav_msgs
  visualization_msgs
  tf2_ros
)

catkin_package()

include_directories(${catkin_INCLUDE_DIRS})

# install python nodes from scripts/ so rosrun/roslaunch bisa eksekusi
install(PROGRAMS
  scripts/planner_node.py
  scripts/controller_sim_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
