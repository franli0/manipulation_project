cmake_minimum_required(VERSION 3.8)
project(moveit2_scripts)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(moveit_ros_move_group REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(custom_msgs REQUIRED)

# Create executable
add_executable(pick_and_place src/pick_and_place.cpp)
add_executable(pick_and_place_real src/pick_and_place_real.cpp)
add_executable(pick_and_place_perception src/pick_and_place_perception.cpp)
add_executable(pick_and_place_perception_real src/pick_and_place_perception_real.cpp)

# Add dependencies
ament_target_dependencies(pick_and_place
  rclcpp
  moveit_ros_planning_interface
  moveit_ros_move_group
  geometry_msgs
  tf2_geometry_msgs
)

ament_target_dependencies(pick_and_place_real
  rclcpp
  moveit_ros_planning_interface
  moveit_ros_move_group
  geometry_msgs
  tf2_geometry_msgs
  moveit_msgs
)

ament_target_dependencies(pick_and_place_perception
  rclcpp
  moveit_ros_planning_interface
  moveit_ros_move_group
  geometry_msgs
  tf2_geometry_msgs
  moveit_msgs
  custom_msgs
)

ament_target_dependencies(pick_and_place_perception_real
  rclcpp
  moveit_ros_planning_interface
  moveit_ros_move_group
  geometry_msgs
  tf2_geometry_msgs
  moveit_msgs
  custom_msgs
)

# Install targets
install(TARGETS
  pick_and_place
  pick_and_place_real
  pick_and_place_perception
  pick_and_place_perception_real
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
