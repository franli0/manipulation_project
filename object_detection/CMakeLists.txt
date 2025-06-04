cmake_minimum_required(VERSION 3.8)
project(object_detection)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/DetectedSurfaces.msg"
  "msg/DetectedObjects.msg"
  DEPENDENCIES geometry_msgs std_msgs
)

ament_export_dependencies(rosidl_default_runtime)
rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")

add_executable(static_transform_publisher_cpp src/static_transform_publisher.cpp)
ament_target_dependencies(static_transform_publisher_cpp rclcpp tf2_ros geometry_msgs)

add_executable(object_detection_cpp src/object_detection.cpp)
target_link_libraries(object_detection_cpp ${cpp_typesupport_target})
ament_target_dependencies(object_detection_cpp rclcpp sensor_msgs geometry_msgs visualization_msgs std_msgs tf2_ros tf2_geometry_msgs)

install(TARGETS static_transform_publisher_cpp object_detection_cpp
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch/ DESTINATION share/${PROJECT_NAME}/launch FILES_MATCHING PATTERN "*.launch.py")
install(DIRECTORY rviz/ DESTINATION share/${PROJECT_NAME}/rviz FILES_MATCHING PATTERN "*.rviz")

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()