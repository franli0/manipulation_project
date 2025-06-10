#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>

#include <moveit_msgs/msg/display_robot_state.hpp>
#include <moveit_msgs/msg/display_trajectory.hpp>

#include <chrono>
#include <cmath>
#include <memory>
#include <thread>
#include <vector>

#include "custom_msgs/msg/detected_objects.hpp"

// program variables
static const rclcpp::Logger LOGGER = rclcpp::get_logger("pick_and_place_perception_node");
static const std::string PLANNING_GROUP_ROBOT = "ur_manipulator";
static const std::string PLANNING_GROUP_GRIPPER = "gripper";

class PickAndPlacePerceptionController {
public:
  PickAndPlacePerceptionController(rclcpp::Node::SharedPtr base_node_) : base_node_(base_node_) {
    RCLCPP_INFO(LOGGER, "Initializing Pick and Place Perception Controller...");

    // configure node options
    rclcpp::NodeOptions node_options;
    node_options.automatically_declare_parameters_from_overrides(true);

    // initialize move_group node
    move_group_node_ = rclcpp::Node::make_shared("move_group_perception_node", node_options);
    
    // start move_group node in a new executor thread and spin it
    executor_.add_node(move_group_node_);
    std::thread([this]() { this->executor_.spin(); }).detach();

    // initialize move_group interfaces
    move_group_robot_ = std::make_shared<MoveGroupInterface>(move_group_node_, PLANNING_GROUP_ROBOT);
    
    // Try to initialize gripper
    try {
      move_group_gripper_ = std::make_shared<MoveGroupInterface>(move_group_node_, PLANNING_GROUP_GRIPPER);
      has_gripper_ = true;
      RCLCPP_INFO(LOGGER, "Gripper group initialized successfully");
      
      // get initial state of gripper
      joint_model_group_gripper_ = move_group_gripper_->getCurrentState()->getJointModelGroup(PLANNING_GROUP_GRIPPER);
      current_state_gripper_ = move_group_gripper_->getCurrentState(10);
      current_state_gripper_->copyJointGroupPositions(joint_model_group_gripper_, joint_group_positions_gripper_);
      move_group_gripper_->setStartStateToCurrentState();
      
    } catch (const std::exception& ex) {
      RCLCPP_WARN(LOGGER, "No gripper group found: %s", ex.what());
      has_gripper_ = false;
    }

    // get initial state of robot
    joint_model_group_robot_ = move_group_robot_->getCurrentState()->getJointModelGroup(PLANNING_GROUP_ROBOT);

    // print out basic system information
    RCLCPP_INFO(LOGGER, "Planning Frame: %s", move_group_robot_->getPlanningFrame().c_str());
    RCLCPP_INFO(LOGGER, "End Effector Link: %s", move_group_robot_->getEndEffectorLink().c_str());

    // get current state of robot
    current_state_robot_ = move_group_robot_->getCurrentState(10);
    current_state_robot_->copyJointGroupPositions(joint_model_group_robot_, joint_group_positions_robot_);
    move_group_robot_->setStartStateToCurrentState();

    // Set planning parameters
    move_group_robot_->setPlanningTime(10.0);
    move_group_robot_->setNumPlanningAttempts(5);
    move_group_robot_->setMaxVelocityScalingFactor(0.1);
    move_group_robot_->setMaxAccelerationScalingFactor(0.1);

    // Subscribe to object detection topic
    object_subscription_ = base_node_->create_subscription<custom_msgs::msg::DetectedObjects>(
        "/object_detected", 10,
        std::bind(&PickAndPlacePerceptionController::object_callback, this, std::placeholders::_1));

    // Initialize perception variables
    object_detected_ = false;
    execution_started_ = false;

    RCLCPP_INFO(LOGGER, "Pick and Place Perception Controller Initialized");
    RCLCPP_INFO(LOGGER, "Waiting for object detection...");
  }

  ~PickAndPlacePerceptionController() {
    RCLCPP_INFO(LOGGER, "Pick and Place Perception Controller Terminated");
  }

  void spin() {
    // Keep spinning until object is detected and execution is complete
    while (rclcpp::ok() && (!object_detected_ || !execution_completed_)) {
      rclcpp::spin_some(base_node_);
      
      // If object detected but execution not started, start execution
      if (object_detected_ && !execution_started_) {
        execution_started_ = true;
        RCLCPP_INFO(LOGGER, "Starting Pick and Place sequence with detected object...");
        execute_pick_and_place_with_perception();
        execution_completed_ = true;
      }
      
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  }

private:
  // using shorthand for lengthy class references
  using MoveGroupInterface = moveit::planning_interface::MoveGroupInterface;
  using JointModelGroup = moveit::core::JointModelGroup;
  using RobotStatePtr = moveit::core::RobotStatePtr;
  using Plan = MoveGroupInterface::Plan;
  using Pose = geometry_msgs::msg::Pose;
  using RobotTrajectory = moveit_msgs::msg::RobotTrajectory;

  // declare rclcpp base node class
  rclcpp::Node::SharedPtr base_node_;
  rclcpp::Node::SharedPtr move_group_node_;
  rclcpp::executors::SingleThreadedExecutor executor_;

  // declare move_group_interface variables
  std::shared_ptr<MoveGroupInterface> move_group_robot_;
  std::shared_ptr<MoveGroupInterface> move_group_gripper_;
  bool has_gripper_ = false;

  // declare joint_model_group for robot and gripper
  const JointModelGroup *joint_model_group_robot_;
  const JointModelGroup *joint_model_group_gripper_;

  // declare trajectory planning variables for robot
  std::vector<double> joint_group_positions_robot_;
  RobotStatePtr current_state_robot_;
  Plan robot_trajectory_plan_;
  Pose target_pose_robot_;
  bool plan_success_robot_ = false;

  // declare trajectory planning variables for gripper
  std::vector<double> joint_group_positions_gripper_;
  RobotStatePtr current_state_gripper_;
  Plan gripper_trajectory_plan_;
  bool plan_success_gripper_ = false;

  // declare cartesian trajectory planning variables
  std::vector<Pose> cartesian_waypoints_;
  RobotTrajectory cartesian_trajectory_plan_;
  const double jump_threshold_ = 0.0;
  const double end_effector_step_ = 0.01;
  double plan_fraction_robot_ = 0.0;

  std::vector<double> starting_joint_positions_;

  // Perception-related variables
  rclcpp::Subscription<custom_msgs::msg::DetectedObjects>::SharedPtr object_subscription_;
  bool object_detected_;
  bool execution_started_;
  bool execution_completed_ = false;
  
  // Detected object coordinates
  double detected_x_, detected_y_, detected_z_;
  
  // Distance to approach down (based on detected object)
  const double APPROACH_DISTANCE = 0.065;

  void object_callback(const custom_msgs::msg::DetectedObjects::SharedPtr msg)
  {
    RCLCPP_INFO(LOGGER, "Received object detection:");
    RCLCPP_INFO(LOGGER, "Object ID: %d", msg->object_id);
    RCLCPP_INFO(LOGGER, "Position: [%.3f, %.3f, %.3f]", 
               msg->position.x, msg->position.y, msg->position.z);
    RCLCPP_INFO(LOGGER, "Dimensions: [%.3f x %.3f x %.3f]",
               msg->width, msg->height, msg->thickness);

    // Store the detected object coordinates
    detected_x_ = msg->position.x;
    detected_y_ = msg->position.y;
    detected_z_ = msg->position.z + 0.2;

    object_detected_ = true;
    
    RCLCPP_INFO(LOGGER, "Object coordinates stored for pick and place");
  }

  void execute_pick_and_place_with_perception() {
    RCLCPP_INFO(LOGGER, "Starting Pick and Place Sequence with Perception...");

    // Step 1: 
    RCLCPP_INFO(LOGGER, "Step 1: Storing starting position...");
    starting_position();

    // Step 2: Open Gripper
    RCLCPP_INFO(LOGGER, "Step 2: Opening gripper...");
    open_gripper();

    // Step 3: Go to Pre-grasp Position using detected coordinates
    RCLCPP_INFO(LOGGER, "Step 3: Moving to pre-grasp position...");
    go_to_pregrasp_position_perception();

    // Step 4: Approach Object (Cartesian Path)
    RCLCPP_INFO(LOGGER, "Step 4: Approaching detected object...");
    approach_object();

    // Wait a moment
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    RCLCPP_INFO(LOGGER, "Step 4 Completed!!!");

    // Step 5: Close Gripper
    RCLCPP_INFO(LOGGER, "Step 5: Closing gripper to grasp object...");
    close_gripper();

    // Wait a moment
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));

    // Step 6: Retreat from Object (Cartesian Path)
    RCLCPP_INFO(LOGGER, "Step 6: Lifting object...");
    retreat_from_object();

    // Step 7: Move to Place Position
    RCLCPP_INFO(LOGGER, "Step 7: Moving to place position...");
    go_to_place_position();

    // Step 8: Open Gripper to Release
    RCLCPP_INFO(LOGGER, "Step 8: Releasing object...");
    open_gripper();

    // Step 9: Return to Starting Position
    RCLCPP_INFO(LOGGER, "Step 9: Returning to starting position...");
    return_to_starting_position();

    RCLCPP_INFO(LOGGER, "Pick and Place with Perception completed successfully!");
  }

  void go_to_pregrasp_position_perception() {
    RCLCPP_INFO(LOGGER, "Planning trajectory to detected object pre-grasp position...");
    RCLCPP_INFO(LOGGER, "Target coordinates: [%.3f, %.3f, %.3f]", detected_x_, detected_y_, detected_z_);
    
    // Use pose-based approach with detected coordinates
    setup_goal_pose_target(
        detected_x_ + 0.01, detected_y_ - 0.01, detected_z_,  // Position from perception
        0, 1, 0, 0          // Orientation (gripper pointing down)
    );
    plan_trajectory_kinematics();
    execute_trajectory_kinematics();
  }

  void approach_object() {
    RCLCPP_INFO(LOGGER, "Planning Cartesian approach to detected object...");
    
    // Use Cartesian path to approach detected object
    setup_waypoints_target(0.000, 0.000, -APPROACH_DISTANCE);
    plan_trajectory_cartesian();
    execute_trajectory_cartesian();
  }

  void retreat_from_object() {
    RCLCPP_INFO(LOGGER, "Planning Cartesian retreat from detected object...");

    // Use Cartesian path to retreat
    setup_waypoints_target(0.000, 0.000, +APPROACH_DISTANCE);
    plan_trajectory_cartesian();
    execute_trajectory_cartesian();
  }

  void go_to_place_position() {
    RCLCPP_INFO(LOGGER, "Planning Joint Value Trajectory to Place Position...");
    // Get current joint positions
    current_state_robot_ = move_group_robot_->getCurrentState(10);
    current_state_robot_->copyJointGroupPositions(joint_model_group_robot_, joint_group_positions_robot_);
    
    // Rotate shoulder pan joint by 180 degrees (π radians)
    joint_group_positions_robot_[0] += M_PI;
    
    move_group_robot_->setJointValueTarget(joint_group_positions_robot_);
    plan_trajectory_kinematics();
    execute_trajectory_kinematics();
  }

  void open_gripper() {
    if (!has_gripper_) {
      RCLCPP_WARN(LOGGER, "No gripper available - simulating gripper open");
      std::this_thread::sleep_for(std::chrono::milliseconds(1000));
      return;
    }

    RCLCPP_INFO(LOGGER, "Planning Gripper Open...");
    
    // Try named target first
    try {
      setup_named_pose_gripper("gripper_open");
      plan_trajectory_gripper();
      execute_trajectory_gripper();
      return;
    } catch (const std::exception& ex) {
      RCLCPP_WARN(LOGGER, "Named target 'open' failed: %s", ex.what());
    }

    // Fallback to joint values
    try {
      setup_joint_value_gripper(0.0);  // Open position
      plan_trajectory_gripper();
      execute_trajectory_gripper();
    } catch (const std::exception& ex) {
      RCLCPP_WARN(LOGGER, "Joint value gripper control failed: %s", ex.what());
    }
  }

  void close_gripper() {
    if (!has_gripper_) {
      RCLCPP_WARN(LOGGER, "No gripper available - simulating gripper close");
      std::this_thread::sleep_for(std::chrono::milliseconds(1000));
      return;
    }

    RCLCPP_INFO(LOGGER, "Planning Gripper Close...");
    
    // Gradual closing
    try {
      RCLCPP_INFO(LOGGER, "Closing gripper gradually...");
      
      setup_joint_value_gripper(0.1);
      plan_trajectory_gripper();
      execute_trajectory_gripper();
      std::this_thread::sleep_for(std::chrono::milliseconds(300));

      setup_joint_value_gripper(0.3);
      plan_trajectory_gripper();
      execute_trajectory_gripper();
      std::this_thread::sleep_for(std::chrono::milliseconds(300));

      setup_joint_value_gripper(0.6);
      plan_trajectory_gripper();
      execute_trajectory_gripper();
      std::this_thread::sleep_for(std::chrono::milliseconds(300));

      setup_joint_value_gripper(0.61);
      plan_trajectory_gripper();
      execute_trajectory_gripper();
      std::this_thread::sleep_for(std::chrono::milliseconds(300));

      setup_joint_value_gripper(0.62);
      plan_trajectory_gripper();
      execute_trajectory_gripper();
      std::this_thread::sleep_for(std::chrono::milliseconds(300));

      setup_joint_value_gripper(0.622);
      plan_trajectory_gripper();
      execute_trajectory_gripper();
      
      RCLCPP_INFO(LOGGER, "Gripper closed gently");
      
    } catch (const std::exception& ex) {
      RCLCPP_WARN(LOGGER, "Gradual gripper control failed: %s", ex.what());
    }
  }

  void starting_position() {
    RCLCPP_INFO(LOGGER, "Starting position");

    setup_joint_value_target(0, -1.571, 0, -1.571, 0, 0);
    plan_trajectory_kinematics();
    execute_trajectory_kinematics();

    current_state_robot_ = move_group_robot_->getCurrentState(10);
    current_state_robot_->copyJointGroupPositions(joint_model_group_robot_, starting_joint_positions_);
  
    RCLCPP_INFO(LOGGER, "Starting position: [%.3f, %.3f, %.3f, %.3f, %.3f, %.3f]",
                starting_joint_positions_[0], starting_joint_positions_[1], starting_joint_positions_[2],
                starting_joint_positions_[3], starting_joint_positions_[4], starting_joint_positions_[5]);
  }
  
  void return_to_starting_position() {
    RCLCPP_INFO(LOGGER, "Planning trajectory to starting position...");
  
    // Set target to the stored starting position
    move_group_robot_->setJointValueTarget(starting_joint_positions_);
  
    // Plan and execute
    plan_trajectory_kinematics();
    execute_trajectory_kinematics();
  
    RCLCPP_INFO(LOGGER, "Returned to starting position");
  }

  // Robot trajectory functions (same as your existing code)
  void setup_joint_value_target(float angle0, float angle1, float angle2,
                                float angle3, float angle4, float angle5) {
    joint_group_positions_robot_[0] = angle0; // Shoulder Pan
    joint_group_positions_robot_[1] = angle1; // Shoulder Lift
    joint_group_positions_robot_[2] = angle2; // Elbow
    joint_group_positions_robot_[3] = angle3; // Wrist 1
    joint_group_positions_robot_[4] = angle4; // Wrist 2
    joint_group_positions_robot_[5] = angle5; // Wrist 3
    move_group_robot_->setJointValueTarget(joint_group_positions_robot_);
  }

  void setup_goal_pose_target(float pos_x, float pos_y, float pos_z,
                              float quat_x, float quat_y, float quat_z,
                              float quat_w) {
    target_pose_robot_.position.x = pos_x;
    target_pose_robot_.position.y = pos_y;
    target_pose_robot_.position.z = pos_z;
    target_pose_robot_.orientation.x = quat_x;
    target_pose_robot_.orientation.y = quat_y;
    target_pose_robot_.orientation.z = quat_z;
    target_pose_robot_.orientation.w = quat_w;
    move_group_robot_->setPoseTarget(target_pose_robot_);
  }

  void setup_waypoints_target(float x_delta, float y_delta, float z_delta) {
    // Get current pose and add to waypoints
    target_pose_robot_ = move_group_robot_->getCurrentPose().pose;
    cartesian_waypoints_.push_back(target_pose_robot_);
    
    // Calculate desired pose from delta values
    target_pose_robot_.position.x += x_delta;
    target_pose_robot_.position.y += y_delta;
    target_pose_robot_.position.z += z_delta;
    cartesian_waypoints_.push_back(target_pose_robot_);
  }

  void plan_trajectory_kinematics() {
    plan_success_robot_ = (move_group_robot_->plan(robot_trajectory_plan_) == moveit::core::MoveItErrorCode::SUCCESS);
  }

  void execute_trajectory_kinematics() {
    if (plan_success_robot_) {
      move_group_robot_->execute(robot_trajectory_plan_);
      RCLCPP_INFO(LOGGER, "Robot Kinematics Trajectory Success!");
    } else {
      RCLCPP_ERROR(LOGGER, "Robot Kinematics Trajectory Failed!");
    }
  }

  void plan_trajectory_cartesian() {
    plan_fraction_robot_ = move_group_robot_->computeCartesianPath(
        cartesian_waypoints_, end_effector_step_, jump_threshold_, cartesian_trajectory_plan_);
  }

  void execute_trajectory_cartesian() {
    if (plan_fraction_robot_ >= 0.0) {
      move_group_robot_->execute(cartesian_trajectory_plan_);
      RCLCPP_INFO(LOGGER, "Robot Cartesian Trajectory Success!");
    } else {
      RCLCPP_ERROR(LOGGER, "Robot Cartesian Trajectory Failed!");
    }
    // Clear waypoints for next use
    cartesian_waypoints_.clear();
  }

  // Gripper control functions (same as your existing code)
  void setup_joint_value_gripper(float angle) {
    if (joint_group_positions_gripper_.size() > 2) {
      joint_group_positions_gripper_[2] = angle;
    } else if (!joint_group_positions_gripper_.empty()) {
      joint_group_positions_gripper_[0] = angle;
    }
    move_group_gripper_->setJointValueTarget(joint_group_positions_gripper_);
  }

  void setup_named_pose_gripper(std::string pose_name) {
    move_group_gripper_->setNamedTarget(pose_name);
  }

  void plan_trajectory_gripper() {
    plan_success_gripper_ = (move_group_gripper_->plan(gripper_trajectory_plan_) == moveit::core::MoveItErrorCode::SUCCESS);
  }

  void execute_trajectory_gripper() {
    if (plan_success_gripper_) {
      move_group_gripper_->execute(gripper_trajectory_plan_);
      RCLCPP_INFO(LOGGER, "Gripper Action Success!");
    } else {
      RCLCPP_ERROR(LOGGER, "Gripper Action Failed!");
    }
  }

};

int main(int argc, char **argv) {
  // initialize program node
  rclcpp::init(argc, argv);

  // initialize base_node as shared pointer
  std::shared_ptr<rclcpp::Node> base_node = std::make_shared<rclcpp::Node>("pick_and_place_perception_controller");

  // instantiate class
  PickAndPlacePerceptionController pick_and_place_perception_node(base_node);

  // keep spinning until execution is complete
  pick_and_place_perception_node.spin();

  // shutdown ros2 node
  rclcpp::shutdown();

  return 0;
}