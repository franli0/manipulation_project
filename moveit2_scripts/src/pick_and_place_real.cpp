#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>

#include <moveit_msgs/msg/display_robot_state.hpp>
#include <moveit_msgs/msg/display_trajectory.hpp>

#include <chrono>
#include <cmath>
#include <memory>
#include <thread>
#include <vector>

// program variables
static const rclcpp::Logger LOGGER = rclcpp::get_logger("pick_and_place_real_node");
static const std::string PLANNING_GROUP_ROBOT = "ur_manipulator";
static const std::string PLANNING_GROUP_GRIPPER = "gripper";

class PickAndPlaceRealController {
public:
  PickAndPlaceRealController(rclcpp::Node::SharedPtr base_node_) : base_node_(base_node_) {
    RCLCPP_INFO(LOGGER, "Initializing Pick and Place Real Robot Controller...");

    // configure node options
    rclcpp::NodeOptions node_options;
    node_options.automatically_declare_parameters_from_overrides(true);

    // initialize move_group node
    move_group_node_ = rclcpp::Node::make_shared("move_group_real_node", node_options);
    
    // start move_group node in a new executor thread and spin it
    executor_.add_node(move_group_node_);
    std::thread([this]() { this->executor_.spin(); }).detach();

    // initialize move_group interfaces
    move_group_robot_ = std::make_shared<MoveGroupInterface>(move_group_node_, PLANNING_GROUP_ROBOT);
    
    // Try to initialize gripper
    try {
      move_group_gripper_ = std::make_shared<MoveGroupInterface>(move_group_node_, PLANNING_GROUP_GRIPPER);
      has_gripper_ = true;
      RCLCPP_INFO(LOGGER, "Real robot gripper group initialized successfully");
      
      // get initial state of gripper
      joint_model_group_gripper_ = move_group_gripper_->getCurrentState()->getJointModelGroup(PLANNING_GROUP_GRIPPER);
      current_state_gripper_ = move_group_gripper_->getCurrentState(10);
      current_state_gripper_->copyJointGroupPositions(joint_model_group_gripper_, joint_group_positions_gripper_);
      move_group_gripper_->setStartStateToCurrentState();
      
    } catch (const std::exception& ex) {
      RCLCPP_WARN(LOGGER, "No gripper group found on real robot: %s", ex.what());
      has_gripper_ = false;
    }

    // get initial state of robot
    joint_model_group_robot_ = move_group_robot_->getCurrentState()->getJointModelGroup(PLANNING_GROUP_ROBOT);

    // print out basic system information
    RCLCPP_INFO(LOGGER, "Real Robot Planning Frame: %s", move_group_robot_->getPlanningFrame().c_str());
    RCLCPP_INFO(LOGGER, "Real Robot End Effector Link: %s", move_group_robot_->getEndEffectorLink().c_str());

    // get current state of robot
    current_state_robot_ = move_group_robot_->getCurrentState(10);
    current_state_robot_->copyJointGroupPositions(joint_model_group_robot_, joint_group_positions_robot_);
    move_group_robot_->setStartStateToCurrentState();

    // Set planning parameters for real robot (more conservative)
    move_group_robot_->setPlanningTime(15.0);  // More time for real robot
    move_group_robot_->setNumPlanningAttempts(10);  // More attempts
    move_group_robot_->setMaxVelocityScalingFactor(0.05);  // Slower for safety
    move_group_robot_->setMaxAccelerationScalingFactor(0.05);  // Slower acceleration

    RCLCPP_INFO(LOGGER, "Pick and Place Real Robot Controller Initialized");
  }

  ~PickAndPlaceRealController() {
    RCLCPP_INFO(LOGGER, "Pick and Place Real Robot Controller Terminated");
  }

  void execute_pick_and_place() {
    RCLCPP_INFO(LOGGER, "Starting Pick and Place Sequence on Real Robot...");

    RCLCPP_INFO(LOGGER, "Starting position...");
    starting_position();

    // Step 2: Open Gripper
    RCLCPP_INFO(LOGGER, "Opening Gripper...");
    open_gripper();

    // Step 3: Go to Pre-grasp Position (adjusted for real robot)
    RCLCPP_INFO(LOGGER, "Going to Pre-grasp Position...");
    go_to_pregrasp_position();

    // Step 4: Approach Object using real robot coordinates
    RCLCPP_INFO(LOGGER, "Approaching Object on Real Robot...");
    approach_object();

    // Wait a moment for stability
    std::this_thread::sleep_for(std::chrono::milliseconds(3000));

    // Step 5: Close Gripper
    RCLCPP_INFO(LOGGER, "Closing Gripper...");
    close_gripper();

    // Wait a moment to ensure grip
    std::this_thread::sleep_for(std::chrono::milliseconds(3000));

    // Step 6: Retreat from Object
    RCLCPP_INFO(LOGGER, "Retreating from Object...");
    retreat_from_object();

    // Step 7: Move to Place Position (Rotate 180 degrees)
    RCLCPP_INFO(LOGGER, "Moving to Place Position...");
    go_to_place_position();

    // Step 8: Open Gripper to Release
    RCLCPP_INFO(LOGGER, "Releasing Object...");
    open_gripper();

    // Step 9: Return to Starting Position
    RCLCPP_INFO(LOGGER, "Returning to Starting Position...");
    return_to_starting_position();

    RCLCPP_INFO(LOGGER, "Pick and Place Sequence on Real Robot Completed Successfully!");
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
  const double end_effector_step_ = 0.005;  // Smaller steps for real robot
  double plan_fraction_robot_ = 0.0;

  std::vector<double> starting_joint_positions_;

  // Real robot object coordinates (from task requirements)
  const double REAL_OBJECT_X = 0.343;  // Real robot cube X position
  const double REAL_OBJECT_Y = 0.132;  // Real robot cube Y position
  const double REAL_OBJECT_Z = 0.15;   // Pre-grasp height above cube
  const double APPROACH_DISTANCE = 0.02;  // Distance to approach down

  void go_to_pregrasp_position() {
    RCLCPP_INFO(LOGGER, "Planning trajectory to real robot pre-grasp position...");
    
    // Use pose-based approach for real robot with specified coordinates
    setup_joint_value_target(-0.314, -1.257, 1.484, -1.937, -1.466, 1.169);
    plan_trajectory_kinematics();
    execute_trajectory_kinematics();
  }

  void approach_object() {
    RCLCPP_INFO(LOGGER, "Planning approach to real robot object...");

    // Use Cartesian path
    setup_waypoints_target(0.000, 0.000, -APPROACH_DISTANCE);
    plan_trajectory_cartesian();
    execute_trajectory_cartesian();
  }

  void retreat_from_object() {
    RCLCPP_INFO(LOGGER, "Planning retreat from real robot object...");

    // Use Cartesian path
    setup_waypoints_target(0.000, 0.000, +APPROACH_DISTANCE);
    plan_trajectory_cartesian();
    execute_trajectory_cartesian();
    
    setup_joint_value_target(-0.314, -1.257, 1.484, -1.937, -1.466, 1.169);
    plan_trajectory_kinematics();
    execute_trajectory_kinematics();
  }

  void go_to_place_position() {
    RCLCPP_INFO(LOGGER, "Planning trajectory to place position...");
    
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
      RCLCPP_WARN(LOGGER, "No gripper available on real robot - simulating gripper open");
      std::this_thread::sleep_for(std::chrono::milliseconds(2000));
      return;
    }

    RCLCPP_INFO(LOGGER, "Planning Real Robot Gripper Open...");
    
    // Try named target first
    try {
      setup_named_pose_gripper("gripper_open");  // Standard name for real robot
      plan_trajectory_gripper();
      execute_trajectory_gripper();
      return;
    } catch (const std::exception& ex) {
      RCLCPP_WARN(LOGGER, "Named target 'open' failed on real robot: %s", ex.what());
    }

    // Fallback to joint values for real robot
    try {
      setup_joint_value_gripper(0.0);  // Open position
      plan_trajectory_gripper();
      execute_trajectory_gripper();
    } catch (const std::exception& ex) {
      RCLCPP_WARN(LOGGER, "Joint value gripper control failed on real robot: %s", ex.what());
    }
  }

  void close_gripper() {
    if (!has_gripper_) {
      RCLCPP_WARN(LOGGER, "No gripper available on real robot - simulating gripper close");
      std::this_thread::sleep_for(std::chrono::milliseconds(2000));
      return;
    }

    RCLCPP_INFO(LOGGER, "Planning Real Robot Gripper Close...");
    
    // Try named target first
    // try {
    //   setup_named_pose_gripper("gripper_pickup");  // Standard name for real robot
    //   plan_trajectory_gripper();
    //   execute_trajectory_gripper();
    //   return;
    // } catch (const std::exception& ex) {
    //   RCLCPP_WARN(LOGGER, "Named target 'close' failed on real robot: %s", ex.what());
    // }

    // Gradual closing for real robot (more gentle)
    try {
      RCLCPP_INFO(LOGGER, "Closing real robot gripper gradually...");
      
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

    //   setup_joint_value_gripper(0.61);
    //   plan_trajectory_gripper();
    //   execute_trajectory_gripper();
    //   std::this_thread::sleep_for(std::chrono::milliseconds(300));

    //   setup_joint_value_gripper(0.62);
    //   plan_trajectory_gripper();
    //   execute_trajectory_gripper();
    //   std::this_thread::sleep_for(std::chrono::milliseconds(300));

    //   setup_joint_value_gripper(0.63);
    //   plan_trajectory_gripper();
    //   execute_trajectory_gripper();
    //   std::this_thread::sleep_for(std::chrono::milliseconds(300));

    //   setup_joint_value_gripper(0.64);
    //   plan_trajectory_gripper();
    //   execute_trajectory_gripper();
    //   std::this_thread::sleep_for(std::chrono::milliseconds(300));

    //   setup_joint_value_gripper(0.65);
    //   plan_trajectory_gripper();
    //   execute_trajectory_gripper();
    //   std::this_thread::sleep_for(std::chrono::milliseconds(300));
      
    //   setup_joint_value_gripper(0.6525);
    //   plan_trajectory_gripper();
    //   execute_trajectory_gripper();
      
      RCLCPP_INFO(LOGGER, "Real robot gripper closed gently");
      
    } catch (const std::exception& ex) {
      RCLCPP_WARN(LOGGER, "Gradual gripper control failed on real robot: %s", ex.what());
    }
  }

  void starting_position() {
    
    RCLCPP_INFO(LOGGER, "Real robot starting position");

    setup_joint_value_target(0, -1.571, 0, -1.571, 0, 0);
    plan_trajectory_kinematics();
    execute_trajectory_kinematics();

    // Get and store the current joint positions as starting position
    current_state_robot_ = move_group_robot_->getCurrentState(10);
    current_state_robot_->copyJointGroupPositions(joint_model_group_robot_, starting_joint_positions_);
  }
  
  void return_to_starting_position() {
    RCLCPP_INFO(LOGGER, "Planning trajectory to real robot starting position...");
  
    // Set target to the stored starting position
    move_group_robot_->setJointValueTarget(starting_joint_positions_);
  
    // Plan and execute
    plan_trajectory_kinematics();
    execute_trajectory_kinematics();
  
    RCLCPP_INFO(LOGGER, "Real robot returned to starting position");
  }

  // Robot trajectory functions
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
      RCLCPP_INFO(LOGGER, "Real Robot Kinematics Trajectory Success!");
    } else {
      RCLCPP_ERROR(LOGGER, "Real Robot Kinematics Trajectory Failed!");
    }
  }

  void plan_trajectory_cartesian() {
    plan_fraction_robot_ = move_group_robot_->computeCartesianPath(
        cartesian_waypoints_, end_effector_step_, jump_threshold_, cartesian_trajectory_plan_);
  }

  void execute_trajectory_cartesian() {
    if (plan_fraction_robot_ >= 0.0) {
      move_group_robot_->execute(cartesian_trajectory_plan_);
      RCLCPP_INFO(LOGGER, "Real Robot Cartesian Trajectory Success!");
    } else {
      RCLCPP_ERROR(LOGGER, "Real Robot Cartesian Trajectory Failed!");
    }
    // Clear waypoints for next use
    cartesian_waypoints_.clear();
  }

  // Gripper control functions
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
      RCLCPP_INFO(LOGGER, "Real Robot Gripper Action Success!");
    } else {
      RCLCPP_ERROR(LOGGER, "Real Robot Gripper Action Failed!");
    }
  }

}; // class PickAndPlaceRealController

int main(int argc, char **argv) {
  // initialize program node
  rclcpp::init(argc, argv);

  // initialize base_node as shared pointer
  std::shared_ptr<rclcpp::Node> base_node = std::make_shared<rclcpp::Node>("pick_and_place_real_controller");

  // instantiate class
  PickAndPlaceRealController pick_and_place_node(base_node);

  // execute pick and place sequence
  pick_and_place_node.execute_pick_and_place();

  // shutdown ros2 node
  rclcpp::shutdown();

  return 0;
}