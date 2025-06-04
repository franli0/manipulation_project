#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/static_transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2/LinearMath/Quaternion.h>

class StaticTransformPublisher : public rclcpp::Node
{
public:
    StaticTransformPublisher() : Node("static_transform_publisher")
    {
        tf_static_broadcaster_ = std::make_shared<tf2_ros::StaticTransformBroadcaster>(this);
        publishStaticTransform();
        RCLCPP_INFO(this->get_logger(), "Static transform publisher initialized (C++)");
    }

private:
    void publishStaticTransform()
    {
        geometry_msgs::msg::TransformStamped static_transform;
        static_transform.header.stamp = this->get_clock()->now();
        static_transform.header.frame_id = "base_link";
        static_transform.child_frame_id = "wrist_rgbd_camera_depth_optical_frame";
        
        static_transform.transform.translation.x = 0.3;
        static_transform.transform.translation.y = 0.0;
        static_transform.transform.translation.z = 0.4;
        
        tf2::Quaternion quat;
        quat.setRPY(1.5708, 0, 0);
        static_transform.transform.rotation.x = quat.x();
        static_transform.transform.rotation.y = quat.y();
        static_transform.transform.rotation.z = quat.z();
        static_transform.transform.rotation.w = quat.w();
        
        tf_static_broadcaster_->sendTransform(static_transform);
        RCLCPP_INFO(this->get_logger(), "Published static transform");
    }
    
    std::shared_ptr<tf2_ros::StaticTransformBroadcaster> tf_static_broadcaster_;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<StaticTransformPublisher>());
    rclcpp::shutdown();
    return 0;
}