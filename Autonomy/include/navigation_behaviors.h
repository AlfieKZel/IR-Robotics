#include "rclcpp/rclcpp.hpp"
#include "behaviortree_cpp_v3/behavior_tree.h"
#include "rclcpp_action/rclcpp_action.hpp"
#include "nav2_msgs/action/navigate_to_pose.hpp"
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <cstdlib> // For system function
#include "std_msgs/msg/string.hpp"

class GoToPose : public BT::StatefulActionNode
{
public:
  GoToPose(const std::string &name,
           const BT::NodeConfiguration &config,
           rclcpp::Node::SharedPtr node_ptr);

  using NavigateToPose = nav2_msgs::action::NavigateToPose;
  using GoalHandleNav = rclcpp_action::ClientGoalHandle<NavigateToPose>;
  
  // Add a publisher
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr goal_reached_publisher_;

// Add a subscriber
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr goal_reached_subscriber_;

  rclcpp::Node::SharedPtr node_ptr_;
  rclcpp_action::Client<NavigateToPose>::SharedPtr action_client_ptr_;
  bool done_flag_;

  // Method overrides
  BT::NodeStatus onStart() override;
  BT::NodeStatus onRunning() override;
  void onHalted() override{};

  static BT::PortsList providedPorts();

   // Action Client callback
  void nav_to_pose_callback(const GoalHandleNav::WrappedResult &result);

  /* Not in use as audio_player_node.cpp not in use 
  // Play audio function
  void playAudio(const std::string &mp3_file_path);
  */
};
