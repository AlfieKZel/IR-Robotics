#include "navigation_behaviors.h"
#include "yaml-cpp/yaml.h"
#include <string>
#include <cstdlib> //For system function
#include <cstdio>

GoToPose::GoToPose(const std::string &name,
                   const BT::NodeConfiguration &config,
                   rclcpp::Node::SharedPtr node_ptr)
    : BT::StatefulActionNode(name, config), node_ptr_(node_ptr)
{
  action_client_ptr_ = rclcpp_action::create_client<NavigateToPose>(node_ptr_, "/navigate_to_pose");
  done_flag_ = false;
}

BT::PortsList GoToPose::providedPorts()
{
  return {BT::InputPort<std::string>("loc")};
}

BT::NodeStatus GoToPose::onStart()
{
  // Get location key from port and read YAML file
  BT::Optional<std::string> loc = getInput<std::string>("loc");
  const std::string location_file = node_ptr_->get_parameter("location_file").as_string();

  YAML::Node locations = YAML::LoadFile(location_file);

  std::vector<float> pose = locations[loc.value()].as<std::vector<float>>();

  // setup action client
  auto send_goal_options = rclcpp_action::Client<NavigateToPose>::SendGoalOptions();
  send_goal_options.result_callback = std::bind(&GoToPose::nav_to_pose_callback, this, std::placeholders::_1);

  // make pose
  auto goal_msg = NavigateToPose::Goal();
  goal_msg.pose.header.frame_id = "map";
  goal_msg.pose.pose.position.x = pose[0];
  goal_msg.pose.pose.position.y = pose[1];

  tf2::Quaternion q;
  q.setRPY(0, 0, pose[2]);
  q.normalize(); // todo: why?
  goal_msg.pose.pose.orientation = tf2::toMsg(q);

  // send pose
  done_flag_ = false;
  action_client_ptr_->async_send_goal(goal_msg, send_goal_options);
  //RCLCPP_INFO(node_ptr_->get_logger(), "Sent Goal to Nav2\n");
  RCLCPP_INFO(node_ptr_->get_logger(), "[%s] Goal reached\n", this->name().c_str());
  return BT::NodeStatus::RUNNING;
}

BT::NodeStatus GoToPose::onRunning()
{
  if (done_flag_)
  {
    RCLCPP_INFO(node_ptr_->get_logger(), "[%s] Goal reached\n", this->name().c_str());
    return BT::NodeStatus::SUCCESS;
  }
  else
  {
    return BT::NodeStatus::RUNNING;
  }
}

void GoToPose::nav_to_pose_callback(const GoalHandleNav::WrappedResult &result)
{
  if (result.result)
  {
    done_flag_ = true;

 /*// Play the MP3 file using mpg123
    std::string mp3_file_path = "/home/ubuntu/audiotest/R2-D2.mp3";
    playAudio(mp3_file_path);
  }*/ 
}
/*
// Implementation of the playAudio method
void GoToPose::playAudio(const std::string &mp3_file_path)
{
    if (!node_ptr_)
    {
        RCLCPP_ERROR(node_ptr_->get_logger(), "Node pointer is null. Cannot play audio.");
        return;
    }
    // SSH into RPi to Play Audio File
    //std::string sshPassword = "SUTD1234";
    //std::string sshCommand = "sshpass -p '" + sshPassword + "' ssh ubuntu@172.20.10.12";
    //std::string command = sshCommand + " 'mpg123 " + mp3_file_path + " &'";

    // Debugging output: Print the command being executed
    RCLCPP_INFO(node_ptr_->get_logger(), "Executing command: %s", command.c_str());

    // Execute the command on the Raspberry Pi via SSH
    /FILE *pipe = popen(command.c_str(), "r");
    if (!pipe)
    {
        RCLCPP_ERROR(node_ptr_->get_logger(), "Failed to open pipe for command execution.");
        return;
    }

    // Note: No need to wait for the command to complete here

    // Close the pipe
    pclose(pipe);
    
    char buffer[128];
    std::string result = "";
    while (!feof(pipe))
    {
        if (fgets(buffer, 128, pipe) != NULL)
        {
            result += buffer;
        }
    }

    int status = pclose(pipe);
    if (WIFEXITED(status) && WEXITSTATUS(status) == 0)
    {
        RCLCPP_INFO(node_ptr_->get_logger(), "Audio playback successful");
    }
    else
    {
        RCLCPP_ERROR(node_ptr_->get_logger(), "Audio playback failed with exit code: %d", WEXITSTATUS(status));
    }
    */
}

