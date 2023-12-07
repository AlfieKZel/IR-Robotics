//Not in use, attempt for publishing/subscribing nodes to play audio file
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/String.hpp"
#include <cstdlib>

class AudioPlayerNode : public rclcpp::Node
{
public:
    AudioPlayerNode() : Node("audio_player_node")
    {
        // Create a subscriber for the /play_audio topic
        audio_sub_ = create_subscription<std_msgs::msg::String>(
            "/play_audio", 10, std::bind(&AudioPlayerNode::audioCallback, this, std::placeholders::_1));
    }

private:
    void audioCallback(const std_msgs::msg::String::SharedPtr msg)
    {
        // Check the message content and play the audio file
        if (msg->data == "play_audio")
        {
            // Use system command or a library to play the audio file
            // Example using system command:
            std::system("mpg123 /home/ubuntu/audiotest/R2-D2.mp3");
        }
    }

    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr audio_sub_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<AudioPlayerNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
