# IR-Robotics

How to run commands:
In your rpi:
ros2 launch turtlebot3_bringup robot.launch.py

CV:
remote PC: 
cd com vis
python3 file.txt

Autonomy:
remote PC:
ros2 launch turtlebot3_navigation2 navigation2.launch.py map:=myMaps/test_map.yaml
ros2 launch tb3_autonomy autonomy.launch.py
