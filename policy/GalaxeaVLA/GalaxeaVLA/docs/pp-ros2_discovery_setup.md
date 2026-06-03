# ROS2 Discovery Setup

1. SSH into the robot

   ```bash
   ssh r1lite@10.42.0.<ROBOT_PORT>
   ```

2. Create a new configuration file `super_client_configuration_file.xml`

    ```bash
    vim /home/r1lite/Public/super_client_configuration_file.xml
    ```

2. Write the content as shown in [a template we provide](./supports/pick_up_anything_demo/super_client_configuration_file.xml). Please make sure to fill in `<ROBOT_PORT>` according to your robot's IP address.

3. Add the environment variable `FASTRTPS_DEFAULT_PROFILES_FILE` to your bashrc

    ```bash
    vim ~/.bashrc    
    export FASTRTPS_DEFAULT_PROFILES_FILE=/home/r1lite/Public/super_client_configuration_file.xml
    ```

4. Source your bashrc

    ```bash
    source ~/.bashrc
    ```

5. Restart the ros2 process

    ```bash
    ros2 daemon stop
    ros2 daemon start
    ```