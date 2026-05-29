# RoboChallengeInference

## Project Structure

```
- RoboChallengeInference/
    - README.md
    - requirements.txt
    - demo.py
    - test.py  # Main test entry script
    - robot/
        - __init__.py
        - interface_client.py
        - job_worker.py
    - mock_server
        - mock_rc_robot.py
        - mock_robot_server.py
        - mock_settings.py
        - utils.py
    - utils/
        - __init__.py
        - enums.py
        - log.py
        - util.py
```

## User Guide

### 1. Installation

```bash
# Clone the repository and checkout the specified branch
git clone https://github.com/RoboChallenge/RoboChallengeInference.git
cd RoboChallengeInference

# (Recommended) Create and activate a virtual environment to avoid polluting your global Python environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

```

### 2. Checkout & Modification

```bash
# Checkout
git checkout -b my-feature-branch
# Follow the instructions in demo.py to modify parameters and implement your custom inference logic based on DummyPolicy.
```

### 3. Test

```bash
# Open the mock_settings.py file and set the ROBOT_TAG and RECORD_DATA_DIR variables according to your robot and data directory requirements.
# Notes:
#   Only one pair of ROBOT_TAG and RECORD_DATA_DIR should be active at a time.
#   Ensure that the RECORD_DATA_DIR path matches the structure of your data folder.
#   You can find the appropriate ROBOT_TAG in your training data or on our website.
# Start the test service
python3 mock_robot_server.py
# Use test.py for testing; it will automatically invoke the mock interface to help you debug your model
# Replace {your_args} with the actual parameters you want to test, for example: --checkpoint xxx.
python3 test.py {your_args}
```

### 4. Submit

- Log in to RoboChallenge Web
- Submit an evaluation request
- On the "My Submission" page, you can view your submissions. Click "Detail" to see more information about a submission.
- The Run ID displayed on the details page will be required for the evaluation process.

### 5. Execute

- Wait for a notification (on the website or via email) indicating that your task has been assigned.
- Ensure the modified code from the previous steps is actively running during the assigned period.
- After the task is completed, the program will exit normally. If you encounter any issues or exceptions, please feel
  free to contact us.

### 6. Result

Once your task has been executed, you can view the results by visiting the "My Submissions" page on the website.

## Key API Parameter Descriptions

This is the direct interface for the robot.
The base URL is `/api/robot/<id>/direct`. For example, if the robot ID is `1`, the full URL to get the state is
`/api/robot/1/direct/state.pkl`.

### Sync Clock

**Endpoint:** `/clock-sync`  
**Method:** `GET`

#### Request Parameters

None

#### Response Example

```json
{
  "timestamp": 0.0
}
```

#### Response Fields

| Field     | Type  | Description                 |
|-----------|-------|-----------------------------|
| timestamp | float | unix timestamp on the robot |

---

### Get State

**Endpoint:** `/state.pkl`  
**Method:** `GET`

#### Request Parameters

| Parameter   | Type        | Required | Default | Description                                                                                                                                                                                                                                                                                                                                                                                          |
|-------------|-------------|----------|---------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| width       | integer     | No       | 224     | Width of the image                                                                                                                                                                                                                                                                                                                                                                                   |
| height      | integer     | No       | 224     | Height of the image                                                                                                                                                                                                                                                                                                                                                                                  |
| image_type  | list of str | Yes      | None    | Camera positions; can be one or more of `left_hand`, `right_hand`, `high`                                                                                                                                                                                                                                                                                                                            |
| action_type | str         | Yes      | None    | Control mode; must be `joint` or `pos`, and can optionally be concatenated with `left` or `right`. All possible options are `joint`, `pos`, `leftjoint`, `leftpos`, `rightjoint`, `rightpos`. The value should remain consistent during a job. Usually this is consistent with the parameter in Post Action. See the [Robot specific Notes](#robot-specific-notes) section for detailed information. |

Additional notes on camera positions:

For a dual-arm robot, `left_hand` and `right_hand` refer to the cameras mounted on the left and right arms,
respectively. The `high` camera is positioned above the robot, providing a top-down view of the workspace.

For a single-arm robot, `left_hand` always refers to the camera mounted on the arms. `right_hand` is on the opposite
side of the robot, and `high` is on the right side of the robot.

Some single-arm robots may lack cameras on the arm or right side. `left_hand` or `right_hand` are not available for
those robots. See the [Robot specific Notes](#robot-specific-notes) section for detailed information.

#### Response Example

The response is a pickle file containing a dictionary with the following structure:

```python
{
    "state": 'normal',
    "timestamp": 0.0,
    "pending_actions": 10,
    "action": [0.0, 0.0, ..., 0.0],
    "images": {
        "high": b'PNG',
        "left_hand": b'PNG',
        "right_hand": b'PNG'
    }
}
```

#### Response Fields

| Field             | Type          | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
|-------------------|---------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| state             | string        | Robot state. Should be `normal` if the robot is operational. If the value is `fault` or `abnormal`, there is an issue with the robot. If the value is `size_none`, the request parameter `image_type` or `action_type` is missing.                                                                                                                                                                                                                                                                                                                                                                                |
| timestamp         | float         | Unix timestamp on the robot                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| pending_actions   | integer       | Number of pending actions in the queue                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| action            | list of float | Current robot joint or position values. If `action_type` in the request contains `joint`, the joint values will be returned. If it contains `pos`, the tool end positions will be returned. If it contains `left` or `right`, only the values for the left or right arm will be returned. If neither is specified, values for both arms will be returned. For example, if the robot is Aloha with two arm, the list consists with `[joints of left arm, gripper of left arm, joints of right arm, gripper of right arm]`. See the [Robot specific Notes](#robot-specific-notes) section for detailed information. |
| images            | dict          | Dictionary of images. Only includes camera positions specified in the `image_type` request parameter.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| images.high       | bytes         | PNG image bytes, if present                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| images.left_hand  | bytes         | PNG image bytes, if present                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| images.right_hand | bytes         | PNG image bytes, if present                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |

---

### Post Action

**Endpoint:** `/action`  
**Method:** `POST`

#### Request Parameters

| Parameter   | Type | Required | Default | Description                                                                                                                                                                                                                                       |
|-------------|------|----------|---------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| action_type | str  | Yes      | None    | Control mode. All possible options are `joint`, `pos`, `leftjoint`, `leftpos`, `rightjoint`, `rightpos`. The value should remain consistent during a job. See the [Robot specific Notes](#robot-specific-notes) section for detailed information. |

The HTTP body should be a JSON object with the following structure:

```json
{
  "actions": [
    [
      0.0,
      0.0
    ],
    [
      0.0,
      0.0
    ],
    [
      0.0,
      0.0
    ]
  ],
  "duration": 0.0
}
```

| Field    | Type          | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
|----------|---------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| actions  | 2D float list | Target joint or position values. If `action_type` in the request contains `joint`, the target values control the robot joints. If it contains `pos`, the tool end positions will be controlled. If it contains `left` or `right`, only the left or right arm will be controlled. If neither is specified, both arms will be controlled. The shape of the array is (number of actions, target values per action). For example, if you are using ALOHA and `action_type` is `joint`, then the shape of the actions array should be (N, 14): 6 joints and 1 gripper per arm, N is the number of steps your model infers. See the [Robot specific Notes](#robot-specific-notes) section for detailed information.  |
| duration | float         | Duration (second) per action                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |

#### Response Example

```json
{
  "result": "success",
  "message": ""
}

```

#### Response Fields

| Field   | Type   | Description                                                                                                                                                        |
|---------|--------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| result  | string | Result of the request. Only `success` or `error` will be returned.                                                                                                 |
| message | string | Reason for `error` result, if any. possible message: the robot is not running (fault or logging), the action shape is wrong, action queue is full, other exception |

## Robot specific Notes

Different robots have different action shapes and camera placement.

- Aloha
    - Dual-arm robot
    - 7 DOF per arm (6 joints + 1 gripper)
    - Joint control:
        - one arm(left or right): 7 numbers total: `[6 joints, 1 gripper]`
        - two arms: 14 numbers total: `[left 6 joints, left 1 gripper, right 6 joints, right 1 gripper]`
    - Pose control
        - one arm(left or right): 8 numbers total: `[x, y, z, quaternion(xyzw), gripper]`
        - two arms: 16 numbers
          total: `[left x, left y, left z, left quaternion(xyzw), left gripper, right x, right y, right z, right quaternion(xyzw), right gripper]`
    - 3 cameras: mounted on left/right arm, and on the top of the robot

- Arx5
    - Single-arm robot
    - 7 DOF (6 joints + 1 gripper)
    - Joint control: 7 numbers total: `[6 joints, 1 gripper]`
    - Pose control: 7 numbers total: `[x, y, z, roll, pinch, yaw, gripper]`
    - You must always use `left` in the `action_type` parameter, e.g., `leftjoint` or `leftpos`.
    - 3 cameras: mounted on the arm, opposite to the arm, and on the right side of the arm

- Ur5
    - Single-arm robot
    - 7 DOF (6 joints + 1 gripper)
    - Joint control: 7 numbers total: `[6 joints, 1 gripper]`
    - Pose control: 8 numbers total: `[x, y, z, quaternion(xyzw), gripper]`
    - You must always use `left` in the `action_type` parameter, e.g., `leftjoint` or `leftpos`.
    - 2 cameras: mounted on the arm, and opposite to the arm

- Franka
    - Single-arm robot
    - 8 DOF (7 joints + 1 gripper)
    - Joint control: 8 numbers total: `[7 joints, 1 gripper]`
    - Pose control: 8 numbers total: `[x, y, z, quaterion(xyzw), gripper]`
    - You must always use `left` in the `action_type` parameter, e.g., `leftjoint` or `leftpos`.
    - 3 cameras: mounted on the arm, opposite to the arm, and on the right side of the arm


## Contact

For official inquiries or support, you can reach us via:
- **GitHub Issues:** [https://github.com/RoboChallenge/RoboChallengeInference/issues](https://github.com/RoboChallenge/RoboChallengeInference/issues)
- **Reddit:** [https://www.reddit.com/r/RoboChallenge/](https://www.reddit.com/r/RoboChallenge/)
- **Discord:** [https://discord.gg/8pD8QWDv](https://discord.gg/8pD8QWDv)
- **X (Twitter):** [https://x.com/RoboChallengeAI](https://x.com/RoboChallengeAI)
- **HuggingFace:** [https://huggingface.co/RoboChallenge](https://huggingface.co/RoboChallenge)
- **GitHub:** [https://github.com/RoboChallenge](https://github.com/RoboChallenge)
- **Support Email:** [support@robochallenge.ai](mailto:support@robochallenge.ai)
