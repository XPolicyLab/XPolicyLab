# 基于 RoboProbe 的动作生成
调用 call_tool，完整参数示例：
{"id":"robot-action-timing-tool","arguments":{"op":"move","arm":"left","position":[x,y,z]}}
这里 x,y,z 是本轮从公开观测推断的数字。arguments 必须显式带 op；不要写 action，也不要把参数嵌套在 move 对象内。
闭爪示例：{"id":"robot-action-timing-tool","arguments":{"op":"grip","arm":"left","opening":0}}。
- move: arm="left" 或 "right"，position=[x,y,z]，世界坐标米、RoboProbe 抓取点；可加 quaternion_wxyz=[qw,qx,qy,qz]，省略则保持官方观测朝向。
quaternion_wxyz 是世界坐标系下的机器人法兰朝向；法兰局部 +X 轴沿手指伸出方向。RoboProbe 的固定竖直向下参考四元数为 [0.5,-0.5,0.5,0.5]；它是机器人接口约定，不是场景标定。需要朝下接近时可从这个参考选择姿态，绕竖直方向的调整依据当前 RGB。
position 是抓取点而不是法兰原点。view_env_state 的 grasp_points 给出当前抓取点 x/y/z；state 中 *_ee_pose 前三项是法兰位置。保存归位目标时，记录初始 grasp_points 位置和 *_ee_pose 后四项的法兰四元数，不能混用两个位置。
- grip: arm、opening 在 0..1；0 闭合，1 打开。
- grip_both: left 和 right 为各侧夹爪目标，两臂保持。
- move_both: left/right 各为 {"position":[x,y,z],"quaternion_wxyz":[qw,qx,qy,qz]}，朝向可省略。两侧均为抓取点位置；只有两臂整条通道与持物互不干涉且没有顺序依赖时才合并，否则单臂串行。
- visual_align: arm、samples=[{"tcp":[x,y,z],"pixel":[u,v]},...]、gripper_pixel=[u,v]、target_pixel=[u,v]。使用本回合已经执行并观察到的机器人运动样本，拟合像素与 XY 位移，保持当前高度与朝向做一次校正。
visual_align 至少需要三个不共线 XY 样本；若样本包含升降，需要四个独立 XYZ 样本。必须使用同一相机、同一臂、同一跟踪点和朝向，不能用历史标定。max_correction_m 默认 0.06、gain 默认 0.8，都是显式动作参数，不是场景答案或评测阈值。只在当前可见净空允许时使用。
移动目标使用动作后的最新 RGB 像素。模型无法读取原生累计步数，不提供按内部步数外推的 target_track 输入；模型思考的墙钟时间也不能作为仿真位移依据。物体接触或改变运动后重新观测。
图像不清时用 inspect_rgb_region(camera,region) 放大当前 RGB，region 为归一化 [left,top,right,bottom]。返回映射用于把裁剪像素换回原图；不提供深度或相机参数。
移动和夹爪分开执行，然后重新观察。不要传 steps、gripper、motion_stage、duration_s 或 timeout；RoboProbe 根据机器人约束生成原生关节动作。没有实测手指反馈控制。工具回复描述计划而不是已到位；抓取、支撑、释放均用下一次 RGB 和公开机器人姿态核验。
