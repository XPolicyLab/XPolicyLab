# RoboProbe 动作原语
调用 call_tool(id="primitive-control", arguments=...)。arm 是 left 或 right。
- move_tcp: pose=[x,y,z,qw,qx,qy,qz]，世界坐标米，RoboProbe 抓取点。
- move_ee: 同样的 pose 格式，目标是法兰。不要重复加抓取点偏移。
- shift_ee: delta=[dx,dy,dz]，世界坐标法兰平移，保持朝向。
- rotate_ee: axis="x"/"y"/"z"、degrees、frame="world" 或 "tool"，保持法兰位置。
- grip: value 在 0..1，0 闭合，1 打开；用新 RGB 验证夹持。
- wait: steps 为正整数，所有任务省略时统一为 1，每一步是官方原生动作步。只在确实需要推进物理时间时使用。
只有 wait 可指定 steps；移动、夹爪的动作长度由 RoboProbe 按机器人运动约束生成。没有 motion_stage、endpoint_feedback 或 precision 控制模式。工具回复是计划，新观测才显示实际执行效果。未指定的另一臂和夹爪保持原目标。

普通绝对移动和开合爪优先使用 robot-action-timing-tool 的 move/grip；本接口保留相对平移、法兰旋转和原生 wait。它们同样服从任务接触要求与首要执行经验。
