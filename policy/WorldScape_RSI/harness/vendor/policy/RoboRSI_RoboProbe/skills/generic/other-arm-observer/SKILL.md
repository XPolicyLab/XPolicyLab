# 用另一臂的公开视角观察
先检查另一臂是否承担夹持或支撑，不能为了观察破坏任务。
只读调用 call_tool(id="other-arm-observer", arguments={"working_arm":"left"}) 返回右臂官方法兰姿态及相机名称。working_arm="right" 则返回左臂。
需要移动空闲观察臂时，加入 move=true 和 delta=[dx,dy,dz]。这是世界坐标平移米，默认向上移动 0.02 米；默认值不是场景高度或避障保证。先根据当前 RGB 检查完整手臂通道，再选择 delta。
需要旋转时，在平移后的新观测基础上另调 primitive-control rotate_ee。观察工具不接受 degrees 或 motion_stage；不存在自动扫描、隐藏相机、真实夹爪测量。
