import numpy as np

class Demo_Policy:
    def __init__(self, deploy_cfg):
        self.deploy_cfg = deploy_cfg
        # Initialize your policy model here according to deploy_cfg
    
    def set_language(self, instruction):
        # Process the instruction if needed
        print("[Model] Received instruction:", instruction)
        pass
    
    def update_obs(self, obs):
        # Update your model's observation here if needed
        print("[Model] Received observation:")
        pass

    def get_action(self):
        # Generate action according to your model and current observation
        # This is a dummy action for demonstration, replace it with your model's action
        action = np.array([0] * 7)  # Example action, replace with actual action generation logic
        print("[Model] Generated action:", action)
        return [action]

    def reset(self):
        # Reset your model's internal state if needed
        print("[Model] Model successfully reset")
        pass