import os
import sys
import json

# Add the project root directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpt_agent import *
from prompt import *
from task_info import *
from observation_agent import *
from test_gen_code import *  

import argparse
import os


def generate_code(task_info, las_error=None, observation_feedback=None, message:list=None, generate_num_id=None):
    # Extract task information
    if message is None:
        message = []
        
    # Extract task information
    task_name = task_info['task_name']
    task_description = task_info['task_description']
    current_code = task_info['current_code']
    
    # Get the enriched actor list
    original_actor_list = task_info['actor_list']
    actor_list = enrich_actors(original_actor_list)
    
    available_env_function = str(AVAILABLE_ENV_FUNCTION)
    function_example = str(FUNCTION_EXAMPLE)

    # Generate code
    if las_error is not None:
        # Include multimodal observation feedback
        if observation_feedback:
            Prompt = (
                f"The code is unsuccessful, \n# Last Error Message: \n{las_error}\n\n"
                f"# Visual Observation Feedback: \n{observation_feedback}\n\n"
                f"# Task Description: \n{task_description}\n\n"
                f"# Actor List: \n{actor_list}\n\n"
            )
        else:
            Prompt = (
                f"The code is unsuccessful, \n# Last Error Message: \n{las_error}\n\n"
                f"# Task Description: \n{task_description}\n\n"
                f"# Actor List: \n{actor_list}\n\n"
            )
    else:
        res = f'''
from envs._base_task import Base_Task
from envs.{task_name} import {task_name}
from envs.utils import *
import sapien

class gpt_{task_name}({task_name}):
    def play_once(self):
        pass
        '''
        file_name = f"envs_gen/gpt_{task_name}.py"
        with open(file_name, 'w', encoding='utf-8') as file:
            file.write(res)

        # Construct the full prompt with all required information
        Prompt = (
            f"{BASIC_INFO}\n\n"
            f"# Task Description: \n{task_description}\n\n"
            f"# Actor List: \n{actor_list}\n\n"
            f"# Available API: \n{available_env_function}\n\n"
            f"# Function Example: \n{function_example}\n\n"
            f"# Current Code:\n{current_code}"
        )
    message.append({"role": "user", "content": Prompt})

    # Start the generation process
    res = generate(message)
    res = f'''
from envs._base_task import Base_Task
from envs.{task_name} import {task_name}
from envs.utils import *
import sapien

class gpt_{task_name}({task_name}):
    ''' + res[res.find('def play_once'):res.rfind("```")]
    
    # Save the original code for later comparison
    original_code = res
    
    analysis_text = ""  # Initialize analysis text
    
    # Insert observation function regardless of error
    observation_output = insert_observation_points(task_info, res, generate_num_id=generate_num_id)
    print("Observation Output: ", observation_output)
    
    # Extract analysis text (if exists)
    if "# task_step:" in observation_output:
        try:
            step_part = observation_output.split("# task_step:")[1]
            if "# task_code:" in step_part:
                analysis_text = step_part.split("# task_code:")[0].strip()
        except:
            print("Error extracting analysis text")
    
    # Extract the modified code part
    if "# task_code:" in observation_output:
        code_parts = observation_output.split("# task_code:")
        if len(code_parts) > 1:
            code_part = code_parts[1].strip()
            
            # Handle possible markdown code block format
            if "```python" in code_part:
                code_content = code_part.split("```python", 1)[1]
                if "```" in code_content:
                    code_content = code_content.split("```", 1)[0]
                res = code_content.strip()
            elif "```" in code_part:
                code_content = code_part.split("```", 1)[1]
                if "```" in code_content:
                    code_content = code_content.split("```", 1)[0]
                res = code_content.strip()
            else:
                res = code_part

    # Add analysis text as a comment at the end of the code
    if analysis_text:
        formatted_analysis = "\n\n'''\nObservation Point Analysis:\n" + analysis_text + "\n'''\n"
        res = res + formatted_analysis
        
    file_name = f"envs_gen/gpt_{task_name}.py"
    with open(file_name, 'w', encoding='utf-8') as file:
        file.write(res)
    
    print("Task Name: ", task_name)
    print("Task Description: ", task_description)
    
    task, args = setup_task_config(task_name)
    
    try:
        # Update this to match the new return values of run()
        success_rate, error_message, error_count, run_records = run(task, args)
        
        return res, success_rate, error_message, error_count, run_records
    except KeyboardInterrupt:
        print("Testing interrupted by user")
        return res, 0, "Testing interrupted by user", 20, []
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error occurred during testing: {e}\n{error_trace}")
        return res, 0, f"Error occurred during testing: {e}", 20, []


def main(task_info_dic):
    # Keys: "task_name", "task_description", "current_code"
    
    task_info = now_task_info = task_info_dic
    messages=[{"role": "system", "content": "You need to generate relevant code for some robot tasks in a robot simulation environment based on the provided API."}]
    generate_num = 5
    success_threshold = 0.5
    las_error_message = None
    observation_feedback = None
    task_name = task_info['task_name']
    task_description = task_info['task_description']

    # Save the best code and success rate
    best_code = None
    best_success_rate = 0
    best_run_records = None
    
    # Create log file
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = "envs_gen/logs"
    os.makedirs(log_dir, exist_ok=True)
    log_filename = f"{log_dir}/{task_name}_{timestamp}.log"
    
    # Store all trial records
    all_attempts = []
    suc_list = []
    
    # Set the camera image directory path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)  # Get project root directory
    camera_dir = os.path.join(base_dir, "camera_images")
    task_camera_dir = os.path.join(camera_dir, task_name.lower())
    
    # Clear the camera image directory at the start
    def clear_images(directory):
        if os.path.exists(directory):
            print(f"Clearing image directory: {directory}")
            for item in os.listdir(directory):
                item_path = os.path.join(directory, item)
                try:
                    if os.path.isdir(item_path):
                        clear_images(item_path)
                        print(f"Cleaned directory: {item_path} (directory structure retained)")
                    else:
                        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
                        file_ext = os.path.splitext(item_path)[1].lower()
                        
                        if file_ext in image_extensions:
                            os.remove(item_path)
                            print(f"Deleted image: {item_path}")
                        else:
                            print(f"Skipped non-image file: {item_path}")
                except Exception as e:
                    print(f"Error processing item {item_path}: {e}")
    
    clear_images(task_camera_dir)
    
    for id in range(generate_num):
        print("Generate code for task: ", task_name, f"({id+1}/{generate_num})")
        
        # Generate code
        res_code, success_rate, las_error_message, error_count, run_records = generate_code(
            now_task_info, 
            las_error_message, 
            observation_feedback, 
            messages,
            generate_num_id=id
        )

        suc_list.append(success_rate)
        
        # Record this attempt
        attempt_record = {
            "attempt_id": id + 1,
            "success_rate": success_rate,
            "error_message": las_error_message,
            "error_count": error_count,
            "code": res_code,
            "run_records": run_records
        }
        all_attempts.append(attempt_record)
        
        # Save the best code
        if success_rate > best_success_rate:
            best_success_rate = success_rate
            best_code = res_code
            best_run_records = run_records
            print(f"New best code found with success rate: {best_success_rate}")
        
        if success_rate >= success_threshold:
            print("Successfully generated code for task: ", task_name)
            break
            
        # Handle failure case
        print(f"Failed to generate code for task: {task_info['task_name']} {id}\nError message: \n{las_error_message}")
        change_info = """The error may be caused by: 
1. pre_dis_axis is not set correctly in the place_actor function; 
2. the functional point is not set correctly in the place_actor function; 
3. The pre_dis or dis is not set correctly in the place_actor function;
4. The constrain is not set correctly in the place_actor function, free or align is not constantly fixed, if the code did not have above error, please try to set the constrain to another value.
5. The code didn't take into account the note given in the example function.
The task can be accomplished only through the existing API and example function, please do not use any other API that is not listed in the available API list and examples.\n"""
        now_task_info["task_description"] = f"{task_description}\nFailed to generate code, error message: {las_error_message}, error count: {str(error_count)}\n" + change_info
        now_task_info["current_code"] = res_code
        
        # Analyze run_records to decide which failure case to observe
        print("Analyzing run records to determine which error to observe...")
        
        # Define error priorities
        error_list = [
            "The code can not run", 
            "The target position of the object is incorrect.",
            "The left arm failed to grasp the object", 
            "The right arm failed to grasp the object", 
            "Plan execution failed",
            "Unknown error occurred during execution"
        ]
        
        observe_index = 0
        highest_priority = len(error_list)
        
        for i, record in enumerate(run_records):
            if record == "success!":
                continue
                
            current_priority = len(error_list)
            for p, error_pattern in enumerate(error_list):
                if error_pattern in record:
                    current_priority = p
                    break
                    
            if current_priority < highest_priority:
                highest_priority = current_priority
                observe_index = i
        
        if highest_priority == len(error_list) and len(run_records) > 0:
            observe_index = 0
            
        print(f"Selected to observe error at index {observe_index}: {run_records[observe_index]}")
    
        # Get multimodal observation feedback
        print(f"Selected observation index observe_index={observe_index}, corresponding error: {run_records[observe_index]}")
        generate_specific_dir = os.path.join(camera_dir, task_name.lower(), f"generate_num_{id}")
        print(f"Looking for images in: {os.path.abspath(generate_specific_dir)}")
        observation_feedback = observe_task_execution(
            episode_id=observe_index,
            task_name=f"{task_name}", 
            task_info={
                "description": task_info["task_description"],
                "goal": "Successfully execute the robot task"
            },
            problematic_code=res_code,
            save_dir=os.path.dirname(generate_specific_dir),
            generate_dir_name=f"generate_num_{id}"
        )
        print("Observation feedback: ", observation_feedback)
        print("Observation feedback collected")

    # Ensure the best code is saved
    if best_code is not None:
        file_name = f"envs_gen/gpt_{task_name}.py"
        print(f"Saving best code with success rate: {best_success_rate}")
        with open(file_name, 'w', encoding='utf-8') as file:
            file.write(best_code)

    # Save log information to file
    with open(log_filename, 'w', encoding='utf-8') as log_file:
        log_data = {
            "task_name": task_name,
            "task_description": task_info['task_description'],
            "best_success_rate": best_success_rate,
            "success_rates": suc_list,
            "best_code": best_code,
            "best_run_records": best_run_records,
            "all_attempts": all_attempts
        }
        json.dump(log_data, log_file, indent=2)
    
    print("Success rate list: ", suc_list)
    print(f"Best success rate: {best_success_rate}")
    print(f"Log saved to: {log_filename}")
    
    return best_success_rate, suc_list, best_code, best_run_records

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('task_name', type=str)
    now_task = None
    
    try:
        task_name = parser.parse_args().task_name.upper()
        exec(f'now_task = {task_name}')
    except Exception as e:
        raise ValueError(f"The task name is wrong: {e}")

    main(now_task)


"""
Usage:
python code_gen/task_generation_mm.py task_name
"""
