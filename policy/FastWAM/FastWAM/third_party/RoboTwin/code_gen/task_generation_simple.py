import sys
import os
import json

# Add project root directory to system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpt_agent import *
from prompt import *
from task_info import *
from test_gen_code import *
import argparse

def generate_code_once(task_info):
    # Extract task information
    task_name = task_info['task_name']
    task_description = task_info['task_description']
    current_code = task_info['current_code']
    
    # Get the enriched actor_list
    original_actor_list = task_info['actor_list']
    actor_list = enrich_actors(original_actor_list)

    available_env_function = str(AVAILABLE_ENV_FUNCTION)
    function_example = str(FUNCTION_EXAMPLE)

    # Construct prompt
    prompt = (
        f"{BASIC_INFO}\n\n"
        f"# Task Description:\n{task_description}\n\n"
        f"# Actor List:\n{actor_list}\n\n"
        f"# Available API:\n{available_env_function}\n\n"
        f"# Function Example:\n{function_example}\n\n"
        f"# Current Code:\n{current_code}"
    )

    message = [
        {"role": "system", "content": "You need to generate relevant code for some robot tasks in a robot simulation environment based on the provided API."},
        {"role": "user", "content": prompt}
    ]

    # Generate code from model
    res = generate(message, gpt="deepseek", temperature=0)

    # Extract the relevant portion of the generated code
    res = f'''
from envs._base_task import Base_Task
from envs.{task_name} import {task_name}
from envs.utils import *
import sapien

class gpt_{task_name}({task_name}):
    ''' + res[res.find('def play_once'):res.rfind("```")]

    # Save to file
    file_name = f"envs_gen/gpt_{task_name}.py"
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, 'w') as f:
        f.write(res)

    return res


def main(task_info):
    print("Generating code once for task:", task_info['task_name'])
    code = generate_code_once(task_info)

    print("Generated code saved. Testing...")

    task, args = setup_task_config(task_info['task_name'])

    try:
        success_rate, error_message, error_count, run_records = run(task, args)
        print(f"Success Rate: {success_rate}")
        print("Run Records:", run_records)
    except Exception as e:
        import traceback
        print("Error during run:")
        print(traceback.format_exc())
        success_rate, error_message, error_count, run_records = 0, str(e), 1, None

    return code, success_rate, error_message, error_count, run_records




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
python code_gen/task_generation_simple.py task_name
"""
