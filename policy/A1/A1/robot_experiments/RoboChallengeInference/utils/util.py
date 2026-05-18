import threading
import functools
import time
from datetime import datetime
from utils.enums import ReturnCode
import requests

def timeout(seconds):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = [Exception(f"Function '{func.__name__}' timed out after {seconds} seconds.")]
            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    result[0] = e
            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(seconds)
            if thread.is_alive():
                print(f"Function '{func.__name__}' timed out after {seconds} seconds.")
                return ReturnCode.TIMEOUT
            if isinstance(result[0], Exception):
                print("res", result[0])
                return ReturnCode.EXCEPTION

            return result[0]
        return wrapper
    return decorator

class RobotController:
    @timeout(5)
    def wait_for_robot_running(self, poll_interval=2):
        while True:
            time.sleep(poll_interval)
            print('now: ', datetime.now())
            raise Exception(f"Function '{__name__}' timed out after {poll_interval} seconds.")

def retry_request(retries=3, delay=1):
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(retries):
                try:
                    return func(*args, **kwargs)
                except requests.exceptions.RequestException as e:
                    last_exception = e
                    if attempt < retries - 1:
                        time.sleep(delay)
            raise last_exception
        return wrapper
    return decorator


if __name__ == '__main__':
    robot = RobotController()
    res = robot.wait_for_robot_running()
    print(res)

