"""Keep the policy registration bootstrap on an official SDK client restart."""
import os
import sys


def route_native_restarts(bootstrap_arguments):
    original_execute = os.execv

    def execute(executable, arguments):
        if executable == sys.executable and arguments == [sys.executable, *sys.argv]:
            return original_execute(executable, [executable, *bootstrap_arguments])
        return original_execute(executable, arguments)

    os.execv = execute
