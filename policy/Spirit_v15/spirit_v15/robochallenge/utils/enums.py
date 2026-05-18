# ==============================================================================
# Attribution
# ------------------------------------------------------------------------------
# This file is derived from RoboChallengeInference:
#   https://github.com/RoboChallenge/RoboChallengeInference.git
#
# We vendor this code for RoboChallenge integration.
# Local modifications: none.
# ==============================================================================

from enum import Enum

"""
return code:
0 expect result
1 not expect result
2 timeout
3 raise exception result
"""

class ReturnCode(int, Enum):
    SUCCESS = 0
    FAILURE = 1
    TIMEOUT = 2
    EXCEPTION = 3


if __name__ == '__main__':
    print(ReturnCode.SUCCESS)