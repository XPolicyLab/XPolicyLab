export PYTHONPATH=$PYTHONPATH:$PWD/LIBERO

TASK_SUITE=$1
SAVE_NAME=$2
HOST=${3:-127.0.0.1}
PORT=${4:-9000}

python main.py --task_suite_name=${TASK_SUITE} --save_name=${SAVE_NAME} --host=${HOST} --port=${PORT} 