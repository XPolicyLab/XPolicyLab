#!/bin/bash
# bash run_evaluations.bash PORT_NUMBER

# Create logs directory if it doesn't exist
mkdir -p logs


LOG_FILE="logs/evaluation_results-${1}.log"

# Clear the log file
> $LOG_FILE

# List of environments to evaluate
# envs=(
#     "widowx_spoon_on_towel"
#     "widowx_carrot_on_plate"
#     "widowx_put_eggplant_in_basket"
#     "widowx_put_eggplant_in_sink"
#     "widowx_stack_cube"
#     "widowx_close_drawer"
#     "widowx_open_drawer"
# )

envs=(
    "google_robot_pick_coke_can"
    "google_robot_pick_object"
    "google_robot_move_near"
    "google_robot_open_drawer"
    "google_robot_close_drawer"
    "google_robot_place_in_closed_drawer"
)


echo "Starting evaluations..." | tee -a $LOG_FILE
echo "=================================" | tee -a $LOG_FILE

for env in "${envs[@]}"; do
    echo "Running evaluation for: $env" | tee -a $LOG_FILE
    echo "---------------------------------" | tee -a $LOG_FILE
    
    # Run the evaluation and capture output
    output=$(python eval_simpler.py --env "$env" --groot_port $1 --eval_count 350 --headless --episode_length 300 2>&1)
    exit_code=$?

    # Get the last 2 lines of output
    last_lines=$(echo "$output" | tail -n 2)

    # Log the results
    echo "Environment: $env" >> $LOG_FILE
    echo "$last_lines" >> $LOG_FILE
    echo "Exit code: $exit_code" >> $LOG_FILE
    echo "---------------------------------" >> $LOG_FILE
    echo "" >> $LOG_FILE
    
    # Print to console as well
    echo "Last 2 lines for $env:"
    echo "$last_lines"
    echo "Exit code: $exit_code"
    echo ""
done

echo "All evaluations completed. Results saved to $LOG_FILE" 
