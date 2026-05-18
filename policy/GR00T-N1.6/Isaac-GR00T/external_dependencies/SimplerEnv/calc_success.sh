#!/usr/bin/env bash
# Usage: ./calc_success.sh evaluation_results-5556.log

LOGFILE="$1"

# Extract "Final Success rate: X/Y" lines and compute totals (X and Y can be any integers)
grep -oE "Final Success rate: [0-9]+/[0-9]+" "$LOGFILE" \
| awk -F'[: /]' '{
    success+=$5; total+=$6; n++
} END {
    if (n>0) {
        printf "Runs: %d\nTotal Successes: %d\nTotal Trials: %d\nAverage: %.2f%%\n",
               n, success, total, 100*success/total
    } else {
        print "No results found."
    }
}'
