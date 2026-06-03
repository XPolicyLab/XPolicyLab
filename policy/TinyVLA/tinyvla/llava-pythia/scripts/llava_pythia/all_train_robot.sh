#!/bin/bash
#"1B" "410M" "14M" "1_4B" "70M" "2_8B"
# 循环调用另一个脚本
#"14M" "70M" "1B" "160M"
for i in "14M" "70M" "160M" "1B"; do
    echo "Loop iteration $i"
    # 调用另一个脚本并传递参数
    ./scripts/llava_pythia/lora_train_robot.sh "$i"
done
