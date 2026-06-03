sudo iptables -A OUTPUT -o wlo1 -p udp -j DROP

cd /home/r1lite/galaxea/install/install/startup_config/share/startup_config/script
./robot_startup.sh boot ../sessions.d/ATCStandard/R1LITEBody.d/