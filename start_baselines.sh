cd /home/Lxr/Fed/src/baselines

rm -rf /home/Lxr/Fed/src/baselines/logs/*
rm -rf /home/Lxr/Fed/src/baselines/nohup.out

nohup /home/Lxr/.conda/envs/multi_fed/bin/python /home/Lxr/Fed/src/baselines/main.py &

echo "训练已启动，查看训练日志请使用命令：tail -f /home/Lxr/Fed/src/baselines/nohup.out， 具体日志文件请查看/home/Lxr/Fed/src/baselines/logs"

cd -