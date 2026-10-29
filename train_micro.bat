@echo off
cd /d D:\Code\world_class_model
"C:\Users\Administrator\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe" train.py --config micro --steps 5000 --eval_every 100 --save_every 1000 > train_micro.log 2>&1
