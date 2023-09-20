# 电镀强化学习仿真环境
![自动上料演示](./images/demo-auto.jpg)

## 安装
Python 3.9+.
``` python
pip install -r requirements.txt
```
## 配置

- 全局参数在 config/args.yaml
- 生产数据在 config/test01 ~test03 目录

## 运行
### 自动上料demo

``` python
python play_auto.py
```
### 手工上料demo

``` python
python play_manual_control-ma.py
```

### 使用PPO算法训练

``` python
pip install stable-baselines3[extra]

python train.py
```

### 使用强化学习控制任务调度

``` python
python play.py
```
