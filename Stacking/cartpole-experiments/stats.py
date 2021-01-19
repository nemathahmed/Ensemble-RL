import json

with open('dqn_Cartpole_log.json') as f:
    data = json.load(f)

print(data.keys())