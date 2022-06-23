# Configure class for yaml file
class Configure():

    def __init__(self, cfg) -> None:
        self._cfg = cfg

    def __getattr__(self, key):
        value = self._cfg[key]
        if isinstance(value, dict):
            return Configure(value)
        return value

'''
# Sample Code

import yaml

# read yaml file and parse into dictionary
res = None
with open('./config.yaml') as f:
    res = yaml.safe_load(f)

config = Configure(res)
print(config.common.run_label)
print(config.dataset.name)
print(config.model.model_name)

'''