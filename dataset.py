import yaml
import os
import shutil
with open(os.path.join('config.yaml')) as stream:
    config = yaml.safe_load(stream)
data_path = config["dataset"]["data_path"]
test_data_path = config["dataset"]["test_data_path"]
breakpoints = config["dataset"]["breakpoints"]
labels = config["dataset"]["labels"]
for i in range(1, len(breakpoints), 2):
    filename = str(breakpoints[i]) + '.jpg'
    src = data_path + filename
    dest = test_data_path + filename
    newname = labels[int(i/2)] + '.jpg'
    old = dest
    new = test_data_path + newname
    if os.path.isfile(src):
        shutil.move(src, dest)
        os.rename(old, new)
