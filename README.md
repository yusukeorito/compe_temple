 # Ayniyを使用したコンペティションテンプレート

# Ayniy, All You Need is YAML

Ayniy is a supporting tool for machine learning competitions.

[**Documentation**](https://upura.github.io/ayniy-docs/) | [**Slide (Japanese)**](https://speakerdeck.com/upura/ayniy-with-mlflow)

```python
# Import packages
from sklearn.model_selection import StratifiedKFold
import yaml

from ayniy.model.runner import Runner

# Load configs
f = open('configs/run000.yml', 'r+')
configs = yaml.load(f, Loader=yaml.SafeLoader)

# Difine CV strategy as you like
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)

# Modeling
runner = Runner(configs, cv)
runner.run_train_cv()
runner.run_predict_cv()
runner.submission()
```



## Starter Kit

### Scripts

```bash
mkdir project_dir
cd project_dir
sh start.sh
```

[kaggle_utils](https://github.com/upura/kaggle_utils/tree/update-refactoring) is used for feature engineering.

#### Environment

```bash
docker-compose -d --build
docker exec -it ayniy-test bash
```

#### MLflow

```bash
cd experiments
mlflow ui -h 0.0.0.0
```

### Kaggle Notebook

```bash
!git clone https://github.com/upura/ayniy
import sys
sys.path.append("/kaggle/working/ayniy")
!pip install -r /kaggle/working/ayniy/requirements.txt
!mkdir '../output/'
!mkdir '../output/logs'
from sklearn.model_selection import StratifiedKFold
from ayniy.model.runner import Runner
```

## For Developers

### Test

```bash
# pytest
pytest tests/ --cov=. --cov-report=html
# black
black .
# flake8
flake8 .
# mypy
mypy .
```

### Docs
In container,
```bash
cd docs
make html
```

Out of container,
```bash
sh deploy.sh
```
https://github.com/upura/ayniy-docs
