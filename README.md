# Rumor_Detection_Model-RBEM
> Author:Zhutian Lin
> Computer College

# Need
- Numpy
- Pandas
- sklearn
- keras
- matplotlib

# arg
Begin with argv:
```
main.py
- ds_path = sys.argv[1]  ： path of dataset
- ratio = sys.argv[2]  ： ratio of train_size/total_size, from 0.1 to 0.9

data_distribution.py
- ds_path = sys.argv[1]  : path of dataset
- font_path = sys.argv[2]  : font file path in folder 'figure'

```

# Demo
```
python main.py <path of dataset> <ratio of train_size/total_size>
python data_distribution.py <path of dataset> <font file path>
```
