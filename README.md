# Single-Cell-Type-Classification-with-PFL

# Prepare Datset

```
python datapre.py
```

# Training 
```
python main.py --rounds=50 --num_user=4 --frac=1.0 --local_ep=5 --local_bs=32 --lr=0.001 --model=ClassicNN 
--dataset=pancreas_0 --gpu=0 --all_client --dim1=1136 --dim2=100
    
```

```
# UWB example: change 'X' to 'x' in scDGN.py (for small sub dataset test)
python main.py --rounds=50 --num_user=8 --frac=1.0 --local_ep=2 --local_bs=5 --lr=0.01 --model=ClassicNN --dataset=UWB
--dataset=UWB --gpu=0 --all_client --lamda=1.0 --dim1=32 --dim2=16
```
