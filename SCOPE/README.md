# UGMSC-SCOPE

This code is the SCOPE model employed UGMSC framework. The code are modified from repository: [SCOPE](https://github.com/jiahaozhenbang/SCOPE).

## Requirements

- Python 3.8.17
- PyTorch 1.9.1
- cuda 10.2, cudnn 7.6.5

Anaconda is recommended to install PyTorch  :

```bash
conda install pytorch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 cudatoolkit=10.2 -c pytorch
```

Then install other packages:

```bash
pip install -r requirements.txt
```

Fixed version bug:

```bash
pip install setuptools==59.5.0
```

## Instructions

1. Download the pre-trained model from the SCOPE official repository: https://rec.ustc.edu.cn/share/18549500-4936-11ed-bdbb-75a980e00e16 and unzip it (FPT.zip).

2. Get train data:

   - You can follow the official instructions of SCOPE:

     **Note that the random seed in data_process/get_train_data.py is not fixed, you may not reproduce the exact same results.**

     - Download the processed data from the SCOPE official repository: https://rec.ustc.edu.cn/share/b8470c00-4884-11ed-abb5-01b9f59aa971 and unzip it (data.zip).

     - Process data:
     
       ```bash
        python data_process/get_train_data.py \
            --data_path data \
            --output_dir data
       ```
     

   - Or directly download the processed data we used in our experiments to reproduce out results: https://drive.google.com/file/d/1xUmy3HHeavGaN4UVtt-NHZAp19ARcDaH/view?usp=drive_link


3. Make UGMSC-SCOPE model checkpoint:

   ```bash
   cp -r FPT UGFPT
   python tran2ug_weights.py
   ```

4. Modified the `REPO_PATH` parameter in `ug_train.sh` and train the UGMSC-SCOPE model:

   ```bash
   bash ug_train.sh
   ```

5. Modified the `REPO_PATH` parameter in `ug_predict.sh`  and test the trained UGMSC-SCOPE model:

   ```bash
   bash ug_predict.sh
   ```

