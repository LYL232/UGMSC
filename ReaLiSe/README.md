# UGMSC-ReaLiSe

This code is the ReaLiSe model employed UGMSC framework. The code are modified from repository: [ReaLiSe](https://github.com/DaDaMrX/ReaLiSe).

## Requirements

- Python 3.6.12
- PyTorch 1.2.0
- cuda 10.0, cudnn 7.6.2

Anaconda is recommended to install PyTorch  :

```
conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch
```

Then install other packages:

```bash
pip install -r requirements.txt
```

## Instructions

1. Download the pre-trained model from the ReaLiSe official repository: https://drive.google.com/drive/folders/14zQ6L6nAumuBqPO3hV3YzWJHHpTSJir2 and put them in the `pretrained` directory:

   ```
   pretrained
   |- pytorch_model.bin
   |- vocab.txt
   |- config.json
   ```

2. Download the processed data from the ReaLiSe official repository: https://drive.google.com/drive/folders/1dC09i57lobL91lEbpebDuUBS0fGz-LAk and put them in the `data` directory:

   ```
   data
   |- trainall.times2.pkl
   |- test.sighan15.pkl
   |- test.sighan15.lbl.tsv
   |- test.sighan14.pkl
   |- test.sighan14.lbl.tsv
   |- test.sighan13.pkl
   |- test.sighan13.lbl.tsv
   ```

3. Train the UGMSC-ReaLiSe model:

   ```bash
   bash train_ug.sh
   ```

4. Test the trained UGMSC-ReaLiSe model, note that you can change the `YEAR` in it to test different SIGHAN test sets.

   ```bash
   bash test_ug.sh
   ```

**Note that we do not guarantee that this code can run the ReaLiSe model properly.** So `train.sh` and `test.sh` may not work properly.
