# UGMSC-PLOME

This code is the PyTorch reimplementation of [PLOME](https://github.com/liushulinle/PLOME) finetune part, and employed UGMSC framework. The code are modified from repository: [PLOME_finetune_pytorch](https://github.com/Zhouyuhao97/PLOME_finetune_pytorch). 

## Requirements

It is recommended to directly use the environment of UGMSC-ReaLiSe model, please refer to the requirements of the UGMSC-ReaLiSe model.

## Instructions

1. Download the pre-trained model from the PLOME official repository: https://drive.google.com/file/d/1aip_siFdXynxMz6-2iopWvJqr5jtUu3F/view?usp=sharing or https://share.weiyun.com/OREEY0H3

2. Extract the data in this dictionary, then the `datas/pretrained_plome/` directory would look like this:

   ```
   datas/pretrained_plome
   |- bert_config.json
   |- bert_model.ckpt.data-00000-of-00001
   |- bert_model.ckpt.index
   |- bert_model.ckpt.meta
   |- checkpoint
   |- vocab.txt
   ```

3. copy/rename the `bert_config.json`  to `config.json`, so that the code can recognize the config:

   ```bash
   cp datas/pretrained_plome/bert_config.json datas/pretrained_plome/config.json
   ```

4. Transform the Tensorflow weights to PyTorch:

   ```bash
   python tf2torch.py
   ```

   Then there should be a file named `pytorch_model.bin` in the `datas/pretrained_plome/` directory.

5. The command to train the reimplemented PLOME model:

   ```bash
   python main.py --model_type=plome
   ```

   The command to train the UGMSC-PLOME model with 3 corrector layers:

   ```bash
   python main.py --model_type=ug --corrector_layers 3
   ```
