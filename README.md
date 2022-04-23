
# model_hub

This is where i save model used in face-recognition task

1. Setup environments:

```powershell
conda create --name model_env python=3.6
conda activate model_env
pip install torch==1.8.0+cpu torchvision==0.9.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

2. Run:

To convert pytorch model into onnx, run:

```powershell
conda activate model_env
python main.py
```

To test time running on each onnx model, run:

```powershell
conda activate model_env
python time_check.py
```

change config in **config.py** base on what you want to test

## [![Repography logo](https://images.repography.com/logo.svg)](https://repography.com) Recent activity [![Time period](https://images.repography.com/25022152/Tamminhdiep97/model_hub/recent-activity/a6384a256c757af24021651c01cbd485_badge.svg)](https://repography.com)

[![Timeline graph](https://images.repography.com/25022152/Tamminhdiep97/model_hub/recent-activity/a6384a256c757af24021651c01cbd485_timeline.svg)](https://github.com/Tamminhdiep97/model_hub/commits)
[![Issue status graph](https://images.repography.com/25022152/Tamminhdiep97/model_hub/recent-activity/a6384a256c757af24021651c01cbd485_issues.svg)](https://github.com/Tamminhdiep97/model_hub/issues)
[![Pull request status graph](https://images.repography.com/25022152/Tamminhdiep97/model_hub/recent-activity/a6384a256c757af24021651c01cbd485_prs.svg)](https://github.com/Tamminhdiep97/model_hub/pulls)
[![Trending topics](https://images.repography.com/25022152/Tamminhdiep97/model_hub/recent-activity/a6384a256c757af24021651c01cbd485_words.svg)](https://github.com/Tamminhdiep97/model_hub/commits)
[![Top contributors](https://images.repography.com/25022152/Tamminhdiep97/model_hub/recent-activity/a6384a256c757af24021651c01cbd485_users.svg)](https://github.com/Tamminhdiep97/model_hub/graphs/contributors)

### Reference

https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html

https://onnxruntime.ai/

https://github.com/ZhaoJ9014/face.evoLVe
