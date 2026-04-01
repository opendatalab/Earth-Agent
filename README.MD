# Earth-Agent Online Inference Version

---

## 1. Project Overview

In the previous release, the CV models in the toolset were implemented using a lookup table approach to enable evaluation on Earth-Bench in GPU-free environments. This version converts the CV models into deployable inference versions for production deployment.

---

## 2. Model Sources


The CV models used in this project are sourced from the following repositories:

| Model | Repository |
|---|---|
| MSCN | https://github.com/TangXu-Group/Remote-Sensing-Images-Classification/tree/main/MSCN |
| RemoteSAM | https://github.com/1e12Leon/RemoteSAM |
| SM3Det | https://github.com/zcablii/SM3Det |
| Strip-R-CNN | https://github.com/HVision-NKU/Strip-R-CNN?tab=readme-ov-file |
| RemoteCLIP | https://github.com/ChenDelong1999/RemoteCLIP |
| InstructSAM | https://github.com/VoyagerXvoyagerx/InstructSAM |
| ChangeOS | https://github.com/Z-Zheng/ChangeOS |

---

## 3. Model Weights


Please download the `models` folder using the link below and replace the `models` folder in this project:

**Link**: https://pan.baidu.com/s/1DPEHueuGLy9unuPUSqgQPQ?pwd=1234

---

## 4. Configuration Before Running

Before running, modify the following files to suit your environment:

- `run.sh`
- `run_agent.py`
- `agent_config.json`

