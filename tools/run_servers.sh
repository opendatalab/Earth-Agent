#!/bin/bash

CUDA_VISIBLE_DEVICES=0 conda run -n fastapi python -m uvicorn tools.app.sam2_app:app --host 0.0.0.0 --port 16000 &
CUDA_VISIBLE_DEVICES=1 conda run -n RemoteSAM python -m uvicorn tools.app.remotesam_app:app --host 0.0.0.0 --port 16001 2>/dev/null &
CUDA_VISIBLE_DEVICES=2 conda run -n insam python -m uvicorn tools.app.instructsam_app:app --host 0.0.0.0 --port 16002 2>/dev/null &
CUDA_VISIBLE_DEVICES=3 conda run -n insam python -m uvicorn tools.app.remoteclip_app:app --host 0.0.0.0 --port 16003 2>/dev/null &
CUDA_VISIBLE_DEVICES=4 conda run -n RemoteSAM python -m uvicorn tools.app.striprcnn_app:app --host 0.0.0.0 --port 16004 2>/dev/null &
CUDA_VISIBLE_DEVICES=5 conda run -n RemoteSAM python -m uvicorn tools.app.sm3det_app:app --host 0.0.0.0 --port 16005 2>/dev/null &
CUDA_VISIBLE_DEVICES=6 conda run -n insam python -m uvicorn tools.app.changeos_app:app --host 0.0.0.0 --port 16006 2>/dev/null &
CUDA_VISIBLE_DEVICES=7 conda run -n RemoteSAM python -m uvicorn tools.app.mscn_app:app --host 0.0.0.0 --port 16007 2>/dev/null &



conda run -n fastapi python tools/Analysis.py --port 20000 --temp_dir tmp/tmp 2>/dev/null &
conda run -n fastapi python tools/Index.py --port 20001 --temp_dir tmp/tmp 2>/dev/null &
conda run -n fastapi python tools/Inversion.py --port 20002 --temp_dir tmp/tmp 2>/dev/null &
conda run -n fastapi python tools/Perception.py \
  --port 20003 \
  --temp_dir tmp/tmp \
  --sam2_endpoint http://0.0.0.0:16000 \
  --remotesam_endpoint http://0.0.0.0:16001 \
  --instructsam_endpoint http://0.0.0.0:16002 \
  --remoteclip_endpoint http://0.0.0.0:16003 \
  --striprcnn_endpoint http://0.0.0.0:16004 \
  --sm3det_endpoint http://0.0.0.0:16005 \
  --changeos_endpoint http://0.0.0.0:16006 \
  --mscn_endpoint http://0.0.0.0:16007 2>/dev/null &
conda run -n fastapi python tools/Statistics.py --port 20004 --temp_dir tmp/tmp 2>/dev/null &

# wait