CUDA_VISIBLE_DEVICES=2 vllm serve <your model path> \
    --served-model-name <your model name> \
    --port 8001 \
    --tensor-parallel-size 1 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.9
