# Dockerfile for Qwen3.8-Flash-Next on NVIDIA DGX Spark (Grace Blackwell GB10 / SM121)
FROM lmsysorg/sglang:latest

# Environment flags for Grace Blackwell SM121
ENV TORCH_CUDA_ARCH_LIST="12.0;12.1"
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID
ENV SGLANG_USE_MODELSCOPE=false
ENV PYTHONUNBUFFERED=1

WORKDIR /sgl-workspace

# Verify dependencies and FlashInfer kernels
RUN python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Arch: {torch.cuda.get_arch_list()}')"

ENTRYPOINT ["python3", "-m", "sglang.launch_server"]
