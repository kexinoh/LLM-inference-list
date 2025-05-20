# LLM推理引擎对比表

| 名称（含项目链接） | 简介/描述 | 主要特点 | 开源/商业情况 | 支持模型 |
|-------------------|-----------|----------|--------------|----------|
| [vLLM](https://github.com/vllm-project/vllm) | 高吞吐量、多用户优化的LLM推理引擎，专为KV缓存管理与多路复用设计 | 多用户并发、高效批处理和KV缓存、动态批次拼接、支持CUDA/ROCm/CPU | 开源（Apache 2.0） | Llama、GPT、Mistral、Baichuan等主流LLM |
| [Text Generation Inference (TGI)](https://github.com/huggingface/text-generation-inference) | Hugging Face开源高性能分布式推理框架，适用于生产环境 | 多GPU分布式、高吞吐、低延迟、流式输出、内置API | 开源（Apache 2.0），Hugging Face商用服务 | HuggingFace Transformers支持的大多数模型 |
| [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | NVIDIA专为自家GPU优化的高效LLM推理引擎 | NVIDIA硬件极致优化、FP8/INT8量化、高吞吐分布式 | 开源（Apache 2.0），NVIDIA支持 | Llama、GPT、Mistral、Phi等主流LLM |
| [DeepSpeed MII](https://github.com/microsoft/DeepSpeed-MII) | 微软DeepSpeed团队高性能推理，兼容HuggingFace | 深度优化流水线、推理压缩、并行、易集成 | 开源（MIT） | HuggingFace Transformers主流模型 |
| [ONNX Runtime / Olive](https://github.com/microsoft/onnxruntime) | 微软通用推理引擎，Olive为大模型优化子项目 | 多平台（CPU、GPU、NPU）、量化/剪枝、BERT/GPT等 | 开源（MIT） | ONNX格式及部分Transformers模型 |
| [FasterTransformer](https://github.com/NVIDIA/FasterTransformer) | NVIDIA高性能Transformer推理库，C++/CUDA实现 | 多优化算法、多种Transformer变体 | 开源（Apache 2.0） | GPT、BERT、T5、Llama等 |
| [llama.cpp / GGML](https://github.com/ggerganov/llama.cpp) | 高效CPU/GPU/移动端推理，适合本地及边缘侧 | 多平台、量化支持、极低资源需求 | 开源（MIT/Apache 2.0） | Llama, Mistral, Qwen, Baichuan, Phi等 |
| [LightLLM](https://github.com/ModelTC/LightLLM) | 多卡推理优化，专注高效KV Cache管理 | 轻量级、内存占用低、高效多路复用 | 开源（Apache 2.0） | Llama、Qwen、Baichuan等 |
| [TGI4LLM](https://github.com/TGI4LLM/TGI4LLM) | 针对国产模型的推理框架，适配性强 | 中文优化、支持ChatGLM、Qwen等国产LLM | 开源 | ChatGLM, Qwen等国产大模型 |
| [OpenVINO](https://github.com/openvinotoolkit/openvino) | Intel主导，多硬件加速，适合边缘与本地部署 | 支持Intel CPU/GPU/VPU、ONNX兼容 | 开源 | ONNX及部分Transformer模型 |
| [Triton Inference Server](https://github.com/triton-inference-server/server) | NVIDIA统一推理服务，支持多后端与自动扩缩容 | TensorRT/ONNX/PyTorch/TF后端、生产级功能 | 开源（BSD-3） | 主流深度学习框架模型 |
| [LMDeploy](https://github.com/InternLM/lmdeploy) | 商汤开源分布式推理，适合多GPU/多节点 | 高效多卡推理、TensorRT/ONNX支持 | 开源 | Llama、Qwen、Baichuan等 |
| OpenAI API | 官方商业推理API，支持OpenAI自家模型 | 商业闭源、稳定、易用 | 商业闭源 | GPT-3/4系列 |
| 百度文心一言、阿里云PAI等 | 国内云厂商大模型推理服务 | 云端部署、API服务、国产大模型支持 | 商业闭源 | 文心一言、Qwen、ChatGLM等 |
| [DeepSparse](https://github.com/neuralmagic/deepsparse) | CPU推理加速，适合低资源场景 | 极致CPU优化、低资源消耗 | 开源 | ONNX格式为主 |
| [MLX](https://github.com/ml-explore/mlx) | Apple平台专用推理库 | 针对Apple芯片优化 | 开源 | Llama等 |
| [Marlin](https://github.com/kaiokendev/marlin) | 轻量高性能本地推理引擎 | 极致性能、本地部署 | 开源 | Llama等4bit量化模型 |

---

> 参考与补充资料：
> - [LLM推理加速技术全景](https://github.com/Oneflow-Inc/llm_inference_accelerate)
> - [Awesome LLM Serving](https://github.com/HFrost0/awesome-llm-serving)

如需补充，请留言或提交PR。
