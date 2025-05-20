# LLM推理引擎详细列表

本列表整理了当前主流的LLM（大语言模型）推理引擎，包括其主要特点、支持的模型、适用场景以及开源/商业情况。内容持续更新，欢迎补充纠正。

---

## 1. vLLM
- **简介**：由 UC Berkeley 等团队开发，专为高吞吐量、多用户场景优化，支持高效的KV缓存管理与多路复用。
- **主要特点**：
  - 支持多用户并发推理
  - 高效的批处理和KV缓存管理
  - 支持动态批次拼接
  - 支持 CUDA、ROCm、CPU 后端
- **开源/商业**：开源（Apache 2.0）
- **支持模型**：Transformer/LLM主流架构（Llama、GPT、Mistral、Baichuan等）
- **项目链接**：[https://github.com/vllm-project/vllm](https://github.com/vllm-project/vllm)

---

## 2. TGI (Text Generation Inference)
- **简介**：Hugging Face 开源的高性能分布式推理框架，适合大规模生产环境部署。
- **主要特点**：
  - 多GPU分布式推理
  - 高吞吐量、低延迟
  - 支持流式输出
  - 内置API服务器，易于部署
- **开源/商业**：开源（Apache 2.0），Hugging Face提供商业服务
- **支持模型**：Transformers支持的绝大多数模型
- **项目链接**：[https://github.com/huggingface/text-generation-inference](https://github.com/huggingface/text-generation-inference)

---

## 3. TensorRT-LLM
- **简介**：NVIDIA推出的专为NVIDIA GPU优化的大模型推理引擎。
- **主要特点**：
  - 针对NVIDIA硬件极致优化
  - 支持高效的FP8/INT8量化
  - 高吞吐大并发
  - 支持分布式并行
- **开源/商业**：开源（Apache 2.0），NVIDIA官方支持
- **支持模型**：Llama、GPT、Mistral、Phi等主流LLM
- **项目链接**：[https://github.com/NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)

---

## 4. DeepSpeed MII
- **简介**：微软DeepSpeed团队开发，专注于高性能、高吞吐的推理。
- **主要特点**：
  - 深度优化的推理流水线
  - 支持推理压缩、并行
  - 易集成，兼容HuggingFace Transformers
- **开源/商业**：开源（MIT）
- **支持模型**：Transformers架构主流模型
- **项目链接**：[https://github.com/microsoft/DeepSpeed-MII](https://github.com/microsoft/DeepSpeed-MII)

---

## 5. ONNX Runtime / Olive
- **简介**：微软主导的通用推理引擎，支持多平台部署。Olive是针对大模型推理优化的子项目。
- **主要特点**：
  - 支持CPU、GPU、NPU等多种后端
  - 丰富的量化/剪枝优化
  - 支持BERT、GPT等架构
- **开源/商业**：开源（MIT）
- **支持模型**：ONNX格式模型、部分HuggingFace模型
- **项目链接**：[https://github.com/microsoft/onnxruntime](https://github.com/microsoft/onnxruntime)

---

## 6. FasterTransformer
- **简介**：NVIDIA开源的高性能Transformer推理库。
- **主要特点**：
  - C++/CUDA实现，极致性能
  - 支持多种优化算法
  - 支持多种Transformer变体
- **开源/商业**：开源（Apache 2.0）
- **支持模型**：GPT、BERT、T5、Llama等
- **项目链接**：[https://github.com/NVIDIA/FasterTransformer](https://github.com/NVIDIA/FasterTransformer)

---

## 7. GGML / llama.cpp
- **简介**：高效的CPU/GPU/移动端推理框架，轻量化，适合边缘侧或本地推理。
- **主要特点**：
  - 多平台支持（x86, ARM, macOS, WASM 等）
  - 量化支持（4bit/5bit/8bit等）
  - 适合本地部署和离线应用
- **开源/商业**：开源（MIT/Apache 2.0）
- **支持模型**：Llama, Mistral, Qwen, Baichuan, Phi 等
- **项目链接**：[https://github.com/ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp)

---

## 8. LightLLM
- **简介**：专为大规模多卡推理优化，重点在于高效KV Cache管理。
- **主要特点**：
  - 轻量级设计，内存占用低
  - 高效多路复用
  - 适合云端与本地混合部署
- **开源/商业**：开源（Apache 2.0）
- **支持模型**：Llama、Qwen、Baichuan等
- **项目链接**：[https://github.com/ModelTC/LightLLM](https://github.com/ModelTC/LightLLM)

---

## 9. TGI4LLM
- **简介**：专为国产LLM适配的推理引擎，兼容主流国产大模型。
- **主要特点**：
  - 针对中文场景优化
  - 支持主流国产LLM（如ChatGLM、Qwen等）
- **开源/商业**：开源
- **支持模型**：ChatGLM, Qwen等
- **项目链接**：[https://github.com/TGI4LLM/TGI4LLM](https://github.com/TGI4LLM/TGI4LLM)

---

## 10. OpenVINO
- **简介**：Intel主导的深度学习推理优化工具，支持多种硬件加速。
- **主要特点**：
  - 支持Intel CPU、GPU和VPU
  - 面向边缘与本地低成本部署
- **开源/商业**：开源
- **支持模型**：ONNX格式、部分Transformer模型
- **项目链接**：[https://github.com/openvinotoolkit/openvino](https://github.com/openvinotoolkit/openvino)

---

## 11. Triton Inference Server
- **简介**：NVIDIA的统一推理服务器，支持多种后端和自动扩缩容。
- **主要特点**：
  - 支持TensorRT、ONNX、PyTorch、TF等多后端
  - 丰富的生产级功能（A/B测试、监控等）
- **开源/商业**：开源（BSD-3）
- **支持模型**：主流框架模型
- **项目链接**：[https://github.com/triton-inference-server/server](https://github.com/triton-inference-server/server)

---

## 12. LMDeploy
- **简介**：商汤开源，专注于高效多GPU/多节点分布式推理。
- **主要特点**：
  - 高效的多卡推理优化
  - 支持TensorRT、ONNX等后端
- **开源/商业**：开源
- **支持模型**：Llama、Qwen、Baichuan等
- **项目链接**：[https://github.com/InternLM/lmdeploy](https://github.com/InternLM/lmdeploy)

---

## 13. Others / 补充
- **OpenAI API**：商业闭源，官方推理服务，支持OpenAI自家模型。
- **百度文心一言服务**、**阿里云PAI-EAS**、**华为昇腾MindSpore Serving** 等国内云厂商大模型推理服务。
- **DeepSparse**：专注于CPU推理加速。
- **MLX**：Apple平台专用推理库。
- **Marlin**：轻量高性能推理引擎，适合本地推理。

---

## 参考资料
- [LLM推理加速技术全景](https://github.com/Oneflow-Inc/llm_inference_accelerate)
- [Awesome LLM Serving](https://github.com/HFrost0/awesome-llm-serving)

---

如有补充，请在本列表后留言或提交PR。
