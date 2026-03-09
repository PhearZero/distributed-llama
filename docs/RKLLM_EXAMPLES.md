# Rockchip NPU Examples

This guide provides step-by-step examples for converting models and using them with Rockchip NPU acceleration in Distributed Llama.

## 1. Model Conversion

Distributed Llama uses its own `.m` model format. To use a model from Hugging Face or the original Llama weights, you first need to convert it.

### Converting from Hugging Face

To convert a Hugging Face model (e.g., Llama-3-8B-Instruct), use the `convert-hf.py` script:

```bash
# 1. Install dependencies
pip install torch safetensors transformers

# 2. Run the conversion (e.g., to Q4_0 quantization)
python converter/convert-hf.py /path/to/hf_model_dir q40
```

This will generate a file like `dllama_model_llama-3-8b-instruct_q40.m`.

### Converting from Llama PTH

To convert original Llama `.pth` weights:

```bash
python converter/convert-llama.py /path/to/llama_weights_dir q40
```

---

## 2. Running with Fine-Grained NPU Offloading

Once you have a `.m` model, you can run it with NPU acceleration using the `--rkllm 1` flag. This will offload Matrix Multiplication (MatMul) operations to the Rockchip NPU, while other operations remain on the CPU.

### Library Dependencies

Before building, you must ensure the Rockchip NPU runtime libraries are in the `src/rkllama/lib/` directory:

```bash
mkdir -p src/rkllama/lib/
cp rknn-toolkit2/rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so src/rkllama/lib/
cp rknn-llm/rkllm-runtime/Linux/librkllm_api/aarch64/librkllmrt.so src/rkllama/lib/
```

### Basic Inference

```bash
# Build with RKLLM support
make DLLAMA_RKLLM=1

# Run inference
./dllama inference \
  --model dllama_model_llama-3-8b-instruct_q40.m \
  --tokenizer tokenizer.t \
  --rkllm 1 \
  --prompt "Write a short poem about a robot." \
  --steps 128
```

### Distributed Inference with NPU

You can also use the NPU on worker nodes in a distributed setup:

**On the Worker (Rockchip Device):**
```bash
./dllama worker --port 9990 --rkllm 1
```

**On the Root (e.g., PC or another Rockchip Device):**
```bash
./dllama inference \
  --model model.m \
  --tokenizer tokenizer.t \
  --workers 192.168.1.100:9990 \
  --prompt "Hello from the NPU cluster!"
```

---

## 3. Advanced: Using Specialized Rockchip Tools

The `rkllama` directory contains specialized tools for Rockchip devices that might offer even better performance or specific features using the full `RKLLM` SDK.

### Using the Python API

```python
from rkllama.api import RKLLM

def callback(result, userdata, state):
    if state == 0: # RKLLM_RUN_NORMAL
        print(result.contents.text, end="", flush=True)

# Initialize RKLLM
model = RKLLM(callback, "your_model.rkllm", "rkllama/models")

# Run inference
model.run(0, 0, "What is the capital of France?")
```

### Using the REST API

Distributed Llama includes a specialized REST API for Rockchip devices:

```bash
# Start the server
python rkllama/src/rkllama/main.py

# Load a model via API
curl -X POST http://localhost:8080/load_model \
     -H "Content-Type: application/json" \
     -d '{"model_name": "llama3.rkllm"}'

# Generate text
curl -X POST http://localhost:8080/generate \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Tell me a joke."}'
```

Refer to `rkllama/documentation/api/english.md` for full API details.
