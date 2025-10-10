# Ollama Setup Guide for WJP ANALYSER

## 🚀 **Why Ollama + GPT-OSS-20B?**

- **🔒 Privacy**: Your DXF files stay local
- **💰 Cost-effective**: No per-token charges
- **⚡ Lower latency**: No network round-trips
- **🎯 Specialized**: Can fine-tune for manufacturing
- **🔄 Offline capable**: Works without internet

## 📥 **Installation**

### **1. Install Ollama**
```bash
# Windows (PowerShell)
winget install Ollama.Ollama

# Or download from: https://ollama.ai/download
```

### **2. Start Ollama Service**
```bash
# Start Ollama (runs on http://localhost:11434)
ollama serve
```

### **3. Pull GPT-OSS-20B Model**
```bash
# Pull the model (this will download ~4GB)
ollama pull gpt-oss-20b

# Verify installation
ollama list
```

## 🛠 **Usage with WJP ANALYSER**

### **Analyze DXF with Ollama**
```bash
py -m cli.wjdx ollama-analyze data/samples/medallion_sample.dxf --out ollama_output
```

### **Generate Design Suggestions**
```bash
py -m cli.wjdx ollama-design "intricate floral pattern for waterjet cutting" --out design_output
```

### **Custom Model Configuration**
```bash
# Use different model
py -m cli.wjdx ollama-analyze sample.dxf --model llama2 --out output

# Custom Ollama server
py -m cli.wjdx ollama-analyze sample.dxf --base-url http://192.168.1.100:11434 --out output

# Increase timeout for complex analysis
py -m cli.wjdx ollama-analyze sample.dxf --timeout 300 --out output
```

## 🔧 **Available Commands**

```bash
# Show all commands
py -m cli.wjdx --help

# Ollama-specific help
py -m cli.wjdx ollama-analyze --help
py -m cli.wjdx ollama-design --help
```

## 📊 **Model Comparison**

| Model | Size | Speed | Quality | Use Case |
|-------|------|-------|---------|----------|
| GPT-OSS-20B | 4GB | Fast | Good | General manufacturing analysis |
| Llama2-7B | 4GB | Very Fast | Good | Quick analysis |
| Llama2-13B | 7GB | Medium | Better | Detailed analysis |
| CodeLlama | 4GB | Fast | Excellent | Technical/engineering focus |

## 🎯 **Fine-tuning for Manufacturing**

You can fine-tune models for better manufacturing analysis:

```bash
# Create training data from your analysis reports
# Then fine-tune the model for your specific use cases
ollama create my-manufacturing-model -f Modelfile
```

## 🚨 **Troubleshooting**

### **Ollama Not Running**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama
ollama serve
```

### **Model Not Found**
```bash
# List available models
ollama list

# Pull missing model
ollama pull gpt-oss-20b
```

### **Slow Performance**
- Use smaller models for faster responses
- Increase timeout: `--timeout 300`
- Check system resources (RAM/CPU)

## 🔄 **Migration from OpenAI**

If you want to switch from OpenAI to Ollama:

1. **Install Ollama** (see above)
2. **Replace commands**:
   ```bash
   # Old (OpenAI)
   py -m cli.wjdx ai-analyze sample.dxf --out output
   
   # New (Ollama)
   py -m cli.wjdx ollama-analyze sample.dxf --out output
   ```

3. **Benefits**: No API costs, better privacy, offline capability

## 📈 **Performance Tips**

- **RAM**: 8GB+ recommended for GPT-OSS-20B
- **GPU**: Optional but speeds up inference
- **Batch processing**: Analyze multiple files in sequence
- **Caching**: Ollama caches responses automatically
