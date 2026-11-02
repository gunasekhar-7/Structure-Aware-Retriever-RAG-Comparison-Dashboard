# Structure-Aware Retriever RAG Comparison Dashboard
# Environment Configuration and Setup Instructions

## Project Overview

Structure-Aware Retriever is a portfolio project demonstrating the STAIR (Structure-Aware Information Retriever) architecture vs traditional chunked RAG. This implementation focuses on building a working comparison dashboard that showcases the technical expertise required for advanced RAG systems.

**Target Environment:** Kaggle T4 GPU (12-24GB VRAM) with free Groq API access
**Timeline:** Part-time implementation (1-2 months)
**Portfolio Goal:** Live Streamlit demo for technical interviews

## Installation and Setup

### Step 1: Environment Setup

#### Option A: Quick Start (Recommended)
```bash
# Clone this repository
cd Structure-Aware-Retriever

# Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

#### Option B: Manual Installation
```bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install streamlit fitz torch transformers unsloth datasets trl requests python-dotenv
```

### Step 2: Download Documentation PDFs

The project requires technical documentation PDFs. For the portfolio, we recommend:

#### Python 3.11 Documentation
```bash
# Download from official Python docs
# https://docs.python.org/3/download.html
# Or use local copy if available
mkdir -p data
cp path/to/python_311_docs.pdf data/
```

#### FastAPI Documentation  
```bash
# Download from official FastAPI docs
# https://fastapi.tiangolo.com/fastapi-people/
mkdir -p data
cp path/to/fastapi_docs.pdf data/
```

### Step 3: Environment Configuration

#### Create .env file
```bash
# Create .env file in project root
touch .env
```

#### Add Groq API Key
```bash
# Get free Groq API key from https://console.groq.com/
echo "GROQ_API_KEY=your_api_key_here" >> .env
```

#### Verify installation
```bash
# Test Python imports
python -c "import streamlit, fitz, torch, transformers, unsloth; print('✅ All dependencies installed')"
```

## Project Execution Workflow

### Phase 1: PDF Processing
```bash
# Run ToC extraction (requires PDFs first)
python src/1_extract_toc.py

# Expected output:
# - data/toc_and_sections.json created
# - Summary of extracted leaf nodes printed
```

### Phase 2: QA Generation
```bash
# Generate synthetic training data
python src/2_generate_qa.py

# Configure in .env:
# GROQ_API_KEY=your_key_here

# Expected output:
# - data/synthetic_qa_train.jsonl created
# - ~1,000 training examples generated
```

### Phase 3: Model Fine-tuning
```bash
# Fine-tune STAIR model (may take several hours)
python src/3_train_stair_lora.py

# Monitor training with:
# tail -f stair_lora_adapter/training.log

# Expected output:
# - stair_lora_adapter/ directory created
# - Fine-tuned model weights saved
```

### Phase 4: Streamlit Dashboard
```bash
# Launch the comparison dashboard
streamlit run app.py

# Access at: http://localhost:8501
```

## Development Workflow

### Initial Setup Commands
```bash
# Create project structure
mkdir -p data src

# Install dependencies
pip install -r requirements.txt

# Test environment
python -c "import streamlit; print('Streamlit version:', streamlit.__version__)"
```

### Continuous Development
```bash
# Monitor file changes
# Watch for PDF updates or script changes
# Re-run processing as needed

# Quick test of STAIR engine
python -c "
from src.inference_engine import STAIRInferenceEngine
engine = STAIRInferenceEngine()
result = engine.predict_leaf('What is Python?')
print('Prediction:', result['predicted_leaf'])
print('Valid:', result['is_valid'])
"
```

## Troubleshooting Guide

### Common Issues

#### 1. GPU Not Available
```
# Check GPU availability
python -c "
import torch
print('CUDA Available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name())
    print('Memory:', torch.cuda.get_device_properties(0).total_memory / (1024**3), 'GB')
else:
    print('Running on CPU - training will be very slow')
"
```

#### 2. Missing Groq API Key
```bash
# Set environment variable
export GROQ_API_KEY=your_api_key_here

# Or add to .env file
echo "GROQ_API_KEY=your_api_key_here" >> .env
```

#### 3. PDF Processing Issues
```bash
# Check if PDFs exist
ls -la data/

# Common solutions:
# 1. Download official PDFs first
# 2. Use proper PDF format (not scanned images)
# 3. Ensure PDFs have table of contents/bookmarks
```

#### 4. Model Loading Issues
```bash
# Check if model exists
ls -la stair_lora_adapter/

# Re-train if needed:
python src/3_train_stair_lora.py
```

### Performance Optimization

#### For Kaggle T4 GPU
```python
# In src/3_train_stair_lora.py, adjust:
# Use Llama-3.2-3B instead of Mistral-7B
# Reduce batch sizes if needed
# Monitor GPU memory usage
```

#### For Local Development
```python
# Adjust settings based on available resources
# Use CPU mode if no GPU available
# Reduce training epochs for faster iteration
```

## Project Structure

```
docustair-rag/
├── .env                    # Environment variables
├── requirements.txt        # Python dependencies
├── data/
│   ├── python_311_docs.pdf    # Technical documentation
│   ├── fastapi_docs.pdf        # Web framework docs
│   ├── toc_and_sections.json   # Extracted structure
│   └── synthetic_qa_train.jsonl # Training data
├── src/
│   ├── 1_extract_toc.py        # PDF processing script
│   ├── 2_generate_qa.py        # QA generation script
│   ├── 3_train_stair_lora.py   # Model fine-tuning
│   └── 4_inference_engine.py   # STAIR inference
├── app.py                    # Streamlit dashboard
└── README.md                 # Project documentation
```

## Alternative Project Names

For portfolio flexibility, consider these alternatives:

1. **TechRAG Navigator** - Emphasizes technical manual navigation
2. **Structure-Aware Retriever** - More descriptive, less brand-specific
3. **DocuGuide** - Focuses on documentation guidance
4. **ManualMate** - Casual, approachable tech assistant
5. **StructRAG** - Concise, technical branding

## Technical Specifications

### Model Configuration
- **Base Model**: Llama-3.2-3B-Instruct (T4-compatible)
- **Fine-tuning Method**: LoRA (r=16, α=32)
- **Training Data**: 1,000+ synthetic QA pairs
- **Max Sequence Length**: 8,192 tokens
- **Batch Size**: 2 with gradient accumulation (4x)

### API Configuration
- **Provider**: Groq
- **Model**: Llama-3.3-70B (versatile)
- **Cost**: ~$0.60 per 1M tokens
- **Rate Limits**: Respects free tier quotas

### Hardware Requirements
- **Minimum**: CPU (very slow, for prototyping)
- **Recommended**: Kaggle T4 GPU (12-24GB VRAM)
- **Optimal**: Local RTX 3090/4090 (24-48GB VRAM)

## Quick Start Summary

```bash
# 1. Setup environment
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# 2. Download documentation PDFs
setup_download_pdfs.py  # or manual download

# 3. Process PDFs
python src/1_extract_toc.py

# 4. Generate QA data
python src/2_generate_qa.py

# 5. Train model (hours)
python src/3_train_stair_lora.py

# 6. Launch dashboard
streamlit run app.py
```

## Final Project State

The project will be complete when:

- ✅ `data/toc_and_sections.json` exists with leaf sections
- ✅ `data/synthetic_qa_train.jsonl` has training examples
- ✅ `stair_lora_adapter/` contains fine-tuned model
- ✅ `app.py` runs Streamlit dashboard
- ✅ README documents all components
- ✅ Portfolio demo ready for technical interviews

## Support and Next Steps

### Need Help?
- Check `src/1_extract_toc.py` for PDF processing issues
- Verify Groq API key in `.env` file
- Monitor training progress with `tail -f stair_lora_adapter/training.log`
- Test inference with the Streamlit dashboard

### Next Development Steps
1. Add custom evaluation metrics
2. Implement active learning for question generation
3. Add support for multiple document types
4. Deploy to cloud platform for interview sharing
5. Create performance benchmarks report

### Project Benefits
- **Technical Showcase**: Advanced RAG architecture demonstration
- **Portfolio Ready**: Complete, working implementation
- **Cost Efficient**: Uses free APIs and GPU resources
- **Time Optimized**: 1-2 month timeline achievable
- **Interview Ready**: Live Streamlit demo for technical discussions
```
