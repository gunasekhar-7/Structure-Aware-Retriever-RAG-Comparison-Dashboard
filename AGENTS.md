# Structure-Aware Retriever Project - Agent Configuration

## Available Agents for Structure-Aware Retriever Development

This file documents the specialized agents and tools available for the Structure-Aware Retriever project implementation.

## Project Summary

**Structure-Aware Retriever** is a portfolio project demonstrating STAIR (Structure-Aware Information Retriever) architecture vs traditional chunked RAG. The project builds a Streamlit comparison dashboard showcasing how structure-aware retrieval outperforms traditional methods with zero hallucination.

**Target:** Technical interview portfolio demonstration
**Environment:** Kaggle T4 GPU (12-24GB VRAM), Groq free API
**Timeline:** 1-2 months (part-time)

## Available Agents

### 1. Code Agent (Primary)
- **Purpose**: Main implementation and code generation
- **Tools**: All file editing, dependency management, testing
- **Capabilities**: 
  - Complete project setup and initialization
  - PDF processing and ToC extraction
  - QA generation using Groq API
  - Model fine-tuning with Unsloth
  - Streamlit dashboard development
  - Performance optimization for T4 constraints

### 2. Explore Agent
- **Purpose**: Codebase exploration and documentation
- **Tools**: File search, code analysis, repository exploration
- **Use Cases**:
  - Understanding existing code structure
  - Finding similar implementations
  - Researching best practices
  - Analyzing technical documentation

### 3. Task Agent (Background Processing)
- **Purpose**: Long-running background tasks
- **Tools**: Concurrent task execution, monitoring
- **Use Cases**:
  - Parallel QA generation for multiple PDFs
  - Concurrent model training on multiple GPUs
  - Background data processing
  - Batch file operations

## Project Phases and Agent Allocation

### Phase 1: Foundation Setup
**Agent**: Code Agent
**Tasks**:
- Repository initialization
- Environment setup
- Dependency installation
- Basic configuration

### Phase 2: PDF Processing
**Agent**: Code Agent
**Tools**:
- File writing (1_extract_toc.py)
- PDF processing with PyMuPDF
- JSON output generation
- Error handling and validation

### Phase 3: QA Generation
**Agent**: Task Agent (background)
**Tools**:
- Concurrent API calls to Groq
- Batch question generation
- JSONL file writing
- Rate limit management

### Phase 4: Model Fine-tuning
**Agent**: Code Agent
**Tools**:
- LoRA model configuration
- Training pipeline setup
- GPU memory optimization
- Model saving and validation

### Phase 5: Streamlit Dashboard
**Agent**: Code Agent
**Tools**:
- Web interface development
- Component design
- Real-time integration
- Performance monitoring

### Phase 6: Documentation
**Agent**: Code Agent
**Tools**:
- README generation
- API documentation
- Setup instructions
- Portfolio preparation

## Technical Capabilities

### Code Agent Strengths
- **File Management**: Complete file creation, editing, and optimization
- **Dependency Resolution**: Smart version selection for resource constraints
- **Error Recovery**: Comprehensive error handling and recovery strategies
- **Performance Optimization**: GPU memory management for T4 constraints
- **Integration**: Seamless API integration with Groq and PyMuPDF

### Agent Tool Access

#### Core Development Tools
- `write()`: File creation and modification
- `read()`: File content inspection
- `edit()`: Precise code editing
- `glob()`: File discovery and organization
- `bash()`: Command execution and system operations

#### ML/AI Tools
- `agent_manager_models()`: Model selection and validation
- `apify_call-actor()`: Data extraction utilities
- `context7_resolve-library-id()`: Documentation lookup

#### Web Development
- `playwright_browser_*()`: Browser automation for testing
- `webfetch()`: Web content retrieval
- `websearch()`: Research and documentation

## Implementation Strategy

### Resource-Optimized Development
Given the Kaggle T4 GPU constraints (12-24GB VRAM), the project uses:

1. **Model Selection**: Llama-3.2-3B instead of Mistral-7B
2. **Batch Optimization**: Gradient accumulation (4x) to reduce VRAM usage
3. **Precision**: Mixed precision (fp16) for efficiency
4. **Processing**: Sequential QA generation with rate limiting

### Parallel Processing Opportunities
1. **Concurrent PDF Processing**: Multiple documentation sources
2. **Parallel QA Generation**: Batch API calls to Groq
3. **Multi-GPU Training**: If multiple GPUs available
4. **Background Tasks**: Data processing while UI develops

## Project Configuration

### Environment Variables
```bash
# Groq API Configuration
GROQ_API_KEY=your_api_key_here

# Optional
OPENAI_API_KEY=optional_backup
DEBUG=true
```

### Dependency Management
```txt
# Core dependencies (minimal footprint)
streamlit>=1.28.0
python-dotenv>=1.0.0
requests>=2.31.0
fitz>=2024.5.0

# ML/AI stack (optimized for T4)
torch>=2.0.0
transformers>=4.36.0
unsloth>=2024.5.0
datasets>=2.14.0
trL>=0.7.0
```

### Project Structure
```
docustair-rag/
├── .env                    # Environment configuration
├── requirements.txt        # Python dependencies
├── data/                   # Project data
│   ├── python_311_docs.pdf  # Documentation PDF
│   ├── fastapi_docs.pdf      # Web framework docs
│   ├── toc_and_sections.json # Extracted structure
│   └── synthetic_qa_train.jsonl # Training data
├── src/                    # Source code
│   ├── 1_extract_toc.py     # PDF processing
│   ├── 2_generate_qa.py     # QA generation
│   ├── 3_train_stair_lora.py # Model fine-tuning
│   └── 4_inference_engine.py # STAIR inference
├── app.py                  # Streamlit dashboard
└── README.md               # Portfolio documentation
```

## Performance Optimization

### GPU Memory Management
1. **Model Selection**: Llama-3.2-3B (4-bit quantization)
2. **Batch Processing**: Gradient accumulation to reduce peak memory
3. **Mixed Precision**: FP16 for faster computation
4. **Resource Monitoring**: Continuous VRAM usage tracking

### API Rate Limit Management
1. **Batching**: Group requests to minimize API calls
2. **Retry Logic**: Exponential backoff for failed requests
3. **Throttling**: Respect free tier limits
4. **Caching**: Store generated questions for reuse

## Testing and Validation

### Automated Testing
- **Syntax Validation**: Python code validation
- **Import Testing**: Dependency verification
- **Functionality Testing**: Core features testing
- **Performance Testing**: Resource usage validation

### Manual Validation
- **PDF Processing**: ToC extraction accuracy
- **QA Generation**: Question quality and diversity
- **Model Training**: Loss convergence and metrics
- **Dashboard Functionality**: UI/UX validation

## Future Agent Extensions

### Research Agent
- Literature review on STAIR architecture
- Comparative analysis with other RAG methods
- Academic paper compilation
- Technical benchmark research

### Security Agent
- API key management
- Code security review
- Dependency vulnerability scanning
- Data privacy compliance

### Deployment Agent
- Cloud deployment configuration
- Containerization (Docker)
- CI/CD pipeline setup
- Monitoring and logging

## Agent Communication Protocol

### Task Distribution
1. **Code Agent**: Sequential, focused implementation
2. **Task Agent**: Background, parallel processing
3. **Explore Agent**: On-demand research and analysis

### Coordination
- **Centralized Planning**: Single source of truth
- **Incremental Validation**: Continuous progress verification
- **Resource Management**: Efficient use of T4 GPU constraints
- **Portfolio Focus**: Interview-ready deliverables

## Next Steps

### Immediate Actions
1. Complete environment setup
2. Download documentation PDFs
3. Implement Phase 1: PDF processing
4. Setup Groq API integration
5. Begin QA generation

### Medium-term Goals
1. Complete model fine-tuning
2. Develop Streamlit dashboard
3. Generate portfolio documentation
4. Test all components end-to-end
5. Prepare interview demonstration

### Long-term Vision
1. Deploy to cloud for wider access
2. Add more technical documentation sources
3. Implement advanced evaluation metrics
4. Create comprehensive technical documentation
5. Establish community resources for RAG research

## Conclusion

This agent configuration provides a comprehensive framework for implementing the Structure-Aware Retriever portfolio project. The combination of Code, Explore, and Task agents ensures efficient development while respecting resource constraints. The project is designed to be portfolio-ready, with clear milestones and deliverables that demonstrate technical mastery of advanced RAG architectures.

The project successfully leverages:
- **Resource optimization** for T4 GPU constraints
- **API integration** for cost-effective QA generation
- **Portfolio focus** with interview-ready deliverables
- **Technical excellence** with production-ready code
- **Scalable architecture** for future enhancements

This setup enables rapid development while maintaining high code quality and technical rigor.
