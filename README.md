# Skill Gap Analyst

**🔴 LIVE INTEGRATIONS ONLY - NO MOCKS/OFFLINE MODE**

An AI-powered system that analyzes CVs against job market requirements to identify skill gaps and provide actionable recommendations for career advancement in AI/ML roles.

**⚠️ CRITICAL REQUIREMENTS:**
- **Live internet connection** required for all operations
- **GEMINI_API_KEY** or **MISTRAL_API_KEY** mandatory for LLM analysis and embeddings  
- **TAVILY_API_KEY** mandatory for web search and market intelligence
- **No offline fallbacks** or mock data supported

## 🎯 Project Goal

This tool helps AI professionals and job seekers:
- **Analyze their CV** against market demands for specific roles
- **Identify skill gaps** in technical and soft skills  
- **Get personalized recommendations** for skill development
- **Understand market trends** for AI/ML positions globally

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Input Layer   │    │  Analysis Core  │    │  Output Layer   │
├─────────────────┤    ├─────────────────┤    ├─────────────────┤
│                 │    │                 │    │                 │
│ • CV Upload     │───▶│ • CV Parser     │───▶│ • Markdown      │
│   (.txt/.pdf)   │    │ • Skill Analyst │    │   Report        │
│                 │    │ • Market Intel  │    │ • JSON Data     │
│ • Job Role      │    │ • Report Gen    │    │ • Artifacts     │
│ • Region        │    │                 │    │                 │
│ • Language      │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
          │                      │                      │
          ▼                      ▼                      ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Interface Layer │    │ LangGraph Flow  │    │🌐 LIVE APIs     │
├─────────────────┤    ├─────────────────┤    ├─────────────────┤
│                 │    │                 │    │                 │
│ • CLI (Typer)   │    │ • State Mgmt    │    │ • **Gemini**    │
│ • Web (Streamlit│    │ • Agent Coord   │    │ • **Mistral**   │
│                 │    │ • Error Handle  │    │ • **Tavily**    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+** (required)
- **Virtual environment** (recommended)  
- **Live internet connection** (required)
- **Required API keys:**
  - `GEMINI_API_KEY` or `MISTRAL_API_KEY` - For LLM analysis and embeddings
  - `TAVILY_API_KEY` - For web search and market intelligence

### Installation

1. **Clone and setup environment:**
   ```bash
   git clone <repository-url>
   cd skill-gap-analyst
   
   # Create virtual environment
   python -m venv .venv
   
   # Activate virtual environment
   # Windows:
   .venv\Scripts\activate
   # macOS/Linux:
   source .venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   # Basic installation
   pip install -e .
   
   # With UI support
   pip install -e ".[ui]"
   
   # With development tools
   pip install -e ".[dev]"
   
   # With PDF support (optional)
   pip install -e ".[pdf]"
   
   # Install everything
   pip install -e ".[ui,dev,pdf]"
   ```

3. **Configure environment:**
   ```bash
   # Copy environment template
   cp .env.example .env
   
   # Edit .env with your API keys
   nano .env  # or your preferred editor
   ```

## ⚙️ Environment Configuration

**🔴 REQUIRED API KEYS - NO FALLBACKS**

Create a `.env` file with your API credentials:

```bash
# REQUIRED: LLM Provider Configuration (choose one or both)
GEMINI_API_KEY=your_gemini_api_key_here
MISTRAL_API_KEY=your_mistral_api_key_here
LLM_PROVIDER=auto

# Optional: Model overrides
GEMINI_MODEL=gemini-1.5-flash
GEMINI_EMBED_MODEL=models/embedding-001
MISTRAL_MODEL=mistral-large-latest
MISTRAL_EMBED_MODEL=mistral-embed

# REQUIRED: Tavily Search API Configuration
TAVILY_API_KEY=tvly-your-actual-key-here
SEARCH_PROVIDER=tavily

# LIVE MODE ENFORCEMENT (Default: Disabled)
ALLOW_OFFLINE_EMBEDDINGS=false

# Optional: Advanced Configuration
MAX_SEARCH_RESULTS=8
OUTPUT_FORMAT=markdown
```

**🔑 Example Environment Variables:**
```bash
GEMINI_API_KEY=AIzaSyA1B2C3D4E5F6G7H8I9J0...
MISTRAL_API_KEY=sk-XyZ987654321...
LLM_PROVIDER=auto
TAVILY_API_KEY=tvly-XyZ987654321...
ALLOW_OFFLINE_EMBEDDINGS=false
```

**⚠️ IMPORTANT NOTES:**
- At least one of `GEMINI_API_KEY` or `MISTRAL_API_KEY` is mandatory
- No mock data, stubs, or offline modes available
- Set `ALLOW_OFFLINE_EMBEDDINGS=false` to enforce live mode
- Live internet connection required for all operations

## 🌐 Live Mode, No Mocks

**This system operates in LIVE MODE ONLY with real API integrations.**

### Internet Connection Required
- **Real-time web search** via Tavily API for job market data
- **Live LLM analysis** via Gemini or Mistral APIs for CV parsing and recommendations
- **Dynamic embeddings** for skill similarity analysis
- **No offline fallbacks** - internet connectivity is mandatory

### Required API Keys
At least one LLM API key must be configured for the system to function:

```bash
# REQUIRED: LLM Provider Configuration (choose one or both)
GEMINI_API_KEY=your_gemini_api_key_here
MISTRAL_API_KEY=your_mistral_api_key_here
LLM_PROVIDER=auto

# Optional: Model overrides
GEMINI_MODEL=gemini-1.5-flash
MISTRAL_MODEL=mistral-large-latest

# REQUIRED: Tavily Search API Configuration  
TAVILY_API_KEY=tvly-your-actual-key-here
SEARCH_PROVIDER=tavily

# LIVE MODE ENFORCEMENT
ALLOW_OFFLINE_EMBEDDINGS=false  # Must be false for live mode
```

### Offline Embeddings (Reduced Quality)
By default, offline embeddings are **disabled** to ensure high-quality analysis:

```bash
# Default: High-quality live embeddings (RECOMMENDED)
ALLOW_OFFLINE_EMBEDDINGS=false

# Alternative: Enable offline embeddings (REDUCED QUALITY)
ALLOW_OFFLINE_EMBEDDINGS=true
```

**⚠️ Warning:** Offline embeddings provide significantly reduced quality compared to live embeddings and are only recommended when API quota is temporarily exceeded.

### Troubleshooting API Issues

#### Rate Limiting and Quotas
```bash
# Gemini rate limit errors (429)
Error: "Quota exceeded for quota metric"
Solution: Check quota at https://console.cloud.google.com/iam-admin/quotas

# Mistral rate limit errors (429)
Error: "Rate limit exceeded"
Solution: Check usage at https://console.mistral.ai/usage/

# Tavily rate limit errors  
Error: "API rate limit exceeded"
Solution: Wait for rate limit reset or upgrade plan at https://tavily.com

# Temporary quota exceeded
Error: "insufficient_quota"
Solution: System will attempt retry with exponential backoff
```

#### API Connectivity Issues
```bash
# Network connectivity
Error: "Connection timeout"
Solution: Verify internet connection and firewall settings

# Invalid API keys
Error: "Invalid API key"
Solution: Verify API keys in .env file are correct and active

# API service unavailable
Error: "Service temporarily unavailable" 
Solution: Check API status pages and retry after delay
```

#### Retry Behavior
- **Automatic retries** with exponential backoff for transient failures
- **Maximum 3 attempts** per API call before failing
- **Graceful degradation** to regex extraction when LLM quota exceeded
- **Clear error messages** indicating specific failure reasons

## 💻 CLI Usage

### Basic Analysis
```bash
# Analyze a CV for a specific role
python -m src.app.cli analyze \
  --cv sample/sample_cv.txt \
  --role "Senior AI Engineer" \
  --region "Global" \
  --lang en

# Using the Makefile shortcut
make run_cli
```

### Advanced CLI Options
```bash
# Full command with all options
python -m src.app.cli analyze \
  --cv path/to/your/cv.txt \
  --role "Machine Learning Engineer" \
  --region "United States" \
  --lang en \
  --provider openai \
  --outdir ./output

# Analyze PDF CV
python -m src.app.cli analyze \
  --cv resume.pdf \
  --role "Data Scientist" \
  --region "Europe"

# Different regions and languages
python -m src.app.cli analyze \
  --cv cv.txt \
  --role "AI Research Scientist" \
  --region "Asia Pacific" \
  --lang zh
```

### CLI Parameters
- `--cv`: Path to CV file (.txt or .pdf)
- `--role`: Target job role (e.g., "Senior AI Engineer")
- `--region`: Geographic region ("Global", "US", "Europe", "Asia", etc.)
- `--lang`: Language code (en, zh, es, fr, etc.)
- `--provider`: LLM provider (openai, qwen) - optional, uses .env default
- `--outdir`: Output directory for reports - optional, defaults to current dir

## 🌐 Streamlit Web Interface

### Launch Web App
```bash
# Start the Streamlit app
streamlit run src/app/streamlit_app.py

# Using Makefile
make run_ui

# Custom port/host
streamlit run src/app/streamlit_app.py --server.port 8080
```

### Web Interface Features
- **File Upload**: Drag-and-drop CV files (.txt/.pdf)
- **Interactive Controls**: Sidebar with role, region, and provider selection
- **Real-time Analysis**: Live progress tracking during analysis
- **Rich Output**: Expandable JSON previews and formatted reports
- **Download Options**: Export all artifacts and final reports
- **Responsive Design**: Works on desktop and mobile devices

### Using the Web Interface
1. **Open** `http://localhost:8501` in your browser
2. **Configure** analysis parameters in the sidebar:
   - LLM Provider (OpenAI/Qwen)
   - Target role
   - Geographic region
   - Language preference
3. **Upload** your CV file using the file uploader
4. **Click** "Analyze CV" to start the analysis
5. **Review** results in expandable sections:
   - CV Structure (parsed data)
   - Skill Profile (categorized skills)
   - Market Summary (demand analysis)
   - Final Report (recommendations)
6. **Download** artifacts using the download buttons

## 🤖 Supported Providers

### LLM Providers

#### OpenAI (Default, Recommended)
- **Models**: GPT-4, GPT-3.5-turbo
- **Strengths**: High-quality analysis, excellent instruction following
- **Setup**: Requires OpenAI API key
- **Cost**: Pay-per-use, typically $0.01-0.03 per analysis

#### Qwen (Alternative)
- **Models**: Qwen-turbo, Qwen-plus
- **Strengths**: Multilingual support, competitive pricing
- **Setup**: Requires Alibaba Cloud API key
- **Cost**: Generally lower cost than OpenAI

### Search Providers

#### Tavily (Default, Recommended)
- **Strengths**: AI-optimized search, clean results
- **Rate Limits**: 1000 requests/month free tier
- **Best For**: Job market research, skill trend analysis

#### SerpAPI (Alternative)
- **Strengths**: Google Search integration, comprehensive data
- **Rate Limits**: 100 searches/month free tier
- **Best For**: Broad market intelligence, company-specific data

## 📊 Output Artifacts

Each analysis generates:

### 1. Structured Data (JSON)
- `cv_struct.json`: Parsed CV data with sections
- `skill_profile.json`: Categorized skills analysis
- `market_summary.json`: Market intelligence data

### 2. Final Report (Markdown)
- **Executive Summary**: Key findings and recommendations
- **Skill Gap Analysis**: Missing vs existing skills
- **Market Insights**: Industry trends and demands
- **Learning Path**: Prioritized skill development plan
- **Resources**: Specific courses, certifications, projects

### 3. Analysis Artifacts
- **Similarity Scores**: Skill-to-market alignment metrics
- **Confidence Levels**: Analysis reliability indicators
- **Source Citations**: Market data references

## ⚠️ Limitations

### Current Limitations
- **PDF Parsing**: Requires additional tools (pdf2txt) for complex PDFs
- **Language Support**: Best results with English CVs, limited multilingual
- **Rate Limits**: Constrained by API provider limits
- **Market Data**: Limited to publicly available job postings
- **Real-time Data**: Market intelligence may have 24-48h delay

### Technical Constraints
- **File Size**: CV files should be under 10MB
- **Processing Time**: Analysis takes 30-60 seconds per CV
- **Concurrent Users**: Streamlit app supports ~10 concurrent sessions
- **Memory Usage**: ~500MB RAM per active analysis session

## 🗺️ Roadmap

### Phase 1: Core Enhancements (Current)
- [ ] **Enhanced PDF Support**: Better OCR and layout parsing
- [ ] **Batch Processing**: Analyze multiple CVs simultaneously
- [ ] **Template Support**: Custom CV formats and structures
- [ ] **Caching System**: Reduce API calls for repeated analyses

### Phase 2: Advanced Features (Q1 2026)
- [ ] **Multi-language Support**: Native analysis in 10+ languages
- [ ] **Industry Specialization**: Tailored analysis for finance, healthcare, etc.
- [ ] **Skill Progression**: Career path mapping and timeline planning
- [ ] **Integration APIs**: REST API for third-party integrations

### Phase 3: Intelligence Expansion (Q2 2026)
- [ ] **Real-time Market Data**: Live job posting analysis
- [ ] **Salary Insights**: Compensation benchmarking
- [ ] **Company Matching**: Best-fit employer recommendations
- [ ] **Skill Verification**: Certificate and project validation

### Phase 4: Platform Features (Q3 2026)
- [ ] **User Accounts**: Personal dashboards and history
- [ ] **Collaboration Tools**: Team skill gap analysis
- [ ] **Learning Integration**: Direct course enrollment
- [ ] **Mobile Apps**: Native iOS/Android applications

## 🧪 Testing

### Run Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test files
pytest tests/test_flow.py -v

# Run tests with offline mode (no API calls)
pytest tests/ -k "not integration"
```

### Test Coverage
Current test coverage: **46%** across core modules
- CV Parser: 73% coverage
- Market Intelligence: 54% coverage  
- Report Generation: 77% coverage
- Skill Analysis: 49% coverage

## 🤝 Contributing

### Development Setup
```bash
# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run formatting and linting
black src tests
ruff check src tests

# Run type checking
mypy src
```

### Contributing Guidelines
1. **Fork** the repository and create a feature branch
2. **Write tests** for new functionality (maintain >70% coverage)
3. **Follow code style**: Black formatting, Ruff linting
4. **Update documentation** for user-facing changes
5. **Submit PR** with clear description and test results

## 📄 License

**MIT License**

```
Copyright (c) 2025 Skill Gap Analyst Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 🆘 Support

### Getting Help
- **Documentation**: Check this README and inline code comments
- **Issues**: Report bugs and request features via GitHub Issues
- **Discussions**: Ask questions in GitHub Discussions
- **Email**: Contact maintainers for urgent issues

### Common Issues

#### "Module not found" errors
```bash
# Ensure you're in the virtual environment
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e .
```

#### API key errors
```bash
# Verify your .env file exists and has correct keys
cat .env | grep -E "(OPENAI|TAVILY)_API_KEY"
```

#### PDF parsing issues
```bash
# Install PDF support
pip install ".[pdf]"
# Or install pdf2txt manually
pip install pdfminer.six
```

---

**Built with ❤️ for the AI community**
   sample_cv.txt           state.py, nodes.py         provider.py, tools
```

## Development

- Format: `black .`
- Lint: `ruff .`
- Test: `pytest`
- Pre-commit: Add hooks for black/ruff

---
Replace stubs with your implementation.
