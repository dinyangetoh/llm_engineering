# LLM Engineering Course - Comprehensive Learning Document

> Created by: David Inyang-Etoh (Bruno assisting)
> Date: March 2, 2026
> Purpose: Assessment preparation and exercise template

---

## Course Overview

This is the **LLM Engineering** course by Edward Donner. The course covers building production-ready LLM applications from basics to advanced agentic systems.

### Course Structure (8 Weeks)
- **Week 1**: First LLM Project - Webpage Summarizer
- **Week 2**: Frontier Model APIs & Building Chat Interfaces
- **Week 3**: Google Colab & Alternative Environments
- **Week 4**: Code Generation (Python → C++)
- **Week 5**: RAG (Retrieval-Augmented Generation)
- **Week 6-8**: [Final projects and advanced agent systems]

---

## Week 1: Your First LLM Project

### Topics Covered
- Connecting to OpenAI API
- Basic LLM calls using Chat Completions API
- Web scraping with BeautifulSoup
- Building a webpage summarizer
- Working with Jupyter Notebooks

### Core Concepts Introduced
- **API Keys**: Loading from .env file using `load_dotenv()`
- **OpenAI Client**: Using `openai.OpenAI()` 
- **Chat Completions**: Message structure with roles (system, user, assistant)
- **Web Scraping**: Using `requests` and `BeautifulSoup` for content extraction

### Day Breakdown
| Day | Topic |
|-----|-------|
| Day 1 | Getting started with OpenAI API |
| Day 2 | Using Ollama (local models) |
| Day 3 | [Content needs review] |
| Day 4 | Tokenization with tiktoken |
| Day 5 | Multi-prompt & brochure generator |

### Exercise Format (Week 1)
Students build a webpage summarizer that:
1. Takes a URL input
2. Fetches the webpage content
3. Uses LLM to summarize the content

### Your Week 1 Exercise: Tennis News Summarizer ✅

**PR**: [#1396](https://github.com/ed-donner/llm_engineering/pull/1396)

**Two Versions Submitted:**
1. **OpenAI Version**: `day1_tennis_news_today.ipynb`
2. **Ollama Version**: `day1_tennis_news_today_ollama.ipynb` (using gemma3)

**What it does:**
- Scrapes ATP Tour homepage for tennis news
- Processes the content with robust website scraping (user-agent headers, HTML cleaning)
- Uses LLM to generate humorous, markdown-formatted summaries including:
  - Player highlights
  - Rankings updates
  - Talking points
  - Match recommendations

**Components Used:**
| Component | Purpose |
|-----------|---------|
| `requests` + BeautifulSoup | Web scraping with user-agent headers |
| OpenAI API / Ollama | LLM for summary generation |
| Markdown display | Show results in Jupyter |

**Key Features:**
- Robust website scraping with proper headers
- Prompt engineering for concise, engaging summaries
- Error handling for API keys
- Step-by-step code organization for reproducibility

---

## Week 2: Frontier Model APIs & Chat Interfaces

### Topics Covered
- Calling multiple frontier model APIs (OpenAI, Anthropic, Ollama)
- Streaming responses token-by-token using `stream=True` and `yield`
- Crafting effective system prompts
- Building conversational chat with multi-turn history
- Creating interactive UIs with Gradio
- Fetching and cleaning website content

### ML Concepts Introduced
- **OpenAI-compatible client pattern**: Using same client interface for multiple providers
- **Streaming**: Real-time token-by-token response generation
- **System prompts**: Grounding LLM responses with context
- **Message history**: Maintaining conversation context across turns
- **Web scraping patterns**: Extracting clean text from HTML

### Your Week 2 Exercise: Website AI Assistant ✅

**Location**: `week2/community-contributions/dinyangetoh/week2_EXERCISE.ipynb`

**What it does**:
- User pastes any public website URL into a chat window
- Assistant reads the site, summarizes it, and answers questions
- Grounded only in what it found on the website

**Components Used**:
| Component | Purpose |
|-----------|---------|
| `requests` + `BeautifulSoup` | Crawl same-domain pages and extract clean readable text |
| OpenAI client (OpenAI-compatible) | Connect to OpenAI, Anthropic Claude, and Ollama |
| System prompt with injected context | All crawled page text appended to system prompt |
| Streaming with `yield` | LLM replies stream token-by-token in real-time |
| Multi-turn message history | Full conversation history passed on every call |
| `gr.Blocks` + `gr.Chatbot` | Gradio chat UI with welcome message |
| URL detection in chat | Detects URLs and triggers crawl inline |
| Loading indicator | Status bubble during crawl |

**How it works**:
1. **Launch** — Gradio opens with welcome message
2. **Paste URL** — Assistant detects URL, shows loading indicator
3. **Crawl** — Up to 20 same-domain pages fetched and cleaned
4. **Summarise** — Model returns 2-3 sentence summary + 3-5 key findings
5. **Conversation** — Follow-up questions answered from crawled text only
6. **Switch site** — New URL resets knowledge base
7. **Model choice** — Switch between GPT, Claude, and Ollama

**Models Supported**:
- `gpt-4.1-mini` (OpenAI)
- `claude-sonnet-4-5-20250929` (Anthropic)
- `llama3.2` (Ollama)

---

## Week 3: Google Colab & HuggingFace Pipelines

### Status: ✅ COMPLETE - Outstanding Exercise Submitted!

**Your Week 3 Exercise**: "Multi-Modal Document Studio" - An end-to-end document analysis pipeline

### Topics Covered
- **Day 1**: Google Colab & Diffusion/TTS Models
  - GPU-accelerated notebooks on free cloud hardware
  - Image generation with SDXL-Turbo, Stable Diffusion XL, FLUX.1-schnell
  - Text-to-speech with speecht5_tts
  - Core theme: Remote GPU compute and multimodal generation

- **Day 2**: HuggingFace Pipelines
  - High-level `pipeline()` API for inference
  - Sentiment analysis, NER, question answering, summarization, translation
  - Zero-shot classification, text generation, image generation, audio TTS
  - Core theme: Frictionless model inference without knowing underlying architecture

- **Day 3**: Tokenizers
  - AutoTokenizer API - encoding/decoding text to token IDs
  - Comparing vocabularies across Llama 3.1, Phi-4, DeepSeek, QwenCoder
  - `apply_chat_template` - seeing how Python message dicts become token sequences
  - Core theme: What actually enters and exits an LLM — the "aha moment"

- **Day 4**: Models (Lower-level API)
  - AutoModelForCausalLM for direct model control
  - Quantization with BitsAndBytesConfig (4-bit NF4)
  - TextStreamer for streaming output
  - Memory management with gc and torch.cuda.empty_cache()
  - Ran Llama 3.2, Phi-4, Gemma, Qwen3, DeepSeek-R1 locally on T4
  - Core theme: Controlling model loading, memory footprint, and generation

- **Day 5**: Meeting Minutes Product
  - End-to-end product combining everything
  - Whisper pipeline for audio transcription (ASR)
  - Llama 3.2 (4-bit quantized) for structured markdown output
  - Core theme: Chaining open-source models into useful products

### ML Concepts Introduced
- **HuggingFace Pipeline API**: High-level abstraction for model inference
- **AutoTokenizer**: Understanding token IDs and chat prompt formatting
- **Quantization**: INT8 via optimum-quanto (MPS) and bitsandbytes (CUDA)
- **TextIteratorStreamer**: Real-time streaming token generation
- **Device-aware loading**: MPS (Apple Silicon), CUDA (NVIDIA), CPU
- **Memory management**: gc.collect(), empty_cache()
- **Multi-modal chaining**: Combining Whisper + LLM + TTS

### Business Value
Colab provides cloud-based GPU access for running LLM experiments without local hardware requirements. Pipeline API enables rapid prototyping without deep ML knowledge.

### Your Week 3 Exercise: Multi-Modal Document Studio ✅

**Status**: COMPLETE - Sophisticated end-to-end project!

**What it does:**
- Accepts PDF, TXT, or pasted text (contracts, emails, articles, policies)
- Extracts and previews token structure and chat-template formatted prompts
- Identifies named entities (people, organisations, locations) via NER pipeline
- Scores each clause for risk categories using zero-shot classification
- Detects overall document tone/sentiment
- Generates structured analyst brief via locally-running quantized LLM with streaming
- Synthesizes brief as audio using text-to-speech model
- Exposes everything through Gradio Blocks UI with file upload

**Concepts Applied:**
| Component | Purpose |
|-----------|---------|
| `pipeline()` API | NER, sentiment, zero-shot classification, text-to-speech |
| `AutoTokenizer` + `apply_chat_template` | Understanding token IDs and chat prompt formatting |
| `AutoModelForCausalLM` | Loading and running local instruction-tuned LLM |
| INT8 quantization | Via optimum-quanto (Apple MPS) and bitsandbytes (CUDA) |
| `TextIteratorStreamer` + background thread | Real-time streaming token generation |
| Device-aware loading | MPS (Mac), CUDA (NVIDIA), or CPU |
| Memory management | `gc.collect()`, `empty_cache()` |

**Hardware Support:** Runs on Apple Silicon (`mps`), NVIDIA GPU (`cuda`), or CPU

**Pipeline Flow:**
1. **Upload** → Extract text from PDF/TXT
2. **Token Preview** → Show token count and chat-formatted prompt
3. **NER** → Identify named entities
4. **Risk Scoring** → Zero-shot classification per clause
5. **Sentiment** → Overall document tone
6. **LLM Brief** → Generate structured summary with streaming
7. **TTS** → Convert brief to speech Pipeline API enables rapid prototyping without deep ML knowledge.

---

## Week 4: Code Generation

### Status: 📝 In Progress (Days 3-5 available)

### Topics Covered
- Using frontier models to generate high-performance code
- Converting Python code to C++ for performance
- Working with high-end models (GPT-5, Claude 4.5 Sonnet, Gemini 2.5 Pro, Grok 4)
- Cost considerations when using premium models

### ML Concepts Introduced
- **Code generation**: Using LLMs to write code from natural language
- **Performance optimization**: Converting to compiled languages (C++)
- **Model selection**: Choosing right model for task vs cost

### Day Breakdown
| Day | Topic |
|-----|-------|
| Day 3 | Code Generator - Python to C++ |
| Day 4 | [To be reviewed] |
| Day 5 | [To be reviewed] |

### Business Value
Code generation can significantly speed up development cycles and help modernize legacy codebases.

---

## Week 5: RAG (Retrieval-Augmented Generation)

### Status: 📝 In Progress

### Topics Covered
- RAG fundamentals for question-answering systems
- Building an expert knowledge worker for business (Insurellm insurance company)
- Chroma vector database
- Embeddings and text encoding
- Document chunking strategies
- Advanced RAG techniques:
  - LLM-based intelligent chunking
  - Document pre-processing
  - Chunk headline generation
  - Evaluation (evals)

### ML Concepts Introduced
- **Retrieval-Augmented Generation (RAG)**: Combining LLM with knowledge base
- **Vector embeddings**: Converting text to numerical representations
- **Chroma DB**: Persistent vector database for similarity search
- **Semantic search**: Finding relevant documents based on meaning
- **Chunk optimization**: Breaking documents for optimal retrieval
- **Evaluation metrics**: Measuring RAG system accuracy

### Day Breakdown
| Day | Topic |
|-----|-------|
| Day 1 | RAG Fundamentals - Simple question answering |
| Day 2 | Embeddings & Chroma DB |
| Day 3 | [To be reviewed] |
| Day 4 | [To be reviewed] |
| Day 5 | Advanced RAG - Pro techniques with LangChain 1.0 |

### Business Value
RAG is "perhaps the most immediately applicable technique" in the course! Commercial applications include:
- Company contract querying
- Product specification search
- Customer support automation
- Internal knowledge base access

---

## Exercise Template for Weeks 4 & 5

Based on your Week 2 exercise structure, here's a template for Weeks 4 and 5:

```markdown
# Week [4/5] Exercise - [Your Project Name]

## What I Learned in Week [4/5]

Week [4/5] covered:
- [Concept 1]
- [Concept 2]
- [Concept 3]
- [ML Concept introduced]

## How I Applied It — The [Project Name]

[Description of what the application does]

## Components Used

| Component | Purpose |
|-----------|---------|
| [Tech 1] | [What it does] |
| [Tech 2] | [What it does] |
| [Tech 3] | [What it does] |

## How It Works

1. **Step 1** — [Description]
2. **Step 2** — [Description]
3. **Step 3** — [Description]
...

## ML Concepts Demonstrated

- **[Concept Name]**: [Brief explanation]
```

---

## Key ML Concepts Summary (All Weeks)

| Concept | Week | Description |
|---------|------|-------------|
| API Keys & Environment Variables | 1 | Loading secrets securely with `load_dotenv()` |
| Chat Completions API | 1 | Using messages array with roles (system, user, assistant) |
| Web Scraping | 1 | BeautifulSoup for HTML parsing and content extraction |
| Tokenization | 1 | Using tiktoken to count and understand tokens |
| Multi-prompt Chaining | 1 | Breaking complex tasks into prompt chains |
| OpenAI-Compatible Client | 2 | Single interface for multiple LLM providers |
| Streaming Responses | 2 | Real-time token generation with `yield` and `stream=True` |
| System Prompts | 2 | Grounding responses with context |
| Multi-turn History | 2 | Maintaining conversation state across calls |
| Gradio UI | 2 | Building interactive chat interfaces with `gr.Blocks` |
| Google Colab | 3 | Cloud-based notebook execution with GPU access |
| HuggingFace Pipeline API | 3 | High-level abstraction for model inference (NER, sentiment, etc.) |
| AutoTokenizer | 3 | Encoding/decoding text to token IDs |
| apply_chat_template | 3 | Converting Python message dicts to token sequences |
| Quantization (INT8) | 3 | Reducing model size with optimum-quanto / bitsandbytes |
| TextIteratorStreamer | 3 | Real-time streaming from local LLM |
| Memory Management | 3 | gc.collect(), torch.cuda.empty_cache() |
| Multi-modal Chaining | 3 | Combining Whisper + LLM + TTS in one pipeline |
| Code Generation | 4 | Using LLMs to write code (Python → C++) |
| Retrieval-Augmented Generation (RAG) | 5 | Combining LLM with knowledge base for Q&A |
| Vector Embeddings | 5 | Converting text to numerical representations |
| Chroma DB | 5 | Persistent vector database for semantic search |
| Semantic Search | 5 | Finding relevant documents based on meaning |
| Document Chunking | 5 | Breaking documents for optimal retrieval |
| Advanced RAG | 5 | LLM-based chunking, document pre-processing |
| Evaluation (Evals) | 5 | Measuring RAG system accuracy |

---

## Exercise Template for Week 4 (Code Generation)

Based on Week 2 structure, here's the template for your Week 4 exercise:

```markdown
# Week 4 Exercise - [Your Project Name]

## What I Learned in Week 4

Week 4 covered:
- Using frontier models for code generation (Python → C++)
- Working with high-performance models (GPT-5, Claude 4.5, Gemini 2.5 Pro, Grok 4)
- Performance optimization through code translation
- Cost considerations for premium models

## How I Applied It — The [Project Name]

[Description of your code generation application]

## Components Used

| Component | Purpose |
|-----------|---------|
| [Model used] | [Why you chose it] |
| [Code parsing] | [How you handle code input] |
| [Output handling] | [How you process generated code] |

## How It Works

1. **Input** — [User provides...]
2. **Processing** — [Model generates...]
3. **Output** — [Result delivered...]
```

---

## Exercise Template for Week 5 (RAG)

Based on your Week 2 structure and week 5 content, here's the template for your Week 5 exercise:

```markdown
# Week 5 Exercise - [Your RAG Project Name]

## What I Learned in Week 5

Week 5 covered:
- Retrieval-Augmented Generation (RAG) fundamentals
- Chroma vector database for semantic search
- Text embeddings with OpenAI
- Document chunking strategies
- Building a question-answering system

## How I Applied It — The [Project Name]

This notebook implements a RAG-based [question-answering system / knowledge base assistant] that:
- Loads documents from [source]
- Creates embeddings using [embedding model]
- Stores in Chroma vector database
- Answers user questions based on retrieved context

## Components Used

| Component | Purpose |
|-----------|---------|
| `OpenAI` client | For embeddings and chat completion |
| `chromadb` | Persistent vector database for similarity search |
| `tiktoken` | For token counting and chunk optimization |
| `gradio` | User interface for Q&A |

## RAG Pipeline

1. **Ingest** — Load and parse documents from knowledge base
2. **Chunk** — Split into optimal sized chunks
3. **Embed** — Convert chunks to vector embeddings using OpenAI
4. **Store** — Save to Chroma collection
5. **Retrieve** — Query with user question, get similar chunks
6. **Generate** — Pass retrieved context to LLM for answer

## Key RAG Concepts Demonstrated

- **Semantic Search**: Finding relevant docs by meaning, not keywords
- **Context Window Management**: Passing relevant chunks to LLM
- **Source Attribution**: Citing sources in answers
- **Chunk Optimization**: Balancing chunk size for retrieval accuracy
```

---

## Next Steps

1. ✅ **Week 3 complete** - Outstanding Multi-Modal Document Studio project!
2. **Review Week 4 day notebooks** (especially day 3 which has code generation)
3. **Build Week 4 exercise** using the code generation template
4. **Review Week 5 RAG content** thoroughly
5. **Build Week 5 exercise** using the RAG template

---

## Quick Reference: Your Completed Exercises

| Week | Project | Location |
|------|---------|----------|
| Week 1 | Tennis News Summarizer (OpenAI + Ollama) | [PR #1396](https://github.com/ed-donner/llm_engineering/pull/1396) |
| Week 2 | Website AI Assistant | `week2/community-contributions/dinyangetoh/` |
| Week 3 | Multi-Modal Document Studio | Colab exercise you shared |
| Week 4 | **Polyglot Code Translator** (TS → Python + Go) | `week4/community-contributions/dinyangetoh/` ✅ |
| Week 5 | **Smart Document Q&A with RAG** | `week5/community-contributions/dinyangetoh/` ✅ |

---

*Document updated March 2, 2026*