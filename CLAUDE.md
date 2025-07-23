# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a LangChain Context Engineering template - a comprehensive framework for building production-ready LangChain applications using systematic context engineering principles. The repository provides structured patterns, examples, and workflows for developing robust LangChain chains, agents, and tools.

## Development Setup Commands

```bash
# Environment setup
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Environment configuration
cp .env.example .env
# Edit .env with required API keys

# Verification commands
python examples/basic_chain.py                    # Test basic functionality
python -c "import langchain; print('LangChain installed successfully')"

# Testing commands
pytest examples/tests/ -v                         # Run example tests with verbose output
pytest tests/ -v                                  # Run all repository tests
pytest tests/test_chains.py                       # Run specific test file
pytest --cov=examples --cov-report=term-missing   # Run with coverage report
pytest --tb=short                                 # Short traceback format

# Development commands (if linting tools are available)
black examples/ tests/                             # Format code
flake8 examples/ tests/                           # Lint code
mypy examples/ tests/                             # Type checking
```

## Architecture Structure

```
langchain-context-engineering/
├── .claude/                    # Enhanced Claude Code configuration
│   └── settings.local.json    # Comprehensive permissions & project settings
├── LIPs/                      # LangChain Implementation Plans
│   ├── templates/             # Plan templates
│   └── EXAMPLE_rag_chain.md   # Example implementation plan
├── examples/                  # LangChain code examples (critical for patterns)
│   ├── README.md              # Examples documentation
│   ├── basic_chain.py         # Chain creation, error handling, validation
│   ├── rag_chain.py          # RAG implementation patterns
│   ├── agent_chain.py        # Agent-based chain patterns
│   ├── memory_chain.py       # Conversation memory patterns
│   ├── tools/                # Custom tool implementations
│   │   ├── README.md         # Tools documentation
│   │   ├── custom_tool.py    # Custom tool patterns
│   │   └── web_search_tool.py # Web search integration
│   └── tests/                # Example-specific test patterns
│       ├── README.md         # Testing documentation
│       ├── test_chains.py    # Chain testing patterns
│       ├── test_agents.py    # Agent testing patterns
│       └── conftest.py       # Pytest fixtures and configuration
├── tests/                    # Repository-level test patterns
├── LANGCHAIN_RULES.md        # Comprehensive LangChain development guidelines
├── INITIAL.md               # Template for feature requests
└── requirements.txt          # Python dependencies
```

## Key Development Patterns

### LangChain Component Development
- **Chain Composition**: Follow single-purpose chain patterns from `examples/`
- **Agent Architecture**: Use ReAct patterns with proper error handling
- **Memory Management**: Implement appropriate memory types (Buffer, Window, Summary)
- **Tool Development**: Create custom tools using `BaseTool` interface
- **Prompt Engineering**: Use `PromptTemplate` exclusively, never hardcode prompts

### Required Environment Variables
```bash
# Core API Keys
OPENAI_API_KEY=your_openai_api_key           # Required for OpenAI models
ANTHROPIC_API_KEY=your_anthropic_api_key     # Optional for Claude models

# LangSmith Tracing (Optional)
LANGCHAIN_TRACING_V2=true                    # Enable tracing
LANGCHAIN_API_KEY=your_langsmith_key         # LangSmith API key
LANGCHAIN_PROJECT=your_project_name          # Project name for tracing

# Vector Store Configuration (Optional)
PINECONE_API_KEY=your_pinecone_key_here     # Pinecone vector store
PINECONE_ENVIRONMENT=your_pinecone_env      # Pinecone environment

# Testing Configuration
PYTEST_FAST_MODE=false                       # Skip slow integration tests
PYTEST_ALLOW_API_TESTS=true                  # Allow tests that make API calls
PYTEST_DEBUG=false                           # Debug mode for tests
```

### Core Dependencies & Project Settings
Based on `.claude/settings.local.json`:
- **Default LLM**: `ChatOpenAI` with temperature=0.7, max_tokens=1000
- **Memory Type**: `ConversationBufferWindowMemory` for conversation handling
- **Vector Store**: `Chroma` as default, with `OpenAIEmbeddings`
- **Testing**: pytest with 80% coverage threshold, API mocking enabled
- **Linting**: black, flake8, mypy integration

## Architecture Guidelines

### Context Engineering Methodology
This template follows Context Engineering principles:
1. **Comprehensive Context**: Provide complete context through examples, rules, and documentation
2. **Pattern Consistency**: Follow established patterns across all LangChain components
3. **Systematic Validation**: Use structured testing and validation approaches
4. **Self-Correcting Implementation**: Built-in error handling and recovery patterns

### File Organization Standards
- Keep files under 300 lines
- Group related chains in modules (`rag_chains.py`, `agent_chains.py`)
- Separate tool definitions into `tools/` subdirectory
- Use clear imports and PEP 8 conventions
- Implement comprehensive docstrings with usage examples

### Code Patterns from examples/basic_chain.py
- **Input Validation**: Use Pydantic models (`ChainInput`, `ChainResponse`) for all inputs/outputs
- **LLM Configuration**: Standardized `create_llm()` function with timeout, retries, temperature settings
- **Safe Execution**: `safe_chain_execution()` wrapper with timing, logging, and error handling
- **Environment Validation**: `validate_environment()` function to check required API keys
- **Structured Responses**: Always return structured response objects with success/error status

### Testing Requirements
- Use `examples/tests/` patterns for chain-specific testing
- Mock external API calls using `pytest-mock` (as configured in settings)
- Test both success and failure scenarios with structured response validation
- Maintain 80% coverage threshold (enforced by Claude settings)
- Use performance benchmarks for chain execution timing

### Error Handling Patterns
- Follow `safe_chain_execution()` pattern from examples/basic_chain.py
- Implement graceful degradation for tool failures with structured error responses
- Use comprehensive logging with execution timing
- Validate inputs with Pydantic models before chain execution
- Include request timeout (30s) and retry logic (3 attempts) for API calls

## Important Implementation Notes

### Security Considerations
- Never hardcode API keys in source code
- Use environment variables for all sensitive configuration
- Validate and sanitize all user inputs
- Implement content filtering for inappropriate content

### Performance Optimization
- Monitor and optimize prompt lengths
- Use appropriate context window sizes
- Implement caching for expensive operations
- Use persistent storage for conversation memory

### LangChain Best Practices
- Follow official LangChain documentation patterns
- Use structured outputs to prevent injection attacks
- Implement proper model switching capabilities
- Monitor token usage for cost optimization
- Use LangSmith for tracing and debugging

## Claude Code Integration

### Enhanced Permissions (from .claude/settings.local.json)
- **File Operations**: Read, write, create (delete disabled for safety)
- **Network Access**: Enabled for LangChain docs, OpenAI/Anthropic APIs, GitHub, PyPI
- **Code Execution**: Python and shell execution enabled with package installation
- **Auto-features**: Docstring generation, type hints, example usage documentation

### Development Workflow
1. **Read `LANGCHAIN_RULES.md`** for comprehensive LangChain development guidelines
2. **Study `examples/basic_chain.py`** for standardized patterns (input validation, error handling, logging)
3. **Use `examples/tests/` patterns** for testing new components
4. **Follow Pydantic models** for all input/output validation (`ChainInput`, `ChainResponse`)
5. **Implement `safe_chain_execution()` pattern** for all chain operations
6. **Validate environment** using the established `validate_environment()` function
7. **Use structured logging** with execution timing and error tracking