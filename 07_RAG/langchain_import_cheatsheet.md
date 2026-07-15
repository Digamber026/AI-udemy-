# LangChain Import Cheatsheet

A quick reference of commonly used LangChain imports across its modular packages, with use-case comments.

## langchain-core

```python
from langchain_core.documents import Document                     # for representing a piece of text + metadata (used across loaders, retrievers)
from langchain_core.prompts import PromptTemplate                 # for building reusable prompt templates
from langchain_core.prompts import ChatPromptTemplate             # for building chat-style prompt templates
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage  # for constructing chat message objects
from langchain_core.runnables import RunnableLambda               # for wrapping custom functions into the LCEL pipeline
from langchain_core.runnables import RunnablePassthrough          # for passing input through unchanged in a chain
from langchain_core.output_parsers import StrOutputParser         # for parsing model output into a plain string
from langchain_core.output_parsers import JsonOutputParser        # for parsing model output into JSON
from langchain_core.output_parsers import PydanticOutputParser    # for parsing model output into a Pydantic model
from langchain_core.retrievers import BaseRetriever                # for building custom retriever classes
from langchain_core.tools import tool                              # for defining a function as an LLM-callable tool
from langchain_core.callbacks import BaseCallbackHandler            # for creating custom callback/logging handlers
from langchain_core.vectorstores import VectorStore                 # for building custom vector store integrations
from langchain_core.embeddings import Embeddings                    # for building custom embedding classes
```

## langchain-text-splitters

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter  # for splitting text by characters, respecting paragraph/sentence boundaries
from langchain_text_splitters import CharacterTextSplitter            # for simple fixed-size character-based splitting
from langchain_text_splitters import TokenTextSplitter                 # for splitting based on token count (useful for LLM context limits)
from langchain_text_splitters import MarkdownHeaderTextSplitter        # for splitting Markdown docs by header structure
from langchain_text_splitters import HTMLHeaderTextSplitter            # for splitting HTML docs by header tags
from langchain_text_splitters import PythonCodeTextSplitter            # for splitting Python source code into logical chunks
from langchain_text_splitters import Language                          # for specifying language-aware splitting (e.g. JS, Python, Java)
```

> Note: Older code may show `from langchain.text_splitter import RecursiveCharacterTextSplitter`. This still works in many versions but is deprecated in favor of the dedicated `langchain-text-splitters` package.

## langchain-community

```python
from langchain_community.document_loaders import PyPDFLoader                     # for loading and parsing PDF files
from langchain_community.document_loaders import CSVLoader                       # for loading tabular data from CSV files
from langchain_community.document_loaders import TextLoader                      # for loading plain .txt files
from langchain_community.document_loaders import UnstructuredWordDocumentLoader  # for loading .docx files
from langchain_community.document_loaders import WebBaseLoader                   # for loading and parsing web pages
from langchain_community.document_loaders import DirectoryLoader                 # for bulk-loading all files in a folder
from langchain_community.document_loaders import YoutubeLoader                   # for loading YouTube video transcripts
from langchain_community.document_loaders import WikipediaLoader                 # for pulling documents from Wikipedia
from langchain_community.vectorstores import FAISS                               # for local, in-memory vector similarity search
from langchain_community.vectorstores import Chroma                              # for local vector storage (community version, pre-langchain-chroma split)
from langchain_community.tools import DuckDuckGoSearchRun                        # for web search via DuckDuckGo
from langchain_community.utilities import WikipediaAPIWrapper                     # for querying Wikipedia's API directly
from langchain_community.chat_message_histories import ChatMessageHistory        # for storing conversation history in memory
```

## langchain (main package: chains, agents, memory)

```python
from langchain.chains import create_retrieval_chain                          # for building a RAG-style retrieval + generation chain
from langchain.chains.combine_documents import create_stuff_documents_chain  # for combining multiple docs into one LLM call
from langchain.agents import create_react_agent                              # for building a ReAct-style reasoning + acting agent
from langchain.agents import AgentExecutor                                   # for running an agent with tool-calling loop
from langchain.memory import ConversationBufferMemory                        # for storing full conversation history
from langchain.memory import ConversationSummaryMemory                       # for storing a summarized version of conversation history
```

## langchain-openai

```python
from langchain_openai import ChatOpenAI          # for using OpenAI's chat models (e.g. gpt-4o) in a chain
from langchain_openai import OpenAIEmbeddings    # for generating text embeddings via OpenAI's embedding models
```

## Other integration packages

```python
from langchain_google_genai import ChatGoogleGenerativeAI       # for using Google's Gemini models as the LLM
from langchain_google_genai import GoogleGenerativeAIEmbeddings # for Gemini embeddings
from langchain_anthropic import ChatAnthropic                    # for using Anthropic's Claude models
from langchain_qdrant import QdrantVectorStore                    # for storing/retrieving vectors using Qdrant
from langchain_pinecone import PineconeVectorStore                 # for storing/retrieving vectors using Pinecone
from langchain_chroma import Chroma                                # for storing/retrieving vectors using ChromaDB (dedicated package)
from langchain_huggingface import HuggingFaceEmbeddings             # for embeddings via HuggingFace models
```

## LangGraph (multi-step / agentic workflows)

```python
from langgraph.graph import StateGraph, START, END   # for building stateful graph-based workflows
from langgraph.checkpoint.memory import MemorySaver    # for saving/restoring graph state across runs
from langgraph.prebuilt import create_react_agent      # for a prebuilt ReAct agent (LangGraph version)
```

## LangSmith (observability)

```python
from langsmith import Client       # for tracing, logging, and evaluating LLM runs
from langsmith import traceable    # decorator for auto-tracing custom functions
```

---

### Notes

- Python import paths use underscores (`langchain_openai`) even though the pip/PyPI package names use hyphens (`langchain-openai`).
- Install packages individually as needed, e.g.:
  ```bash
  pip install langchain-openai langchain-community langchain-text-splitters faiss-cpu
  ```
- `langgraph` and `langsmith` are standalone packages, not sub-packages of `langchain`.
- LangChain's package boundaries shift between versions — check the official docs if an import path here doesn't match your installed version.