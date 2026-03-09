"""
Agentic RAG System with Google Gemini File Search + Mem0 Memory Layer
======================================================================

This module implements an agentic Retrieval Augmented Generation (RAG) architecture
using Google's Gemini File Search API with Mem0 for intelligent memory management.
The system features multiple specialized agents that work together to provide
intelligent document retrieval, personalized responses, and long-term memory.

Architecture:
- FileSearchManager: Manages file uploads, indexing, and store operations
- MemoryManager: Manages user memory, preferences, and behavioral learning (NEW)
- QueryAgent: Analyzes and routes user queries with memory context
- RetrievalAgent: Performs semantic search using Gemini File Search
- ResponseAgent: Generates contextual responses with memory-enhanced personalization
- AgentOrchestrator: Coordinates all agents and manages workflow

Memory Integration:
- User Level: Long-term preferences, interests, query patterns
- Session Level: Current conversation context and recent interactions
- Agent Level: System behavior adaptations and learning
"""

import os
import time
import json
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import re

try:
    from google import genai
    from google.genai import types
except ImportError:
    print("Warning: google-genai package not installed. Install with: pip install google-genai")
    genai = None
    types = None

try:
    from mem0 import Memory
except ImportError:
    print("Warning: mem0ai package not installed. Install with: pip install mem0ai")
    Memory = None


class QueryType(Enum):
    """Types of queries the system can handle"""
    FACTUAL = "factual"
    ANALYTICAL = "analytical"
    COMPARISON = "comparison"
    SUMMARIZATION = "summarization"
    CREATIVE = "creative"
    GENERAL = "general"


@dataclass
class Document:
    """Represents a document in the system"""
    name: str
    file_path: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    file_id: Optional[str] = None
    indexed_at: Optional[str] = None


@dataclass
class QueryContext:
    """Context for a user query"""
    query: str
    query_type: QueryType
    metadata_filter: Optional[str] = None
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    retrieved_chunks: List[Dict] = field(default_factory=list)
    user_memories: List[Dict] = field(default_factory=list)  # NEW: Retrieved memories


class MemoryManager:
    """
    Manages intelligent memory using Mem0 for user preferences, behavioral patterns,
    and contextual learning. Provides multi-level memory (user, session, agent).

    Memory enables the system to:
    - Remember user preferences and interests
    - Learn from interaction patterns
    - Personalize responses based on history
    - Track domain expertise and query styles
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize memory manager with optional configuration.

        Args:
            config: Mem0 configuration (API keys, storage backend, etc.)
        """
        if Memory is None:
            raise ImportError("mem0ai package is required. Install with: pip install mem0ai")

        # Initialize Mem0
        try:
            print(f"[MemoryManager] Initializing with config keys: {list((config or {}).keys())}")
            
            # If config is a dict, pass it directly - Mem0 will handle it
            # Mem0 1.0+ accepts dict configs and converts them internally
            if config:
                self.memory = Memory.from_config(config)
            else:
                # Use default config
                self.memory = Memory()
            
            self.enabled = True
            print("[MemoryManager] Initialized with Mem0")
        except Exception as e:
            print(f"[MemoryManager] ERROR during initialization: {e}")
            print(f"[MemoryManager] Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            raise

    def search_memories(
        self,
        query: str,
        user_id: str = "default_user",
        limit: int = 5
    ) -> List[Dict]:
        """
        Search for relevant memories based on query.

        Args:
            query: Search query
            user_id: User identifier for personalized memory
            limit: Maximum number of memories to retrieve

        Returns:
            List of relevant memory objects
        """
        if not self.enabled:
            return []

        try:
            print(f"[MemoryManager] Searching memories for: {query[:50]}...")

            results = self.memory.search(
                query=query,
                user_id=user_id,
                limit=limit
            )

            memories = results if isinstance(results, list) else []
            print(f"[MemoryManager] Found {len(memories)} relevant memories")

            return memories

        except Exception as e:
            print(f"[MemoryManager] Error searching memories: {e}")
            return []

    def add_memory(
        self,
        content: str,
        user_id: str = "default_user",
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Add a new memory from interaction.

        Args:
            content: Memory content (conversation, preference, pattern)
            user_id: User identifier
            metadata: Additional metadata (query_type, timestamp, etc.)

        Returns:
            Success status
        """
        if not self.enabled:
            return False

        try:
            print(f"[MemoryManager] Adding memory for user: {user_id}")

            self.memory.add(
                messages=[{"role": "user", "content": content}],
                user_id=user_id,
                metadata=metadata or {}
            )

            print(f"[MemoryManager] Memory stored successfully")
            return True

        except Exception as e:
            print(f"[MemoryManager] Error adding memory: {e}")
            return False

    def get_user_memories(
        self,
        user_id: str = "default_user",
        limit: int = 10
    ) -> List[Dict]:
        """
        Get all memories for a specific user.

        Args:
            user_id: User identifier
            limit: Maximum memories to retrieve

        Returns:
            List of user memories
        """
        if not self.enabled:
            return []

        try:
            memories = self.memory.get_all(user_id=user_id, limit=limit)
            return memories if isinstance(memories, list) else []
        except Exception as e:
            print(f"[MemoryManager] Error retrieving user memories: {e}")
            return []

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a specific memory by ID"""
        if not self.enabled:
            return False

        try:
            self.memory.delete(memory_id=memory_id)
            print(f"[MemoryManager] Deleted memory: {memory_id}")
            return True
        except Exception as e:
            print(f"[MemoryManager] Error deleting memory: {e}")
            return False

    def clear_user_memories(self, user_id: str = "default_user") -> bool:
        """Clear all memories for a specific user"""
        if not self.enabled:
            return False

        try:
            self.memory.delete_all(user_id=user_id)
            print(f"[MemoryManager] Cleared all memories for user: {user_id}")
            return True
        except Exception as e:
            print(f"[MemoryManager] Error clearing memories: {e}")
            return False

    def extract_memory_context(self, memories: List[Dict]) -> str:
        """
        Extract and format memory context for prompt injection.

        Args:
            memories: List of memory objects

        Returns:
            Formatted string for prompt context
        """
        if not memories:
            return ""

        context_parts = ["Relevant user context and preferences:"]

        for i, memory in enumerate(memories, 1):
            # Extract memory text (handle different response formats)
            if isinstance(memory, dict):
                text = memory.get('memory', memory.get('text', memory.get('content', str(memory))))
            else:
                text = str(memory)

            context_parts.append(f"{i}. {text}")

        return "\n".join(context_parts)


class FileSearchManager:
    """
    Manages file search stores, file uploads, and indexing operations.
    Handles all interactions with Gemini's File Search API.
    """

    def __init__(self, client: Any):
        self.client = client
        self.stores: Dict[str, Any] = {}
        self.documents: Dict[str, Document] = {}

    def create_store(self, display_name: str) -> str:
        """
        Create a new file search store.

        Args:
            display_name: Name for the store

        Returns:
            Store name/ID
        """
        print(f"[FileSearchManager] Creating store: {display_name}")

        file_search_store = self.client.file_search_stores.create(
            config={'display_name': display_name}
        )

        store_name = file_search_store.name
        self.stores[display_name] = store_name

        print(f"[FileSearchManager] Store created: {store_name}")
        return store_name

    def list_stores(self) -> List[Any]:
        """List all file search stores"""
        stores = self.client.file_search_stores.list()
        return list(stores)

    def upload_and_index(
        self,
        file_path: str,
        store_name: str,
        display_name: Optional[str] = None,
        custom_metadata: Optional[List[Dict]] = None,
        chunking_config: Optional[Dict] = None
    ) -> Document:
        """
        Upload a file and index it in the store.

        Args:
            file_path: Path to the file to upload
            store_name: Name of the store
            display_name: Display name for the file
            custom_metadata: Custom metadata for filtering
            chunking_config: Configuration for document chunking

        Returns:
            Document object
        """
        print(f"[FileSearchManager] Uploading and indexing: {file_path}")

        if not display_name:
            display_name = Path(file_path).name

        # Prepare config
        config = {'display_name': display_name}

        if chunking_config:
            config['chunking_config'] = chunking_config

        # Upload and index
        print(f"[FileSearchManager] DEBUG: Calling upload_to_file_search_store with file={file_path}, store={store_name}")
        try:
            operation = self.client.file_search_stores.upload_to_file_search_store(
                file=file_path,
                file_search_store_name=store_name,
                config=config
            )
            print(f"[FileSearchManager] DEBUG: Operation returned: {operation}")
        except Exception as e:
            print(f"[FileSearchManager] ERROR: upload_to_file_search_store failed: {e}")
            raise

        if operation is None:
             raise ValueError("upload_to_file_search_store returned None")

        # Wait for completion
        print(f"[FileSearchManager] Waiting for indexing to complete...")
        while not operation.done:
            time.sleep(2)
            operation = self.client.operations.get(operation)

        if operation.error:
            print(f"[FileSearchManager] ERROR: Indexing failed: {operation.error}")
            raise RuntimeError(f"Indexing failed: {operation.error}")

        # Create document record
        doc = Document(
            name=display_name,
            file_path=file_path,
            metadata=custom_metadata or {},
            indexed_at=time.strftime('%Y-%m-%d %H:%M:%S')
        )

        self.documents[display_name] = doc
        print(f"[FileSearchManager] File indexed successfully: {display_name}")

        return doc

    def import_existing_file(
        self,
        file_name: str,
        store_name: str,
        custom_metadata: Optional[List[Dict]] = None
    ):
        """Import an already uploaded file to the store"""
        print(f"[FileSearchManager] Importing file: {file_name}")

        operation = self.client.file_search_stores.import_file(
            file_search_store_name=store_name,
            file_name=file_name,
            custom_metadata=custom_metadata
        )

        # Wait for completion
        while not operation.done:
            time.sleep(2)
            operation = self.client.operations.get(operation)

        if operation.error:
            print(f"[FileSearchManager] ERROR: Import failed: {operation.error}")
            raise RuntimeError(f"Import failed: {operation.error}")

        print(f"[FileSearchManager] File imported successfully")


class QueryAgent:
    """
    Analyzes user queries to determine intent, type, and routing strategy.
    Decides how queries should be processed by other agents.
    """

    def __init__(self, client: Any):
        self.client = client

    def analyze_query(self, query: str, conversation_history: List[Dict] = None) -> QueryContext:
        """
        Analyze a user query to determine its type and processing requirements.

        Args:
            query: User's query string
            conversation_history: Previous conversation context

        Returns:
            QueryContext with analysis results
        """
        print(f"[QueryAgent] Analyzing query: {query[:100]}...")

        # Use Gemini to classify the query type
        analysis_prompt = f"""
        Analyze this user query and classify it into one of these categories:
        - FACTUAL: Looking for specific facts or information
        - ANALYTICAL: Requires analysis or deep understanding
        - COMPARISON: Comparing multiple concepts or items
        - SUMMARIZATION: Requesting a summary
        - CREATIVE: Open-ended or creative question
        - GENERAL: General conversation

        Query: {query}

        Respond with just the category name.
        """

        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=analysis_prompt
            )

            category = response.text.strip().upper()

            # Map to QueryType
            query_type = QueryType.GENERAL
            for qt in QueryType:
                if qt.name == category:
                    query_type = qt
                    break

            print(f"[QueryAgent] Query classified as: {query_type.name}")

        except Exception as e:
            print(f"[QueryAgent] Error classifying query: {e}")
            query_type = QueryType.GENERAL

        return QueryContext(
            query=query,
            query_type=query_type,
            conversation_history=conversation_history or []
        )


class RetrievalAgent:
    """
    Performs semantic search using Gemini File Search.
    Retrieves relevant document chunks based on query context.
    """

    def __init__(self, client: Any):
        self.client = client

    def retrieve(
        self,
        query_context: QueryContext,
        store_names: List[str],
        metadata_filter: Optional[str] = None
    ) -> List[Dict]:
        """
        Retrieve relevant documents using File Search.

        Args:
            query_context: Context about the query
            store_names: Names of stores to search
            metadata_filter: Optional metadata filter

        Returns:
            List of retrieved document chunks
        """
        print(f"[RetrievalAgent] Retrieving documents for query type: {query_context.query_type.name}")
        print(f"[RetrievalAgent] Store names: {store_names}")
        print(f"[RetrievalAgent] Metadata filter: {metadata_filter}")

        # Build the search query with context
        search_query = self._build_search_query(query_context)
        print(f"[RetrievalAgent] Search query: {search_query[:100]}...")

        # Configure file search tool
        file_search_config = types.FileSearch(
            file_search_store_names=store_names
        )

        if metadata_filter:
            file_search_config.metadata_filter = metadata_filter

        try:
            # Execute search with robust retry for 503/Overload
            response = None
            max_retries = 8
            for attempt in range(max_retries):
                try:
                    response = self.client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=search_query,
                        config=types.GenerateContentConfig(
                            tools=[types.Tool(file_search=file_search_config)]
                        )
                    )
                    break # Success
                except Exception as e:
                    # Check for 503 or overload
                    error_str = str(e).lower()
                    is_overload = "503" in error_str or "overloaded" in error_str or "unavailable" in error_str
                    
                    if is_overload and attempt < max_retries - 1:
                        wait_time = (2 ** (attempt + 2)) + (attempt * 2) # Aggressive backoff: 4s, 10s, 22s, 46s...
                        print(f"[RetrievalAgent] ⚠️ API Overloaded (503). Retrying in {wait_time}s... (Attempt {attempt+1}/{max_retries})")
                        time.sleep(wait_time)
                    else:
                        print(f"[RetrievalAgent] ERROR: Retrieval failed after {attempt+1} attempts: {e}")
                        raise e # Re-raise if not overload or out of retries

            # Extract grounding metadata
            chunks = []
            if hasattr(response.candidates[0], 'grounding_metadata'):
                grounding_metadata = response.candidates[0].grounding_metadata
                chunks = self._extract_chunks(grounding_metadata)
                print(f"[RetrievalAgent] Grounding metadata found with {len(chunks)} chunks")
            else:
                print("[RetrievalAgent] WARNING: No grounding metadata found in response")

            query_context.retrieved_chunks = chunks
            print(f"[RetrievalAgent] Retrieved {len(chunks)} relevant chunks")

            if len(chunks) == 0:
                print("[RetrievalAgent] ERROR: No documents retrieved! Check if:")
                print("  1. Files are uploaded to the correct store")
                print("  2. Store ID is valid")
                print("  3. Query matches document content")

            return chunks

        except Exception as e:
            print(f"[RetrievalAgent] ERROR during retrieval: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _build_search_query(self, query_context: QueryContext) -> str:
        """Build an optimized search query based on context"""
        base_query = query_context.query

        # Enhance query based on type
        if query_context.query_type == QueryType.SUMMARIZATION:
            return f"Provide a comprehensive summary addressing: {base_query}"
        elif query_context.query_type == QueryType.COMPARISON:
            return f"Compare and contrast the following: {base_query}"
        elif query_context.query_type == QueryType.ANALYTICAL:
            return f"Provide detailed analysis of: {base_query}"

        return base_query

    def _extract_chunks(self, grounding_metadata) -> List[Dict]:
        """Extract relevant chunks from grounding metadata"""
        chunks = []

        if hasattr(grounding_metadata, 'grounding_chunks') and grounding_metadata.grounding_chunks:
            for chunk in grounding_metadata.grounding_chunks:
                source = 'Unknown'
                if hasattr(chunk, 'retrieved_context'):
                    if hasattr(chunk.retrieved_context, 'title') and chunk.retrieved_context.title:
                        source = chunk.retrieved_context.title
                    elif hasattr(chunk.retrieved_context, 'uri') and chunk.retrieved_context.uri:
                        source = chunk.retrieved_context.uri
                
                # Clean up source name (remove timestamp prefix)
                # Matches YYYYMMDD_HHMMSS_ pattern
                source = re.sub(r'^\d{8}_\d{6}_', '', source)

                chunks.append({
                    'text': getattr(chunk, 'text', ''),
                    'source': source,
                })

        return chunks


class ResponseAgent:
    """
    Generates contextual responses using retrieved information.
    Synthesizes information from multiple sources into coherent answers.
    """

    def __init__(self, client: Any):
        self.client = client

    def generate_response(
        self,
        query_context: QueryContext,
        store_names: List[str],
        metadata_filter: Optional[str] = None,
        include_citations: bool = True,
        memory_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a response using File Search, context, and memory.

        Args:
            query_context: Query context with retrieved chunks
            store_names: Store names to search
            metadata_filter: Optional metadata filter
            include_citations: Whether to include citation information
            memory_context: User memory context for personalization (NEW)

        Returns:
            Dictionary with response text and metadata
        """
        print(f"[ResponseAgent] Generating response...")


        # Build enhanced prompt with memory
        prompt = self._build_response_prompt(query_context, memory_context)

        # Configure file search
        file_search_config = types.FileSearch(
            file_search_store_names=store_names
        )

        if metadata_filter:
            file_search_config.metadata_filter = metadata_filter

        try:
            # Generate response with robust retry
            response = None
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    response = self.client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            tools=[types.Tool(file_search=file_search_config)]
                        )
                    )
                    break 
                except Exception as e:
                    # Check for 503 or overload
                    error_str = str(e).lower()
                    is_overload = "503" in error_str or "overloaded" in error_str or "unavailable" in error_str
                    
                    if is_overload and attempt < max_retries - 1:
                        wait_time = (2 ** (attempt + 2)) + (attempt * 2)
                        print(f"[ResponseAgent] ⚠️ API Overloaded (503). Retrying in {wait_time}s... (Attempt {attempt+1}/{max_retries})")
                        time.sleep(wait_time)
                    else:
                        print(f"[ResponseAgent] ERROR: Generation failed after {attempt+1} attempts: {e}")
                        raise e

            result = {
                'text': response.text,
                'query_type': query_context.query_type.name,
                'citations': []
            }

            # Extract citations if requested
            if include_citations and hasattr(response.candidates[0], 'grounding_metadata'):
                gm = response.candidates[0].grounding_metadata
                result['citations'] = self._extract_citations(gm)

            print(f"[ResponseAgent] Response generated successfully")
            return result

        except Exception as e:
            print(f"[ResponseAgent] Error generating response: {e}")
            return {
                'text': f"I encountered an error processing your request: {str(e)}",
                'error': str(e)
            }

    def _build_response_prompt(self, query_context: QueryContext, memory_context: Optional[str] = None) -> str:
        """Build an enhanced prompt for response generation with memory"""
        prompt_parts = []

        # Add memory context if available (FIRST for priority)
        if memory_context:
            prompt_parts.append(memory_context)
            prompt_parts.append("")

        # Add conversation history if available
        if query_context.conversation_history:
            prompt_parts.append("Previous conversation context:")
            for msg in query_context.conversation_history[-3:]:  # Last 3 messages
                prompt_parts.append(f"{msg['role']}: {msg['content']}")
            prompt_parts.append("")

        # Add type-specific instructions
        type_instructions = {
            QueryType.FACTUAL: "Provide accurate, factual information based on the documents.",
            QueryType.ANALYTICAL: "Provide a detailed analysis with insights and reasoning.",
            QueryType.COMPARISON: "Compare and contrast clearly, highlighting key differences and similarities.",
            QueryType.SUMMARIZATION: "Provide a comprehensive yet concise summary.",
            QueryType.CREATIVE: "Provide a thoughtful, creative response while staying grounded in the documents."
        }

        if query_context.query_type in type_instructions:
            prompt_parts.append(type_instructions[query_context.query_type])
            prompt_parts.append("")

        # Add the actual query
        prompt_parts.append(r"""
IMPORTANT: You are an expert AI assistant. Your goal is to provide comprehensive, well-structured, and visually appealing responses.

**MANDATORY**: Before answering, you MUST use the file_search tool to retrieve relevant context from the knowledge base. Do NOT answer from your own knowledge. Search the files first, then answer based on what you find.

GROUNDING RULES:
1.  **Strictly Grounded**: Answer ONLY using the provided context (documents).
2.  **No Outside Knowledge**: Do NOT use your general knowledge to answer questions if the information is not in the documents.
3.  **Admit Ignorance**: If the answer is not in the documents, state clearly: "I cannot answer this question based on the provided documents."
4.  **Citations**: Use the provided citations to back up your claims.

FORMATTING RULES:
1.  **Use Markdown**: ALWAYS use Markdown for formatting.
2.  **Headings**: Use Level 3 (`###`) and Level 4 (`####`) headings to organize your answer.
3.  **Lists**: Use bullet points (`-`) or numbered lists (`1.`) for steps or items.
4.  **Bold Terms**: **Bold** key terms, concepts, or important entities.
5.  **Code Blocks**: Use fenced code blocks (\`\`\`) for any code or technical syntax.
6.  **Spacing**: Add blank lines between paragraphs and sections for readability.
7.  **No Wall of Text**: Break long paragraphs into smaller chunks.

RESPONSE STRUCTURE:
- **Summary**: Start with a brief, direct answer.
- **Details**: Provide detailed explanation using the formatting rules above.
- **Key Takeaways**: (Optional) Summarize the most important points.

Question:""")
        prompt_parts.append(f"{query_context.query}")

        return "\n".join(prompt_parts)

    def _extract_citations(self, grounding_metadata) -> List[Dict]:
        """Extract citation information"""
        citations = []

        if hasattr(grounding_metadata, 'grounding_chunks') and grounding_metadata.grounding_chunks:
            for i, chunk in enumerate(grounding_metadata.grounding_chunks):
                source = 'Unknown'
                if hasattr(chunk, 'retrieved_context'):
                    if hasattr(chunk.retrieved_context, 'title') and chunk.retrieved_context.title:
                        source = chunk.retrieved_context.title
                    elif hasattr(chunk.retrieved_context, 'uri') and chunk.retrieved_context.uri:
                        source = chunk.retrieved_context.uri

                # Clean up source name (remove timestamp prefix)
                source = re.sub(r'^\d{8}_\d{6}_', '', source)

                citations.append({
                    'index': i + 1,
                    'source': source,
                    'snippet': getattr(chunk, 'text', '')[:200] + '...'
                })

        return citations


class AgentOrchestrator:
    """
    Orchestrates all agents to process user queries end-to-end with intelligent memory.
    Manages workflow, agent coordination, conversation state, and long-term memory.

    With Mem0 integration, the system remembers:
    - User preferences and interests
    - Query patterns and expertise level
    - Conversation context across sessions
    - Behavioral adaptations
    """

    def __init__(self, api_key: str, memory_config: Optional[Dict] = None, enable_memory: bool = True):
        """
        Initialize the orchestrator with Gemini API key and optional memory.

        Args:
            api_key: Gemini API key
            memory_config: Optional Mem0 configuration
            enable_memory: Whether to enable memory features (default: True)
        """
        if genai is None:
            raise ImportError("google-genai package is required. Install with: pip install google-genai")

        self.client = genai.Client(api_key=api_key)
        self.api_key = api_key  # Store for access by enhanced wrapper

        # Initialize all agents
        self.file_manager = FileSearchManager(self.client)
        self.query_agent = QueryAgent(self.client)
        self.retrieval_agent = RetrievalAgent(self.client)
        self.response_agent = ResponseAgent(self.client)

        # Initialize memory (optional)
        self.memory_manager: Optional[MemoryManager] = None
        self.memory_enabled = enable_memory

        if enable_memory and Memory is not None:
            try:
                self.memory_manager = MemoryManager(config=memory_config)
                print("[AgentOrchestrator] Memory layer enabled")
            except Exception as e:
                print(f"[AgentOrchestrator] Warning: Could not initialize memory: {e}")
                self.memory_enabled = False
        else:
            print("[AgentOrchestrator] Memory layer disabled")

        self.conversation_history: List[Dict[str, str]] = []
        self.current_store: Optional[str] = None
        self.current_user_id: str = "default_user"

        print("[AgentOrchestrator] Initialized with all agents")

    def create_knowledge_base(
        self,
        store_name: str,
        file_paths: List[str],
        chunking_config: Optional[Dict] = None
    ) -> str:
        """
        Create a knowledge base by uploading and indexing files.

        Args:
            store_name: Name for the knowledge base
            file_paths: List of file paths to index
            chunking_config: Optional chunking configuration

        Returns:
            Store name/ID
        """
        print(f"\n[AgentOrchestrator] Creating knowledge base: {store_name}")

        # Create store
        store_id = self.file_manager.create_store(store_name)
        self.current_store = store_id

        # Default chunking config if not provided
        if chunking_config is None:
            chunking_config = {
                'white_space_config': {
                    'max_tokens_per_chunk': 500,
                    'max_overlap_tokens': 50
                }
            }

        # Upload and index each file
        for file_path in file_paths:
            if os.path.exists(file_path):
                self.file_manager.upload_and_index(
                    file_path=file_path,
                    store_name=store_id,
                    chunking_config=chunking_config
                )
            else:
                print(f"[AgentOrchestrator] Warning: File not found: {file_path}")

        print(f"[AgentOrchestrator] Knowledge base created with {len(file_paths)} files")
        return store_id

    def add_to_knowledge_base(
        self,
        store_name: str,
        file_paths: List[str],
        chunking_config: Optional[Dict] = None
    ):
        """
        Add files to an existing knowledge base.

        Args:
            store_name: Name/ID of the existing store
            file_paths: List of file paths to index
            chunking_config: Optional chunking configuration
        """
        print(f"\\n[AgentOrchestrator] Adding to knowledge base: {store_name}")
        
        self.current_store = store_name

        # Default chunking config if not provided
        if chunking_config is None:
            chunking_config = {
                'white_space_config': {
                    'max_tokens_per_chunk': 500,
                    'max_overlap_tokens': 50
                }
            }

        # Upload and index each file
        for file_path in file_paths:
            if os.path.exists(file_path):
                self.file_manager.upload_and_index(
                    file_path=file_path,
                    store_name=store_name,
                    chunking_config=chunking_config
                )
            else:
                print(f"[AgentOrchestrator] Warning: File not found: {file_path}")
        
        print(f"[AgentOrchestrator] Added {len(file_paths)} files to {store_name}")

    def query(
        self,
        question: str,
        store_name: Optional[str] = None,
        metadata_filter: Optional[str] = None,
        include_citations: bool = True,
        user_id: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Process a user query through the agentic RAG pipeline with memory integration.

        Memory-Enhanced Workflow:
        1. Retrieve relevant memories (user preferences, past interactions)
        2. Analyze query with memory context
        3. Generate personalized response
        4. Store new memories from the interaction

        Args:
            question: User's question
            store_name: Store to search (uses current_store if None)
            metadata_filter: Optional metadata filter
            include_citations: Whether to include citations
            user_id: User identifier for personalized memory

        Returns:
            Response dictionary with answer, metadata, and memory info
        """
        print(f"\n[AgentOrchestrator] Processing query: {question[:100]}...")

        # Set user ID
        if user_id:
            self.current_user_id = user_id

        # Use current store if none specified
        if store_name is None:
            store_name = self.current_store

        if store_name is None:
            return {
                'text': "No knowledge base is currently active. Please create one first.",
                'error': 'No active store'
            }

        # Step 1: Retrieve relevant memories (if enabled)
        memories = []
        memory_context = ""
        if self.memory_enabled and self.memory_manager:
            memories = self.memory_manager.search_memories(
                query=question,
                user_id=self.current_user_id,
                limit=5
            )
            memory_context = self.memory_manager.extract_memory_context(memories)

        # Step 2: Analyze query type and intent
        # Use provided history or fall back to internal state (though internal state is shared/unsafe in multi-user)
        history_to_use = conversation_history if conversation_history is not None else self.conversation_history
        
        query_context = self.query_agent.analyze_query(
            question, 
            history_to_use
        )
        query_context.user_memories = memories

        # Step 3: Generate memory-enhanced response
        result = self.response_agent.generate_response(
            query_context=query_context,
            store_names=[store_name],
            metadata_filter=metadata_filter,
            include_citations=include_citations,
            memory_context=memory_context
        )

        # Step 4: Store new memory from this interaction (if enabled)
        if self.memory_enabled and self.memory_manager:
            # Create memory content
            memory_content = f"User asked: {question}\nResponse type: {result.get('query_type', 'GENERAL')}"

            # Add memory with metadata
            self.memory_manager.add_memory(
                content=memory_content,
                user_id=self.current_user_id,
                metadata={
                    'query_type': result.get('query_type', 'GENERAL'),
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'has_citations': len(result.get('citations', [])) > 0
                }
            )

        # Update conversation history
        self.conversation_history.append({'role': 'user', 'content': question})
        self.conversation_history.append({'role': 'assistant', 'content': result['text']})

        # Add memory info to result
        result['memories_used'] = len(memories)
        result['memory_enabled'] = self.memory_enabled

        print(f"[AgentOrchestrator] Query processed successfully with {len(memories)} memories")
        return result

    def clear_conversation(self):
        """Clear conversation history"""
        self.conversation_history = []
        print("[AgentOrchestrator] Conversation history cleared")

    def set_user_id(self, user_id: str):
        """Set the current user ID for memory personalization"""
        self.current_user_id = user_id
        print(f"[AgentOrchestrator] User ID set to: {user_id}")

    def get_user_memories(self, user_id: Optional[str] = None, limit: int = 10) -> List[Dict]:
        """Get all memories for a user"""
        if not self.memory_enabled or not self.memory_manager:
            return []

        uid = user_id or self.current_user_id
        return self.memory_manager.get_user_memories(user_id=uid, limit=limit)

    def clear_user_memories(self, user_id: Optional[str] = None) -> bool:
        """Clear all memories for a user"""
        if not self.memory_enabled or not self.memory_manager:
            return False

        uid = user_id or self.current_user_id
        return self.memory_manager.clear_user_memories(user_id=uid)

    def add_user_preference(self, preference: str, user_id: Optional[str] = None) -> bool:
        """Manually add a user preference to memory"""
        if not self.memory_enabled or not self.memory_manager:
            return False

        uid = user_id or self.current_user_id
        return self.memory_manager.add_memory(
            content=f"User preference: {preference}",
            user_id=uid,
            metadata={'type': 'preference', 'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')}
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        stats = {
            'total_stores': len(self.file_manager.stores),
            'total_documents': len(self.file_manager.documents),
            'conversation_length': len(self.conversation_history),
            'current_store': self.current_store,
            'memory_enabled': self.memory_enabled,
            'current_user_id': self.current_user_id
        }

        # Add memory stats if enabled
        if self.memory_enabled and self.memory_manager:
            user_memories = self.get_user_memories(limit=100)
            stats['total_memories'] = len(user_memories)

        return stats


# Convenience function for quick setup
def create_agentic_rag(
    api_key: Optional[str] = None,
    memory_config: Optional[Dict] = None,
    enable_memory: bool = True
) -> AgentOrchestrator:
    """
    Create an agentic RAG system with memory capabilities.

    Args:
        api_key: Gemini API key (uses GEMINI_API_KEY env var if None)
        memory_config: Optional Mem0 configuration
        enable_memory: Whether to enable memory features (default: True)

    Returns:
        AgentOrchestrator instance with memory

    Example:
        # Basic usage with default memory
        rag = create_agentic_rag(api_key='your-key')

        # With custom memory config
        config = {'llm': {'provider': 'openai', 'config': {'model': 'gpt-4'}}}
        rag = create_agentic_rag(api_key='your-key', memory_config=config)

        # Disable memory
        rag = create_agentic_rag(api_key='your-key', enable_memory=False)
    """
    if api_key is None:
        api_key = os.environ.get('GEMINI_API_KEY')

    if not api_key:
        raise ValueError("API key must be provided or set in GEMINI_API_KEY environment variable")

    return AgentOrchestrator(api_key, memory_config=memory_config, enable_memory=enable_memory)


if __name__ == "__main__":
    # Example usage
    print("Agentic RAG System - Example Usage\n")
    print("This module provides an agentic RAG architecture using Gemini File Search.")
    print("\nBasic usage:")
    print("  from agentic_rag import create_agentic_rag")
    print("  rag = create_agentic_rag(api_key='your-api-key')")
    print("  rag.create_knowledge_base('my-docs', ['file1.pdf', 'file2.txt'])")
    print("  result = rag.query('What is...?')")
    print("  print(result['text'])")
