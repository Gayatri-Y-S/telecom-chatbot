"""
Builds the RAG chain:
  merged retriever → prompt → Qwen3-32B on Groq → string output
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_groq import ChatGroq

from retriever import build_retriever

SYSTEM_PROMPT = """You are a helpful and professional telecom customer care assistant.
Your job is to help customers resolve technical issues with their mobile service.

Use ONLY the context below to answer the customer's question.
The context comes from two sources:
- FAQ entries (general policy and how-to information)
- Past support tickets (real resolved cases with step-by-step resolutions)

If the context does not contain enough information to answer confidently, say so clearly \
and suggest the customer call 611 or use the MyTelecom app.

Context:
{context}
"""

"""
[TICKET]
Slow internet issue...
    
---
    
[GUIDE]
Check signal strength...

# So, the function formats multiple retrieved documents into one clean text block for the LLM.
"""

def _format_docs(docs: list[Document]) -> str:
    sections = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown").upper()
        sections.append(f"[{source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(sections)


def build_chain():
    retriever = build_retriever()

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    llm = ChatGroq(
        model="qwen/qwen3-32b",
        temperature=0,
        max_tokens=None,
        reasoning_format="parsed",
        timeout=None,
        max_retries=2,
    )

    chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

"""
Think of `build_chain()` as **building the complete RAG pipeline**.

### 1. Get the retriever

```python
retriever = build_retriever()
```

→ Creates the **retriever** that finds relevant information from your Chroma database.

---

### 2. Create the prompt

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}"),
])
```

→ Creates the prompt that will be sent to the LLM.

It contains:

```text
System instructions
      +
User's question
```

---

### 3. Create the LLM

```python
llm = ChatGroq(
    model="qwen/qwen3-32b",
    temperature=0,
    ...
)
```

→ Creates the **Qwen3-32B LLM running through Groq**.

---

### 4. Connect everything ⭐

```python
chain = (
    {"context": retriever | _format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

This is the most important part.

Think:

```text
User question
      ↓
Retriever
      ↓
Relevant documents
      ↓
_format_docs()
      ↓
Prompt + question + context
      ↓
Qwen LLM
      ↓
Final text answer
```

Specifically:

```python
retriever | _format_docs
```

→ Find relevant documents → format them.

```python
"question": RunnablePassthrough()
```

→ Pass the **original user question** unchanged.

```python
| prompt
```

→ Put question + retrieved context into the prompt.

```python
| llm
```

→ Send it to Qwen.

```python
| StrOutputParser()
```

→ Convert the LLM response into normal text.

### In one sentence:

**`build_chain()` connects Retriever → Prompt → LLM → Final Answer to create your RAG chatbot pipeline.**
"""