"""
Sample Data Processor for Structure-Aware Retriever.

Generates realistic structured documentation, Table of Contents hierarchy,
and leaf sections for Python 3.11 & FastAPI when source PDFs are not downloaded.
Enables instant end-to-end testing of the pipeline without multi-gigabyte downloads.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def get_project_root() -> Path:
    return Path(__file__).resolve().parent


def create_sample_toc_and_leaves() -> Dict[str, Any]:
    """Create structured Table of Contents and rich leaf sections."""
    
    toc_prompt = """# Python 3.11 & FastAPI Production Guide
Chapter 1: Python Core Architecture
    leaf Section 1.1: Syntax, Operators & Control Flow
    leaf Section 1.2: Advanced Iterators, Generators & Context Managers
Chapter 2: Data Structures & Typing
    leaf Section 2.1: Built-in Collections & Time Complexities
    leaf Section 2.2: Structural Pattern Matching & Type Annotations
Chapter 3: Asynchronous Programming
    leaf Section 3.1: Asyncio Event Loop & Coroutine Mechanics
    leaf Section 3.2: Task Groups, Exception Handling & Deadlock Avoidance
Chapter 4: FastAPI Web Framework
    leaf Section 4.1: Routing, Path Parameters & Request Validation
    leaf Section 4.2: Dependency Injection & Database Session Lifecycles
    leaf Section 4.3: Background Tasks, Middleware & Security OAuth2
"""

    leaves: Dict[str, str] = {
        "Section 1.1: Syntax, Operators & Control Flow": (
            "Section 1.1: Python 3.11 Core Syntax and Flow Control\n\n"
            "Python 3.11 introduces specialized adaptive bytecode interpreters that boost performance by 10-60%.\n"
            "Control flow statements include 'if', 'elif', 'else', 'for', and 'while'.\n"
            "The walrus operator (:=) enables assignment expressions inside conditions:\n"
            "    if (n := len(items)) > 10:\n"
            "        print(f'Batch processing required for {n} items')\n\n"
            "Error locations are now highlighted with exact caret pointers in tracebacks, pointing directly "
            "to the offending token within nested expressions."
        ),
        
        "Section 1.2: Advanced Iterators, Generators & Context Managers": (
            "Section 1.2: Advanced Iterators and Context Managers\n\n"
            "Generators produce values lazily using the 'yield' keyword, preserving stack state without allocating memory for full collections:\n"
            "    def stream_large_dataset(file_path):\n"
            "        with open(file_path, 'r', encoding='utf-8') as f:\n"
            "            for line in f:\n"
            "                yield json.loads(line)\n\n"
            "Custom context managers implement __enter__() and __exit__(). Using @contextmanager from 'contextlib' provides "
            "clean resource cleanup and deterministic transaction lifecycles."
        ),

        "Section 2.1: Built-in Collections & Time Complexities": (
            "Section 2.1: Python Built-in Collections and Performance Characteristics\n\n"
            "1. Lists: Dynamic arrays with amortized O(1) append/pop, O(n) arbitrary insertion and slice assignment.\n"
            "2. Dictionaries: Hash maps with O(1) average lookup, insertion, and deletion. Preserves insertion order since Python 3.7.\n"
            "3. Sets: Hash tables storing unique elements with O(1) membership testing ('x in s') and O(len(s)+len(t)) union/intersection.\n"
            "4. Deques (collections.deque): Double-ended queues with guaranteed O(1) appends and pops from both ends, compared to O(n) for list.pop(0)."
        ),

        "Section 2.2: Structural Pattern Matching & Type Annotations": (
            "Section 2.2: Structural Pattern Matching and Static Type Checking\n\n"
            "Structural pattern matching (PEP 634) operates via the 'match' and 'case' statements:\n"
            "    match event:\n"
            "        case {'type': 'click', 'coordinates': (x, y)}:\n"
            "            handle_click(x, y)\n"
            "        case {'type': 'error', 'code': int(c)} if c >= 500:\n"
            "            log_critical_error(c)\n\n"
            "Type hints using 'typing.Annotated', 'Union' (or | operator in 3.10+), and 'TypeVar' allow Mypy and Pydantic "
            "to enforce compile-time and runtime data validation contracts."
        ),

        "Section 3.1: Asyncio Event Loop & Coroutine Mechanics": (
            "Section 3.1: Asyncio Event Loop & Cooperative Multitasking\n\n"
            "Asyncio runs a single-threaded cooperative multitasking event loop. Coroutines declared with 'async def' "
            "must be scheduled onto the loop via 'await', 'asyncio.create_task()', or TaskGroups.\n"
            "Calling blocking I/O functions (like time.sleep or synchronous requests.get) inside a coroutine blocks the entire event loop.\n"
            "To run blocking code without stalling the loop, delegate to a thread pool worker using:\n"
            "    result = await asyncio.to_thread(blocking_cpu_or_io_func, arg1)"
        ),

        "Section 3.2: Task Groups, Exception Handling & Deadlock Avoidance": (
            "Section 3.2: Structured Concurrency with TaskGroups\n\n"
            "Python 3.11 introduces 'asyncio.TaskGroup' as an asynchronous context manager providing structured concurrency:\n"
            "    async with asyncio.TaskGroup() as tg:\n"
            "        task1 = tg.create_task(fetch_orders())\n"
            "        task2 = tg.create_task(fetch_inventory())\n\n"
            "If any child task raises an exception, remaining tasks are promptly cancelled, and all errors are bundled into "
            "an 'ExceptionGroup', preventing orphaned tasks and resource leaks."
        ),

        "Section 4.1: Routing, Path Parameters & Request Validation": (
            "Section 4.1: FastAPI Routing and Pydantic Request Parsing\n\n"
            "FastAPI uses OpenAPI and Pydantic for high-performance schema serialization and deserialization:\n"
            "    from fastapi import FastAPI, Path, HTTPException\n"
            "    from pydantic import BaseModel, Field\n\n"
            "    class ItemPayload(BaseModel):\n"
            "        name: str = Field(..., min_length=2, max_length=50)\n"
            "        price: float = Field(..., gt=0.0)\n\n"
            "    @app.post('/items/{item_id}')\n"
            "    async def create_item(item_id: int = Path(..., ge=1), payload: ItemPayload = ...):\n"
            "        return {'id': item_id, 'data': payload.dict()}"
        ),

        "Section 4.2: Dependency Injection & Database Session Lifecycles": (
            "Section 4.2: Dependency Injection Pattern in FastAPI\n\n"
            "FastAPI's dependency injection system utilizes 'Depends()' to manage cross-cutting concerns:\n"
            "    async def get_db_session():\n"
            "        session = AsyncSessionLocal()\n"
            "        try:\n"
            "            yield session\n"
            "        finally:\n"
            "            await session.close()\n\n"
            "Using generator dependencies with 'yield' guarantees that database connections, telemetry spans, and "
            "authentication contexts are safely torn down even if exceptions occur during request handling."
        ),

        "Section 4.3: Background Tasks, Middleware & Security OAuth2": (
            "Section 4.3: Background Tasks, CORS Middleware & OAuth2 JWT Authentication\n\n"
            "Background tasks execute after returning HTTP responses:\n"
            "    @app.post('/notify')\n"
            "    async def notify_user(email: str, background_tasks: BackgroundTasks):\n"
            "        background_tasks.add_task(send_email_notification, email)\n"
            "        return {'status': 'notification queued'}\n\n"
            "Security utilities provide OAuth2 password bearer tokens with JWT validation, role-based access control (RBAC), "
            "and CORSMiddleware to prevent unauthorized cross-origin requests."
        )
    }

    return {
        "toc_prompt": toc_prompt,
        "leaves": leaves,
        "source": "synthetic_sample_python_fastapi",
        "num_leaves": len(leaves)
    }


def main():
    project_root = get_project_root()
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    sample_data = create_sample_toc_and_leaves()
    output_file = data_dir / "toc_and_sections.json"
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)
        
    print(f"✓ Created sample structured documentation with {len(sample_data['leaves'])} leaf sections.")
    print(f"✓ Saved to: {output_file}")


if __name__ == "__main__":
    main()
