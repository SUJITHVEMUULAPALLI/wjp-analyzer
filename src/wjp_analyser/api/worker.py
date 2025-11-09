"""
RQ Worker Script
================

Run this script to start RQ workers that process background jobs.

Usage:
    python -m wjp_analyser.api.worker
    python -m wjp_analyser.api.worker --queues analysis,conversion
    python -m wjp_analyser.api.worker --burst
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path

# Add src to path
_THIS_DIR = Path(__file__).parent
_SRC_DIR = _THIS_DIR.parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

try:
    from rq import Worker, Queue, Connection
    from wjp_analyser.api.queue_manager import (
        get_redis_connection,
        QUEUE_DEFAULT,
        QUEUE_ANALYSIS,
        QUEUE_CONVERSION,
        QUEUE_NESTING,
        QUEUE_GCODE,
        is_redis_available,
    )
    RQ_AVAILABLE = True
except ImportError:
    print("Error: RQ and redis are required. Install with: pip install rq redis")
    sys.exit(1)


def main():
    """Main worker function."""
    parser = argparse.ArgumentParser(description="Start RQ worker for WJP ANALYSER")
    parser.add_argument(
        "--queues",
        type=str,
        default=QUEUE_DEFAULT,
        help="Comma-separated list of queue names (default: default)",
    )
    parser.add_argument(
        "--burst",
        action="store_true",
        help="Run in burst mode (exit when no jobs available)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Worker name (default: auto-generated)",
    )
    
    args = parser.parse_args()
    
    # Check Redis availability
    if not is_redis_available():
        print("Error: Redis is not available. Please start Redis server.")
        print(f"  Install: https://redis.io/docs/getting-started/installation/")
        print(f"  Or use Docker: docker run -d -p 6379:6379 redis")
        sys.exit(1)
    
    # Parse queue names
    queue_names = [q.strip() for q in args.queues.split(",")]
    
    # Validate queue names
    valid_queues = {QUEUE_DEFAULT, QUEUE_ANALYSIS, QUEUE_CONVERSION, QUEUE_NESTING, QUEUE_GCODE}
    invalid = [q for q in queue_names if q not in valid_queues]
    if invalid:
        print(f"Warning: Invalid queue names: {invalid}")
        print(f"Valid queues: {', '.join(sorted(valid_queues))}")
        queue_names = [q for q in queue_names if q in valid_queues]
    
    if not queue_names:
        print("Error: No valid queues specified")
        sys.exit(1)
    
    # Get Redis connection
    redis_conn = get_redis_connection()
    
    # Create queues
    queues = [Queue(q, connection=redis_conn) for q in queue_names]
    
    print(f"Starting worker for queues: {', '.join(queue_names)}")
    if args.burst:
        print("Running in burst mode")
    if args.name:
        print(f"Worker name: {args.name}")
    
    # Start worker
    with Connection(redis_conn):
        worker = Worker(
            queues,
            name=args.name,
        )
        worker.work(burst=args.burst)


if __name__ == "__main__":
    main()








