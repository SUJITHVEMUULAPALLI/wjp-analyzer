#!/usr/bin/env python3
"""
Celery Worker Management Script
==============================

Script for managing Celery workers and monitoring.
"""

import os
import sys
import subprocess
import argparse
import time
import signal
from typing import List, Dict, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from wjp_analyser.workers.celery_app import celery_app
from wjp_analyser.config.unified_config_manager import get_config


def start_worker(queues: List[str] = None, concurrency: int = 4, loglevel: str = 'info'):
    """Start Celery worker."""
    if queues is None:
        queues = ['analysis', 'conversion', 'nesting', 'batch', 'maintenance']
    
    queue_str = ','.join(queues)
    
    cmd = [
        'celery',
        '-A', 'src.wjp_analyser.workers.celery_app',
        'worker',
        '--loglevel', loglevel,
        '--concurrency', str(concurrency),
        '--queues', queue_str,
        '--hostname', f'wjp-worker@%h'
    ]
    
    print(f"Starting Celery worker with queues: {queue_str}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nWorker stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"Worker failed to start: {e}")


def start_beat():
    """Start Celery beat scheduler."""
    cmd = [
        'celery',
        '-A', 'src.wjp_analyser.workers.celery_app',
        'beat',
        '--loglevel', 'info'
    ]
    
    print("Starting Celery beat scheduler")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nBeat scheduler stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"Beat scheduler failed to start: {e}")


def start_flower():
    """Start Flower monitoring."""
    cmd = [
        'celery',
        '-A', 'src.wjp_analyser.workers.celery_app',
        'flower',
        '--port', '5555'
    ]
    
    print("Starting Flower monitoring on http://localhost:5555")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nFlower stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"Flower failed to start: {e}")


def get_worker_stats():
    """Get worker statistics."""
    try:
        inspect = celery_app.control.inspect()
        stats = inspect.stats()
        active = inspect.active()
        scheduled = inspect.scheduled()
        
        print("=== Celery Worker Statistics ===")
        print(f"Active workers: {len(stats) if stats else 0}")
        
        if stats:
            for worker, worker_stats in stats.items():
                print(f"\nWorker: {worker}")
                print(f"  Status: {'Active' if worker in active else 'Inactive'}")
                print(f"  Pool: {worker_stats.get('pool', {}).get('implementation', 'Unknown')}")
                print(f"  Processes: {worker_stats.get('pool', {}).get('processes', 'Unknown')}")
                print(f"  Max concurrency: {worker_stats.get('pool', {}).get('max-concurrency', 'Unknown')}")
        
        if active:
            print(f"\nActive tasks: {len(active)}")
            for worker, tasks in active.items():
                print(f"  {worker}: {len(tasks)} tasks")
        
        if scheduled:
            print(f"\nScheduled tasks: {len(scheduled)}")
            for worker, tasks in scheduled.items():
                print(f"  {worker}: {len(tasks)} tasks")
        
    except Exception as e:
        print(f"Failed to get worker stats: {e}")


def purge_queues():
    """Purge all task queues."""
    try:
        celery_app.control.purge()
        print("All task queues purged successfully")
    except Exception as e:
        print(f"Failed to purge queues: {e}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Celery Worker Management')
    parser.add_argument('command', choices=['start', 'beat', 'flower', 'stats', 'purge'],
                       help='Command to execute')
    parser.add_argument('--queues', nargs='+', 
                       choices=['analysis', 'conversion', 'nesting', 'batch', 'maintenance'],
                       help='Queues to process')
    parser.add_argument('--concurrency', type=int, default=4,
                       help='Number of worker processes')
    parser.add_argument('--loglevel', choices=['debug', 'info', 'warning', 'error'],
                       default='info', help='Log level')
    
    args = parser.parse_args()
    
    if args.command == 'start':
        start_worker(args.queues, args.concurrency, args.loglevel)
    elif args.command == 'beat':
        start_beat()
    elif args.command == 'flower':
        start_flower()
    elif args.command == 'stats':
        get_worker_stats()
    elif args.command == 'purge':
        purge_queues()


if __name__ == '__main__':
    main()
