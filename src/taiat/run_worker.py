#!/usr/bin/env python3
"""
Run the Taiat job worker.

This script starts a worker that monitors the job queue and executes jobs
when their dependencies are satisfied.

Usage:
    python run_worker.py [--database-url DATABASE_URL] [--poll-interval SECONDS]
"""

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
from typing import Any, Callable, Dict

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from .init_worker import get_database_url, setup_logging, test_connection, verify_tables
from .worker import TaiatWorker


logger = logging.getLogger(__name__)


def load_taiat_engine():
    """Load the Taiat engine if available."""
    try:
        # Try to import the Taiat engine
        from .builder import TaiatBuilder

        # Create a basic engine - you might want to customize this
        builder = TaiatBuilder()
        engine = builder.build()

        logger.info("Taiat engine loaded successfully")
        return engine

    except ImportError as e:
        logger.warning(f"Could not load Taiat engine: {e}")
        logger.warning("Query jobs will not be supported")
        return None
    except Exception as e:
        logger.warning(f"Error loading Taiat engine: {e}")
        logger.warning("Query jobs will not be supported")
        return None


def load_agent_registry() -> Dict[str, Callable]:
    """Load agent registry from environment or configuration."""
    agents = {}

    # You can extend this to load agents from configuration files
    # or environment variables

    # Example: Load from environment variable
    agent_config = os.getenv("TAIAT_AGENT_CONFIG")
    if agent_config:
        try:
            config = json.loads(agent_config)
            # This would need to be implemented based on your agent loading mechanism
            logger.info(f"Loaded agent configuration: {list(config.keys())}")
        except Exception as e:
            logger.warning(f"Failed to load agent configuration: {e}")

    # Example agents (replace with your actual agents)
    def example_agent(**kwargs):
        """Example agent function."""
        logger.info(f"Running example agent with params: {kwargs}")
        return {"result": "example_agent_completed", "params": kwargs}

    agents["example_agent"] = example_agent

    logger.info(f"Loaded {len(agents)} agents")
    return agents


def setup_signal_handlers(worker: TaiatWorker):
    """Setup signal handlers for graceful shutdown."""

    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        worker.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


async def run_worker(
    database_url: str,
    poll_interval: float = 5.0,
    max_concurrent_jobs: int = 4,
    cleanup_interval_hours: int = 24,
    log_level: str = "INFO",
):
    """Run the worker."""
    # Setup logging
    setup_logging(log_level)

    logger.info("Starting Taiat worker...")
    logger.info(f"Database URL: {database_url}")
    logger.info(f"Poll interval: {poll_interval}s")
    logger.info(f"Max concurrent jobs: {max_concurrent_jobs}")

    # Create database engine
    try:
        engine = create_engine(database_url)
    except Exception as e:
        logger.error(f"Failed to create database engine: {e}")
        return False

    # Test connection
    if not test_connection(engine):
        logger.error("Cannot connect to database")
        return False

    # Verify tables exist
    if not verify_tables(engine):
        logger.error("Required tables do not exist. Run init_worker.py first.")
        return False

    # Create session maker
    session_maker = sessionmaker(bind=engine)

    # Load Taiat engine
    taiat_engine = load_taiat_engine()

    # Load agent registry
    agent_registry = load_agent_registry()

    # Create worker
    worker = TaiatWorker(
        session_maker=session_maker,
        taiat_engine=taiat_engine,
        agent_registry=agent_registry,
        poll_interval=poll_interval,
        max_concurrent_jobs=max_concurrent_jobs,
        cleanup_interval_hours=cleanup_interval_hours,
    )

    # Setup signal handlers
    setup_signal_handlers(worker)

    # Start the worker
    try:
        await worker.start()
        return True
    except KeyboardInterrupt:
        logger.info("Worker interrupted by user")
        return True
    except Exception as e:
        logger.error(f"Worker error: {e}")
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run the Taiat job worker")
    parser.add_argument(
        "--database-url",
        help="Database connection URL (defaults to TAIAT_DATABASE_URL env var or localhost/taiat)",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=5.0,
        help="Poll interval in seconds (default: 5.0)",
    )
    parser.add_argument(
        "--max-concurrent-jobs",
        type=int,
        default=4,
        help="Maximum number of concurrent jobs (default: 4)",
    )
    parser.add_argument(
        "--cleanup-interval-hours",
        type=int,
        default=24,
        help="Cleanup interval in hours (default: 24)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--status", action="store_true", help="Show worker status and exit"
    )

    args = parser.parse_args()

    # Get database URL
    database_url = args.database_url or get_database_url()

    if args.status:
        # Show status and exit
        try:
            engine = create_engine(database_url)
            session_maker = sessionmaker(bind=engine)

            from .job_manager import JobManager

            job_manager = JobManager(session_maker)

            # Get job counts
            pending_jobs = job_manager.get_pending_jobs()
            ready_jobs = job_manager.get_ready_jobs()
            history = job_manager.get_job_history(limit=10)

            print("Taiat Worker Status")
            print("==================")
            print(f"Database: {database_url}")
            print(f"Pending jobs: {len(pending_jobs)}")
            print(f"Ready jobs: {len(ready_jobs)}")
            print(f"Recent history: {len(history)} entries")

            if ready_jobs:
                print("\nReady jobs:")
                for job in ready_jobs[:5]:  # Show first 5
                    print(f"  - {job.id}: {job.name} (priority: {job.priority.name})")

            if history:
                print("\nRecent job history:")
                for entry in history[:5]:  # Show first 5
                    status_icon = "✓" if entry.status == "completed" else "✗"
                    print(
                        f"  {status_icon} {entry.job_id}: {entry.name} ({entry.status})"
                    )

        except Exception as e:
            print(f"Error getting status: {e}")
            sys.exit(1)

        sys.exit(0)

    # Run the worker
    success = asyncio.run(
        run_worker(
            database_url=database_url,
            poll_interval=args.poll_interval,
            max_concurrent_jobs=args.max_concurrent_jobs,
            cleanup_interval_hours=args.cleanup_interval_hours,
            log_level=args.log_level,
        )
    )

    if success:
        logger.info("Worker stopped successfully")
        sys.exit(0)
    else:
        logger.error("Worker failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
