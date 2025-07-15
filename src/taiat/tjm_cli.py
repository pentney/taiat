#!/usr/bin/env python3
"""
Command-line interface for Taiat job manager.

Usage:
    python tjm_cli.py create-job --name "job_name" --type agent --payload '{"agent_name": "my_agent"}'
    python tjm_cli.py list-jobs
    python tjm_cli.py get-job <job_id>
    python tjm_cli.py cancel-job <job_id>
    python tjm_cli.py history [--limit 10]
"""

import argparse
import json
import sys
from typing import Any, Dict, Optional

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from .init_worker import get_database_url, setup_logging, test_connection
from .job_manager import JobManager, JobPriority, JobStatus, JobType


def setup_job_manager(database_url: str) -> JobManager:
    """Setup job manager with database connection."""
    engine = create_engine(database_url)
    if not test_connection(engine):
        raise RuntimeError("Cannot connect to database")

    session_maker = sessionmaker(bind=engine)
    return JobManager(session_maker)


def create_job_command(args):
    """Create a new job."""
    try:
        job_manager = setup_job_manager(args.database_url)

        # Parse payload
        payload = {}
        if args.payload:
            payload = json.loads(args.payload)

        # Parse dependencies
        dependencies = []
        if args.dependencies:
            dependencies = args.dependencies.split(",")

        # Create job
        job = job_manager.create_job(
            job_type=JobType(args.job_type),
            name=args.name,
            description=args.description,
            payload=payload,
            dependencies=dependencies,
            priority=JobPriority(args.priority),
            max_retries=args.max_retries,
        )

        print(f"✅ Created job: {job.id}")
        print(f"   Name: {job.name}")
        print(f"   Type: {job.job_type.value}")
        print(f"   Status: {job.status.value}")
        print(f"   Priority: {job.priority.name}")
        if dependencies:
            print(f"   Dependencies: {', '.join(dependencies)}")

    except Exception as e:
        print(f"❌ Error creating job: {e}")
        sys.exit(1)


def list_jobs_command(args):
    """List jobs."""
    try:
        job_manager = setup_job_manager(args.database_url)

        if args.status:
            jobs = job_manager.get_pending_jobs()
            if args.status != "pending":
                # For now, just show pending jobs
                print(
                    f"Note: Only pending jobs are shown. Use --status pending for all pending jobs."
                )
        else:
            jobs = job_manager.get_pending_jobs()

        if not jobs:
            print("No jobs found.")
            return

        print(f"\nFound {len(jobs)} jobs:")
        print("-" * 80)
        print(f"{'ID':<36} {'Name':<20} {'Type':<8} {'Status':<10} {'Priority':<8}")
        print("-" * 80)

        for job in jobs:
            print(
                f"{job.id:<36} {job.name:<20} {job.job_type.value:<8} {job.status.value:<10} {job.priority.name:<8}"
            )

        if args.ready_only:
            ready_jobs = job_manager.get_ready_jobs()
            print(f"\nReady jobs (dependencies satisfied): {len(ready_jobs)}")

    except Exception as e:
        print(f"❌ Error listing jobs: {e}")
        sys.exit(1)


def get_job_command(args):
    """Get job details."""
    try:
        job_manager = setup_job_manager(args.database_url)

        job = job_manager.get_job(args.job_id)
        if not job:
            print(f"❌ Job {args.job_id} not found")
            sys.exit(1)

        print(f"\nJob Details:")
        print(f"  ID: {job.id}")
        print(f"  Name: {job.name}")
        print(f"  Type: {job.job_type.value}")
        print(f"  Status: {job.status.value}")
        print(f"  Priority: {job.priority.name}")
        print(f"  Description: {job.description or 'N/A'}")
        print(
            f"  Dependencies: {', '.join(job.dependencies) if job.dependencies else 'None'}"
        )
        print(f"  Retry Count: {job.retry_count}/{job.max_retries}")
        print(f"  Created: {job.created_at}")
        print(f"  Updated: {job.updated_at}")

        if job.started_at:
            print(f"  Started: {job.started_at}")
        if job.completed_at:
            print(f"  Completed: {job.completed_at}")

        if job.error_message:
            print(f"  Error: {job.error_message}")

        if job.result:
            print(f"  Result: {json.dumps(job.result, indent=2)}")

        if job.payload:
            print(f"  Payload: {json.dumps(job.payload, indent=2)}")

    except Exception as e:
        print(f"❌ Error getting job: {e}")
        sys.exit(1)


def cancel_job_command(args):
    """Cancel a job."""
    try:
        job_manager = setup_job_manager(args.database_url)

        job = job_manager.get_job(args.job_id)
        if not job:
            print(f"❌ Job {args.job_id} not found")
            sys.exit(1)

        if job.status.value in ["completed", "failed", "cancelled"]:
            print(f"❌ Cannot cancel job in {job.status.value} status")
            sys.exit(1)

        job_manager.update_job_status(args.job_id, JobStatus.CANCELLED)
        print(f"✅ Cancelled job: {args.job_id}")

    except Exception as e:
        print(f"❌ Error cancelling job: {e}")
        sys.exit(1)


def history_command(args):
    """Show job history."""
    try:
        job_manager = setup_job_manager(args.database_url)

        history = job_manager.get_job_history(limit=args.limit)

        if not history:
            print("No job history found.")
            return

        print(f"\nJob History (last {len(history)} entries):")
        print("-" * 100)
        print(
            f"{'Job ID':<36} {'Name':<20} {'Type':<8} {'Status':<10} {'Duration':<10} {'Created':<20}"
        )
        print("-" * 100)

        for entry in history:
            duration = (
                f"{entry.duration_seconds:.1f}s" if entry.duration_seconds else "N/A"
            )
            created = (
                entry.created_at.strftime("%Y-%m-%d %H:%M:%S")
                if entry.created_at
                else "N/A"
            )

            status_icon = "✅" if entry.status.value == "completed" else "❌"

            print(
                f"{entry.job_id:<36} {entry.name:<20} {entry.job_type.value:<8} {status_icon} {entry.status.value:<8} {duration:<10} {created:<20}"
            )

    except Exception as e:
        print(f"❌ Error showing history: {e}")
        sys.exit(1)


def status_command(args):
    """Show worker status."""
    try:
        job_manager = setup_job_manager(args.database_url)

        pending_jobs = job_manager.get_pending_jobs()
        ready_jobs = job_manager.get_ready_jobs()
        history = job_manager.get_job_history(limit=5)

        print("Taiat Job Manager Status")
        print("=" * 40)
        print(f"Pending jobs: {len(pending_jobs)}")
        print(f"Ready jobs: {len(ready_jobs)}")
        print(f"Recent history: {len(history)} entries")

        if ready_jobs:
            print(f"\nReady jobs:")
            for job in ready_jobs[:5]:
                print(f"  - {job.id}: {job.name} (priority: {job.priority.name})")

        if history:
            print(f"\nRecent history:")
            for entry in history:
                status_icon = "✅" if entry.status.value == "completed" else "❌"
                print(
                    f"  {status_icon} {entry.job_id}: {entry.name} ({entry.status.value})"
                )

    except Exception as e:
        print(f"❌ Error getting status: {e}")
        sys.exit(1)


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Taiat Job Manager CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create a simple agent job
  python cli.py create-job --name "data_processing" --type agent --payload '{"agent_name": "data_processor"}'
  
  # Create a query job with dependencies
  python cli.py create-job --name "analysis" --type query --payload '{"query": "Analyze data"}' --dependencies "job_id_1,job_id_2"
  
  # List all pending jobs
  python cli.py list-jobs
  
  # Get job details
  python cli.py get-job <job_id>
  
  # Show job history
  python cli.py history --limit 20
        """,
    )

    parser.add_argument(
        "--database-url",
        help="Database connection URL (defaults to TAIAT_DATABASE_URL env var or localhost/taiat)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Create job command
    create_parser = subparsers.add_parser("create-job", help="Create a new job")
    create_parser.add_argument("--name", required=True, help="Job name")
    create_parser.add_argument(
        "--type", required=True, choices=["query", "agent"], help="Job type"
    )
    create_parser.add_argument("--description", help="Job description")
    create_parser.add_argument("--payload", help="Job payload (JSON string)")
    create_parser.add_argument("--dependencies", help="Comma-separated list of job IDs")
    create_parser.add_argument(
        "--priority",
        type=int,
        default=2,
        choices=[1, 2, 3, 4],
        help="Job priority (1=low, 4=urgent)",
    )
    create_parser.add_argument(
        "--max-retries", type=int, default=3, help="Maximum retry attempts"
    )
    create_parser.set_defaults(func=create_job_command)

    # List jobs command
    list_parser = subparsers.add_parser("list-jobs", help="List jobs")
    list_parser.add_argument("--status", help="Filter by status")
    list_parser.add_argument(
        "--ready-only", action="store_true", help="Show only ready jobs"
    )
    list_parser.set_defaults(func=list_jobs_command)

    # Get job command
    get_parser = subparsers.add_parser("get-job", help="Get job details")
    get_parser.add_argument("job_id", help="Job ID")
    get_parser.set_defaults(func=get_job_command)

    # Cancel job command
    cancel_parser = subparsers.add_parser("cancel-job", help="Cancel a job")
    cancel_parser.add_argument("job_id", help="Job ID")
    cancel_parser.set_defaults(func=cancel_job_command)

    # History command
    history_parser = subparsers.add_parser("history", help="Show job history")
    history_parser.add_argument(
        "--limit", type=int, default=10, help="Number of entries to show"
    )
    history_parser.set_defaults(func=history_command)

    # Status command
    status_parser = subparsers.add_parser("status", help="Show worker status")
    status_parser.set_defaults(func=status_command)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Get database URL
    args.database_url = args.database_url or get_database_url()

    # Setup logging
    setup_logging("INFO")

    # Execute command
    args.func(args)


if __name__ == "__main__":
    main()
