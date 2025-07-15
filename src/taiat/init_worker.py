#!/usr/bin/env python3
"""
Initialize the Taiat job manager database tables.

This script creates the necessary tables for the job management system:
- taiat_jobs: Table for pending and running jobs
- taiat_job_history: Table for completed jobs

Usage:
    python init_worker.py [--database-url DATABASE_URL]
"""

import argparse
import logging
import os
import sys
from typing import Optional

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from .db import metadata as taiat_metadata
from .job_manager import metadata as job_metadata


logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )


def get_database_url() -> str:
    """Get database URL from environment or default."""
    # Check environment variable first
    db_url = os.getenv("TAIAT_DATABASE_URL")
    if db_url:
        return db_url

    # Check for common environment variables
    for env_var in ["DATABASE_URL", "POSTGRES_URL", "PG_URL"]:
        db_url = os.getenv(env_var)
        if db_url:
            return db_url

    # Default to local PostgreSQL
    return "postgresql://localhost/taiat"


def test_connection(engine) -> bool:
    """Test database connection."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False


def create_tables(engine, metadata, table_names: list[str]):
    """Create database tables."""
    try:
        # Check if tables already exist
        existing_tables = []
        with engine.connect() as conn:
            result = conn.execute(
                text(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = 'public'"
                )
            )
            existing_tables = [row[0] for row in result]

        # Create tables that don't exist
        tables_to_create = []
        for table_name in table_names:
            if table_name not in existing_tables:
                tables_to_create.append(table_name)

        if tables_to_create:
            logger.info(f"Creating tables: {', '.join(tables_to_create)}")
            metadata.create_all(
                engine,
                tables=[
                    table
                    for table in metadata.tables.values()
                    if table.name in tables_to_create
                ],
            )
            logger.info("Tables created successfully")
        else:
            logger.info("All tables already exist")

        return True

    except Exception as e:
        logger.error(f"Error creating tables: {e}")
        return False


def create_indexes(engine):
    """Create useful indexes for job management."""
    try:
        with engine.connect() as conn:
            # Indexes for jobs table
            indexes = [
                # Index for finding ready jobs (pending status)
                """
                CREATE INDEX IF NOT EXISTS idx_taiat_jobs_status_priority 
                ON taiat_jobs (status, priority DESC, created_at ASC)
                """,
                # Index for finding jobs by dependencies
                """
                CREATE INDEX IF NOT EXISTS idx_taiat_jobs_dependencies 
                ON taiat_jobs USING GIN (dependencies)
                """,
                # Index for job history cleanup
                """
                CREATE INDEX IF NOT EXISTS idx_taiat_job_history_created_at 
                ON taiat_job_history (created_at DESC)
                """,
                # Index for finding job history by job_id
                """
                CREATE INDEX IF NOT EXISTS idx_taiat_job_history_job_id 
                ON taiat_job_history (job_id)
                """,
            ]

            for index_sql in indexes:
                try:
                    conn.execute(text(index_sql))
                    conn.commit()
                except Exception as e:
                    # Index might already exist, that's okay
                    logger.debug(f"Index creation (may already exist): {e}")

            logger.info("Indexes created/verified successfully")

        return True

    except Exception as e:
        logger.error(f"Error creating indexes: {e}")
        return False


def verify_tables(engine) -> bool:
    """Verify that all required tables exist."""
    try:
        required_tables = [
            "taiat_jobs",
            "taiat_job_history",
            "taiat_query",
            "taiat_query_data",
        ]

        with engine.connect() as conn:
            result = conn.execute(
                text(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = 'public' AND table_name = ANY(:tables)"
                ),
                {"tables": required_tables},
            )

            existing_tables = {row[0] for row in result}
            missing_tables = set(required_tables) - existing_tables

        if missing_tables:
            logger.error(f"Missing tables: {', '.join(missing_tables)}")
            return False

        logger.info("All required tables exist")
        return True

    except Exception as e:
        logger.error(f"Error verifying tables: {e}")
        return False


def main():
    """Main initialization function."""
    parser = argparse.ArgumentParser(
        description="Initialize Taiat job manager database tables"
    )
    parser.add_argument(
        "--database-url",
        help="Database connection URL (defaults to TAIAT_DATABASE_URL env var or localhost/taiat)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--skip-indexes", action="store_true", help="Skip creating indexes"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify tables exist, don't create them",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)

    # Get database URL
    database_url = args.database_url or get_database_url()
    logger.info(f"Using database: {database_url}")

    # Create engine
    try:
        engine = create_engine(database_url)
    except Exception as e:
        logger.error(f"Failed to create database engine: {e}")
        sys.exit(1)

    # Test connection
    if not test_connection(engine):
        logger.error(
            "Cannot connect to database. Please check your connection settings."
        )
        sys.exit(1)

    if args.verify_only:
        # Only verify tables exist
        if verify_tables(engine):
            logger.info("Database verification successful")
            sys.exit(0)
        else:
            logger.error("Database verification failed")
            sys.exit(1)

    # Create Taiat tables (if they don't exist)
    logger.info("Creating Taiat tables...")
    if not create_tables(engine, taiat_metadata, ["taiat_query", "taiat_query_data"]):
        logger.error("Failed to create Taiat tables")
        sys.exit(1)

    # Create job manager tables
    logger.info("Creating job manager tables...")
    if not create_tables(engine, job_metadata, ["taiat_jobs", "taiat_job_history"]):
        logger.error("Failed to create job manager tables")
        sys.exit(1)

    # Create indexes (unless skipped)
    if not args.skip_indexes:
        logger.info("Creating indexes...")
        if not create_indexes(engine):
            logger.warning("Failed to create some indexes, but continuing...")

    # Verify all tables exist
    if not verify_tables(engine):
        logger.error("Table verification failed")
        sys.exit(1)

    logger.info("Database initialization completed successfully!")
    logger.info("You can now start the worker with: python -m taiat.run_worker")


if __name__ == "__main__":
    main()
