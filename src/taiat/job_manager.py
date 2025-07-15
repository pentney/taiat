import enum
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from sqlalchemy import (
    Table,
    Column,
    DateTime,
    Integer,
    String,
    Text,
    MetaData,
    insert,
    select,
    update,
    delete,
    func,
    JSON,
)
from sqlalchemy.orm import sessionmaker

from .base import TaiatQuery


class JobType(str, enum.Enum):
    """Types of jobs that can be executed."""

    QUERY = "query"
    AGENT = "agent"


class JobStatus(str, enum.Enum):
    """Status of a job."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobPriority(int, enum.Enum):
    """Priority levels for jobs."""

    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


class Job(BaseModel):
    """Represents a job to be executed."""

    id: Optional[str] = None
    job_type: JobType
    name: str
    description: Optional[str] = None
    payload: Dict[str, Any] = Field(default_factory=dict)
    dependencies: List[str] = Field(default_factory=list)  # List of job IDs
    priority: JobPriority = JobPriority.NORMAL
    max_retries: int = 3
    retry_count: int = 0
    status: JobStatus = JobStatus.PENDING
    error_message: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage."""
        return {
            "id": self.id,
            "job_type": self.job_type.value,
            "name": self.name,
            "description": self.description,
            "payload": self.payload,
            "dependencies": self.dependencies,
            "priority": self.priority.value,
            "max_retries": self.max_retries,
            "retry_count": self.retry_count,
            "status": self.status.value,
            "error_message": self.error_message,
            "result": self.result,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Job":
        """Create Job from dictionary."""
        return cls(
            id=data.get("id"),
            job_type=JobType(data["job_type"]),
            name=data["name"],
            description=data.get("description"),
            payload=data.get("payload", {}),
            dependencies=data.get("dependencies", []),
            priority=JobPriority(data.get("priority", JobPriority.NORMAL.value)),
            max_retries=data.get("max_retries", 3),
            retry_count=data.get("retry_count", 0),
            status=JobStatus(data["status"]),
            error_message=data.get("error_message"),
            result=data.get("result"),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
        )


class JobHistory(BaseModel):
    """Represents a completed job run."""

    id: Optional[str] = None
    job_id: str
    job_type: JobType
    name: str
    description: Optional[str] = None
    payload: Dict[str, Any] = Field(default_factory=dict)
    dependencies: List[str] = Field(default_factory=list)
    priority: JobPriority
    status: JobStatus
    error_message: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    created_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage."""
        return {
            "id": self.id,
            "job_id": self.job_id,
            "job_type": self.job_type.value,
            "name": self.name,
            "description": self.description,
            "payload": self.payload,
            "dependencies": self.dependencies,
            "priority": self.priority.value,
            "status": self.status.value,
            "error_message": self.error_message,
            "result": self.result,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_seconds": self.duration_seconds,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JobHistory":
        """Create JobHistory from dictionary."""
        return cls(
            id=data.get("id"),
            job_id=data["job_id"],
            job_type=JobType(data["job_type"]),
            name=data["name"],
            description=data.get("description"),
            payload=data.get("payload", {}),
            dependencies=data.get("dependencies", []),
            priority=JobPriority(data.get("priority", JobPriority.NORMAL.value)),
            status=JobStatus(data["status"]),
            error_message=data.get("error_message"),
            result=data.get("result"),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            duration_seconds=data.get("duration_seconds"),
            created_at=data.get("created_at"),
        )


# Database tables
metadata = MetaData()

jobs_table = Table(
    "taiat_jobs",
    metadata,
    Column("id", String, primary_key=True),
    Column("job_type", String, nullable=False),
    Column("name", String, nullable=False),
    Column("description", Text),
    Column("payload", JSON, nullable=False, default={}),
    Column("dependencies", JSON, nullable=False, default=[]),
    Column("priority", Integer, nullable=False, default=JobPriority.NORMAL.value),
    Column("max_retries", Integer, nullable=False, default=3),
    Column("retry_count", Integer, nullable=False, default=0),
    Column("status", String, nullable=False, default=JobStatus.PENDING.value),
    Column("error_message", Text),
    Column("result", JSON),
    Column("started_at", DateTime),
    Column("completed_at", DateTime),
    Column("created_at", DateTime, server_default=func.now()),
    Column("updated_at", DateTime, server_default=func.now(), onupdate=func.now()),
)

job_history_table = Table(
    "taiat_job_history",
    metadata,
    Column("id", String, primary_key=True),
    Column("job_id", String, nullable=False),
    Column("job_type", String, nullable=False),
    Column("name", String, nullable=False),
    Column("description", Text),
    Column("payload", JSON, nullable=False, default={}),
    Column("dependencies", JSON, nullable=False, default=[]),
    Column("priority", Integer, nullable=False),
    Column("status", String, nullable=False),
    Column("error_message", Text),
    Column("result", JSON),
    Column("started_at", DateTime),
    Column("completed_at", DateTime),
    Column("duration_seconds", Integer),
    Column("created_at", DateTime, server_default=func.now()),
)


class JobManager:
    """Manages jobs in the database."""

    def __init__(self, session_maker: sessionmaker):
        self.session_maker = session_maker

    def create_job(
        self,
        job_type: JobType,
        name: str,
        payload: Dict[str, Any],
        dependencies: Optional[List[str]] = None,
        priority: JobPriority = JobPriority.NORMAL,
        max_retries: int = 3,
        description: Optional[str] = None,
    ) -> Job:
        """Create a new job."""
        job_id = str(uuid.uuid4())
        job = Job(
            id=job_id,
            job_type=job_type,
            name=name,
            description=description,
            payload=payload,
            dependencies=dependencies or [],
            priority=priority,
            max_retries=max_retries,
        )

        with self.session_maker() as session:
            stmt = insert(jobs_table).values(job.to_dict())
            session.execute(stmt)
            session.commit()

        return job

    def get_job(self, job_id: str) -> Optional[Job]:
        """Get a job by ID."""
        with self.session_maker() as session:
            stmt = select(jobs_table).where(jobs_table.c.id == job_id)
            result = session.execute(stmt).first()

            if result:
                return Job.from_dict(dict(result._mapping))
            return None

    def get_pending_jobs(self) -> List[Job]:
        """Get all pending jobs."""
        with self.session_maker() as session:
            stmt = (
                select(jobs_table)
                .where(jobs_table.c.status == JobStatus.PENDING.value)
                .order_by(jobs_table.c.priority.desc(), jobs_table.c.created_at.asc())
            )
            results = session.execute(stmt).fetchall()

            return [Job.from_dict(dict(result._mapping)) for result in results]

    def get_ready_jobs(self) -> List[Job]:
        """Get jobs that are ready to run (dependencies satisfied)."""
        with self.session_maker() as session:
            # Get all pending jobs
            stmt = select(jobs_table).where(
                jobs_table.c.status == JobStatus.PENDING.value
            )
            pending_jobs = session.execute(stmt).fetchall()

            ready_jobs = []
            for job_row in pending_jobs:
                job = Job.from_dict(dict(job_row._mapping))

                # Check if all dependencies are completed
                if not job.dependencies:
                    ready_jobs.append(job)
                    continue

                # Check dependency status
                deps_stmt = select(jobs_table.c.status).where(
                    jobs_table.c.id.in_(job.dependencies)
                )
                dep_statuses = session.execute(deps_stmt).fetchall()

                all_completed = all(
                    status[0] == JobStatus.COMPLETED.value for status in dep_statuses
                )

                if all_completed:
                    ready_jobs.append(job)

            # Sort by priority and creation time
            ready_jobs.sort(
                key=lambda j: (j.priority.value, j.created_at or datetime.min),
                reverse=True,
            )

            return ready_jobs

    def update_job_status(
        self,
        job_id: str,
        status: JobStatus,
        error_message: Optional[str] = None,
        result: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update job status."""
        with self.session_maker() as session:
            update_data = {
                "status": status.value,
                "updated_at": datetime.utcnow(),
            }

            if status == JobStatus.RUNNING:
                update_data["started_at"] = datetime.utcnow()
            elif status in [JobStatus.COMPLETED, JobStatus.FAILED]:
                update_data["completed_at"] = datetime.utcnow()

            if error_message is not None:
                update_data["error_message"] = error_message

            if result is not None:
                update_data["result"] = result

            stmt = (
                update(jobs_table).where(jobs_table.c.id == job_id).values(update_data)
            )
            session.execute(stmt)
            session.commit()

    def increment_retry_count(self, job_id: str) -> None:
        """Increment the retry count for a job."""
        with self.session_maker() as session:
            stmt = (
                update(jobs_table)
                .where(jobs_table.c.id == job_id)
                .values(
                    retry_count=jobs_table.c.retry_count + 1,
                    updated_at=datetime.utcnow(),
                )
            )
            session.execute(stmt)
            session.commit()

    def move_to_history(self, job: Job) -> None:
        """Move a completed job to history."""
        # Calculate duration
        duration_seconds = None
        if job.started_at and job.completed_at:
            duration_seconds = (job.completed_at - job.started_at).total_seconds()

        history = JobHistory(
            job_id=job.id,
            job_type=job.job_type,
            name=job.name,
            description=job.description,
            payload=job.payload,
            dependencies=job.dependencies,
            priority=job.priority,
            status=job.status,
            error_message=job.error_message,
            result=job.result,
            started_at=job.started_at,
            completed_at=job.completed_at,
            duration_seconds=duration_seconds,
        )

        with self.session_maker() as session:
            # Insert into history
            stmt = insert(job_history_table).values(history.to_dict())
            session.execute(stmt)

            # Delete from jobs table
            stmt = delete(jobs_table).where(jobs_table.c.id == job.id)
            session.execute(stmt)

            session.commit()

    def get_job_history(self, limit: int = 100) -> List[JobHistory]:
        """Get recent job history."""
        with self.session_maker() as session:
            stmt = (
                select(job_history_table)
                .order_by(job_history_table.c.created_at.desc())
                .limit(limit)
            )
            results = session.execute(stmt).fetchall()

            return [JobHistory.from_dict(dict(result._mapping)) for result in results]

    def cleanup_old_history(self, days: int = 30) -> int:
        """Clean up old job history entries."""
        cutoff_date = datetime.utcnow() - timedelta(days=days)

        with self.session_maker() as session:
            stmt = delete(job_history_table).where(
                job_history_table.c.created_at < cutoff_date
            )
            result = session.execute(stmt)
            session.commit()

            return result.rowcount
