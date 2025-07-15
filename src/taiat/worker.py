import asyncio
import logging
import signal
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from sqlalchemy.orm import sessionmaker

from .base import TaiatQuery
from .job_manager import JobManager, Job, JobStatus, JobType, JobPriority


logger = logging.getLogger(__name__)


class JobExecutor:
    """Base class for job executors."""

    def can_execute(self, job: Job) -> bool:
        """Check if this executor can handle the given job."""
        raise NotImplementedError

    def execute(self, job: Job) -> Dict[str, Any]:
        """Execute the job and return the result."""
        raise NotImplementedError


class QueryJobExecutor(JobExecutor):
    """Executor for query-type jobs."""

    def __init__(self, taiat_engine):
        self.taiat_engine = taiat_engine

    def can_execute(self, job: Job) -> bool:
        return job.job_type == JobType.QUERY

    def execute(self, job: Job) -> Dict[str, Any]:
        """Execute a Taiat query."""
        try:
            query = TaiatQuery(
                query=job.payload.get("query", ""),
                inferred_goal_output=job.payload.get("inferred_goal_output"),
            )

            # Execute the query using the Taiat engine
            result = self.taiat_engine.run(query)

            return {
                "success": True,
                "result": result,
                "query_id": query.id,
            }
        except Exception as e:
            logger.error(f"Error executing query job {job.id}: {e}")
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }


class AgentJobExecutor(JobExecutor):
    """Executor for agent-type jobs."""

    def __init__(self, agent_registry: Dict[str, Callable]):
        self.agent_registry = agent_registry

    def can_execute(self, job: Job) -> bool:
        return job.job_type == JobType.AGENT

    def execute(self, job: Job) -> Dict[str, Any]:
        """Execute an agent job."""
        try:
            agent_name = job.payload.get("agent_name")
            if not agent_name:
                raise ValueError("Agent name is required for agent jobs")

            if agent_name not in self.agent_registry:
                raise ValueError(f"Agent '{agent_name}' not found in registry")

            agent_func = self.agent_registry[agent_name]
            agent_params = job.payload.get("parameters", {})

            # Execute the agent
            result = agent_func(**agent_params)

            return {
                "success": True,
                "result": result,
                "agent_name": agent_name,
            }
        except Exception as e:
            logger.error(f"Error executing agent job {job.id}: {e}")
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }


class TaiatWorker:
    """Worker that monitors and executes jobs."""

    def __init__(
        self,
        session_maker: sessionmaker,
        taiat_engine=None,
        agent_registry: Optional[Dict[str, Callable]] = None,
        poll_interval: float = 5.0,
        max_concurrent_jobs: int = 4,
        cleanup_interval_hours: int = 24,
    ):
        self.session_maker = session_maker
        self.job_manager = JobManager(session_maker)
        self.taiat_engine = taiat_engine
        self.agent_registry = agent_registry or {}
        self.poll_interval = poll_interval
        self.max_concurrent_jobs = max_concurrent_jobs
        self.cleanup_interval_hours = cleanup_interval_hours

        # Executors
        self.executors: List[JobExecutor] = []
        if self.taiat_engine:
            self.executors.append(QueryJobExecutor(self.taiat_engine))
        if self.agent_registry:
            self.executors.append(AgentJobExecutor(self.agent_registry))

        # Runtime state
        self.running = False
        self.executor_pool = ThreadPoolExecutor(max_workers=max_concurrent_jobs)
        self.active_jobs: Dict[str, asyncio.Task] = {}

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop()

    def add_executor(self, executor: JobExecutor):
        """Add a custom job executor."""
        self.executors.append(executor)

    def get_executor(self, job: Job) -> Optional[JobExecutor]:
        """Get the appropriate executor for a job."""
        for executor in self.executors:
            if executor.can_execute(job):
                return executor
        return None

    async def start(self):
        """Start the worker."""
        logger.info("Starting Taiat worker...")
        self.running = True

        # Start cleanup task
        cleanup_task = asyncio.create_task(self._cleanup_loop())

        try:
            while self.running:
                await self._process_jobs()
                await asyncio.sleep(self.poll_interval)
        except Exception as e:
            logger.error(f"Worker error: {e}")
            raise
        finally:
            cleanup_task.cancel()
            await self._shutdown()

    def stop(self):
        """Stop the worker."""
        logger.info("Stopping Taiat worker...")
        self.running = False

    async def _process_jobs(self):
        """Process available jobs."""
        try:
            # Get ready jobs
            ready_jobs = self.job_manager.get_ready_jobs()

            # Filter out jobs that are already running
            available_jobs = [
                job for job in ready_jobs if job.id not in self.active_jobs
            ]

            # Limit concurrent jobs
            available_jobs = available_jobs[
                : self.max_concurrent_jobs - len(self.active_jobs)
            ]

            # Start new jobs
            for job in available_jobs:
                executor = self.get_executor(job)
                if not executor:
                    logger.warning(
                        f"No executor found for job {job.id} (type: {job.job_type})"
                    )
                    self.job_manager.update_job_status(
                        job.id,
                        JobStatus.FAILED,
                        error_message=f"No executor found for job type {job.job_type}",
                    )
                    continue

                # Mark job as running
                self.job_manager.update_job_status(job.id, JobStatus.RUNNING)

                # Create task for job execution
                task = asyncio.create_task(self._execute_job(job, executor))
                self.active_jobs[job.id] = task

                logger.info(f"Started job {job.id} ({job.name})")

        except Exception as e:
            logger.error(f"Error processing jobs: {e}")

    async def _execute_job(self, job: Job, executor: JobExecutor):
        """Execute a job."""
        try:
            # Run the job in a thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor_pool, executor.execute, job
            )

            # Handle result
            if result.get("success", False):
                self.job_manager.update_job_status(
                    job.id, JobStatus.COMPLETED, result=result.get("result")
                )
                logger.info(f"Job {job.id} completed successfully")
            else:
                # Check if we should retry
                if job.retry_count < job.max_retries:
                    self.job_manager.increment_retry_count(job.id)
                    self.job_manager.update_job_status(
                        job.id, JobStatus.PENDING, error_message=result.get("error")
                    )
                    logger.warning(
                        f"Job {job.id} failed, will retry ({job.retry_count + 1}/{job.max_retries})"
                    )
                else:
                    self.job_manager.update_job_status(
                        job.id, JobStatus.FAILED, error_message=result.get("error")
                    )
                    logger.error(
                        f"Job {job.id} failed permanently after {job.max_retries} retries"
                    )

            # Move completed jobs to history
            if job.status in [JobStatus.COMPLETED, JobStatus.FAILED]:
                self.job_manager.move_to_history(job)

        except Exception as e:
            logger.error(f"Unexpected error executing job {job.id}: {e}")
            self.job_manager.update_job_status(
                job.id, JobStatus.FAILED, error_message=str(e)
            )

        finally:
            # Remove from active jobs
            if job.id in self.active_jobs:
                del self.active_jobs[job.id]

    async def _cleanup_loop(self):
        """Periodic cleanup of old job history."""
        while self.running:
            try:
                await asyncio.sleep(self.cleanup_interval_hours * 3600)
                if self.running:
                    deleted_count = self.job_manager.cleanup_old_history()
                    if deleted_count > 0:
                        logger.info(
                            f"Cleaned up {deleted_count} old job history entries"
                        )
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")

    async def _shutdown(self):
        """Shutdown the worker gracefully."""
        logger.info("Shutting down worker...")

        # Cancel all active jobs
        for job_id, task in self.active_jobs.items():
            task.cancel()
            logger.info(f"Cancelled job {job_id}")

        # Wait for tasks to complete
        if self.active_jobs:
            await asyncio.gather(*self.active_jobs.values(), return_exceptions=True)

        # Shutdown thread pool
        self.executor_pool.shutdown(wait=True)

        logger.info("Worker shutdown complete")

    def get_status(self) -> Dict[str, Any]:
        """Get worker status."""
        return {
            "running": self.running,
            "active_jobs": len(self.active_jobs),
            "max_concurrent_jobs": self.max_concurrent_jobs,
            "poll_interval": self.poll_interval,
            "executors": [type(executor).__name__ for executor in self.executors],
        }
