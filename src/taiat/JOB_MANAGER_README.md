# Taiat Job Manager

A PostgreSQL-based job management system for Taiat that supports job dependencies and distributed execution.

## Features

- **Job Dependencies**: Jobs can depend on other jobs and will only run when dependencies are completed
- **Retry Logic**: Failed jobs can be automatically retried with configurable retry limits
- **Job History**: Completed jobs are moved to a history table for auditing
- **Multiple Job Types**: Support for both Taiat queries and custom agent functions
- **Concurrent Execution**: Configurable number of concurrent job executions
- **Graceful Shutdown**: Worker handles shutdown signals gracefully

## Database Schema

### taiat_jobs Table
Stores pending and running jobs.

```sql
CREATE TABLE taiat_jobs (
    id VARCHAR PRIMARY KEY,
    job_type VARCHAR NOT NULL,
    name VARCHAR NOT NULL,
    description TEXT,
    payload JSONB NOT NULL DEFAULT '{}',
    dependencies JSONB NOT NULL DEFAULT '[]',
    max_retries INTEGER NOT NULL DEFAULT 3,
    retry_count INTEGER NOT NULL DEFAULT 0,
    status VARCHAR NOT NULL DEFAULT 'pending',
    error_message TEXT,
    result JSONB,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

### taiat_job_history Table
Stores completed jobs for auditing.

```sql
CREATE TABLE taiat_job_history (
    id VARCHAR PRIMARY KEY,
    job_id VARCHAR NOT NULL,
    job_type VARCHAR NOT NULL,
    name VARCHAR NOT NULL,
    description TEXT,
    payload JSONB NOT NULL DEFAULT '{}',
    dependencies JSONB NOT NULL DEFAULT '[]',
    status VARCHAR NOT NULL,
    error_message TEXT,
    result JSONB,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    duration_seconds INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your database URL:
```bash
export TAIAT_DATABASE_URL="postgresql://username:password@localhost/taiat"
```

3. Initialize the database tables:
```bash
python -m taiat.init_worker
```

## Usage

### Starting the Worker

```bash
# Basic usage
python -m taiat.run_worker

# With custom settings
python -m taiat.run_worker \
    --database-url "postgresql://localhost/taiat" \
    --poll-interval 5.0 \
    --max-concurrent-jobs 4 \
    --log-level DEBUG
```

### Creating Jobs Programmatically

```python
from taiat.job_manager import JobManager, JobType
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Setup database connection
engine = create_engine("postgresql://localhost/taiat")
session_maker = sessionmaker(bind=engine)
job_manager = JobManager(session_maker)

# Create a simple job
job = job_manager.create_job(
    job_type=JobType.AGENT,
    name="data_processing",
    description="Process some data",
    payload={
        "agent_name": "data_processor",
        "parameters": {"input_file": "data.csv"}
    }
)

# Create a job with dependencies
dependent_job = job_manager.create_job(
    job_type=JobType.QUERY,
    name="analyze_results",
    description="Analyze the processed data",
    payload={
        "query": "Analyze the processed data and generate insights",
        "inferred_goal_output": "Analysis report"
    },
    dependencies=[job.id]  # This job will wait for the first job to complete
)
```

### Job Types

#### Query Jobs
Execute Taiat queries using the Taiat engine.

```python
job = job_manager.create_job(
    job_type=JobType.QUERY,
    name="my_query",
    payload={
        "query": "What are the main trends in this dataset?",
        "inferred_goal_output": "Trend analysis report"
    }
)
```

#### Agent Jobs
Execute custom agent functions.

```python
job = job_manager.create_job(
    job_type=JobType.AGENT,
    name="custom_analysis",
    payload={
        "agent_name": "my_custom_agent",
        "parameters": {
            "input_file": "data.csv",
            "output_file": "results.json",
            "config": {"method": "random_forest"}
        }
    }
)
```

### Monitoring Jobs

```python
# Get job status
job = job_manager.get_job(job_id)
print(f"Job {job.name}: {job.status.value}")

# Get ready jobs (jobs whose dependencies are satisfied)
ready_jobs = job_manager.get_ready_jobs()

# Get job history
history = job_manager.get_job_history(limit=10)
```

### Worker Status

Check worker status without starting it:

```bash
python -m taiat.run_worker --status
```

## Configuration

### Environment Variables

- `TAIAT_DATABASE_URL`: Database connection URL
- `TAIAT_AGENT_CONFIG`: JSON configuration for agent registry

### Worker Configuration

- `poll_interval`: How often to check for new jobs (default: 5.0 seconds)
- `max_concurrent_jobs`: Maximum number of jobs to run simultaneously (default: 4)
- `cleanup_interval_hours`: How often to clean up old job history (default: 24 hours)

## Example Workflow

1. **Initialize the database**:
```bash
python -m taiat.init_worker
```

2. **Start the worker** (in one terminal):
```bash
python -m taiat.run_worker
```

3. **Create jobs** (in another terminal):
```bash
python -m taiat.example_usage
```

4. **Monitor progress**:
```bash
python -m taiat.run_worker --status
```

## Job Statuses

- `PENDING`: Job is waiting to be executed
- `RUNNING`: Job is currently being executed
- `COMPLETED`: Job completed successfully
- `FAILED`: Job failed and won't be retried
- `CANCELLED`: Job was cancelled

## Error Handling

- Jobs that fail are automatically retried up to `max_retries` times
- After all retries are exhausted, jobs are marked as `FAILED`
- Failed jobs are moved to the history table with error details
- The worker continues processing other jobs even if some fail

## Cleanup

The worker automatically cleans up old job history entries (default: 30 days old). You can also manually clean up:

```python
deleted_count = job_manager.cleanup_old_history(days=30)
print(f"Cleaned up {deleted_count} old entries")
```

## Extending the System

### Adding Custom Executors

```python
from taiat.worker import JobExecutor

class MyCustomExecutor(JobExecutor):
    def can_execute(self, job):
        return job.job_type == JobType.AGENT and job.payload.get("agent_name") == "my_agent"
    
    def execute(self, job):
        # Your custom execution logic here
        return {"success": True, "result": "custom_result"}

# Add to worker
worker.add_executor(MyCustomExecutor())
```

### Custom Agent Registry

```python
def my_custom_agent(**kwargs):
    # Your agent logic here
    return {"result": "agent_completed"}

agent_registry = {
    "my_custom_agent": my_custom_agent
}

worker = TaiatWorker(
    session_maker=session_maker,
    agent_registry=agent_registry
)
```

## Troubleshooting

### Common Issues

1. **Database connection failed**: Check your `TAIAT_DATABASE_URL` and ensure PostgreSQL is running
2. **Tables don't exist**: Run `python -m taiat.init_worker` first
3. **Jobs not executing**: Check if the worker is running and if job dependencies are satisfied
4. **Agent not found**: Ensure the agent is registered in the agent registry

### Logging

Set log level to DEBUG for detailed information:

```bash
python -m taiat.run_worker --log-level DEBUG
```

### Database Queries

Check job status directly in the database:

```sql
-- Check pending jobs
SELECT * FROM taiat_jobs WHERE status = 'pending';

-- Check job dependencies
SELECT id, name, dependencies FROM taiat_jobs WHERE dependencies != '[]';

-- Check recent history
SELECT * FROM taiat_job_history ORDER BY created_at DESC LIMIT 10;
``` 