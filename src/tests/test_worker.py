import asyncio
import os
import tempfile
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from taiat.base import AgentData, AgentGraphNode, AgentGraphNodeSet
from taiat.builder import TaiatBuilder
from taiat.engine import TaiatEngine
from taiat.job_manager import JobManager, JobType, JobStatus
from taiat.worker import JobExecutor, TaiatWorker


def create_data_processor(**kwargs) -> Dict[str, Any]:
    """Mock data processing agent."""
    return {"processed_data": "data_processed"}


def create_feature_extractor(**kwargs) -> Dict[str, Any]:
    """Mock feature extraction agent."""
    return {"features": "extracted_features"}


def create_model_trainer(**kwargs) -> Dict[str, Any]:
    """Mock model training agent."""
    return {"model": "trained_model"}


def create_analyzer(**kwargs) -> Dict[str, Any]:
    """Mock analysis agent."""
    return {"analysis": "analysis_results"}


@pytest.mark.asyncio
async def test_worker_creates_jobs_from_query_dependency_graph():
    """Test that the worker correctly creates jobs from a query's dependency graph."""
    # Create a temporary file for SQLite database
    temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    temp_db.close()

    try:
        # Setup file-based DB and job manager
        engine = create_engine(f"sqlite:///{temp_db.name}")
        from taiat.job_manager import metadata as job_metadata

        job_metadata.create_all(engine)

        # Verify tables were created
        with engine.connect() as conn:
            result = conn.execute(
                text(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='taiat_jobs'"
                )
            )
            tables = result.fetchall()

        Session = sessionmaker(bind=engine)
        job_manager = JobManager(Session)

        # Create AgentGraphNodeSet with multiple levels of dependencies
        # Level 1: data_processor (no dependencies)
        # Level 2: feature_extractor (depends on data_processor)
        # Level 3: model_trainer (depends on feature_extractor)
        # Level 4: analyzer (depends on both model_trainer and feature_extractor)

        node_set = AgentGraphNodeSet(
            nodes=[
                AgentGraphNode(
                    name="data_processor",
                    description="Process raw data",
                    function=create_data_processor,
                    inputs=[
                        AgentData(
                            name="raw_data",
                            parameters={},
                            description="Raw input data",
                        )
                    ],
                    outputs=[
                        AgentData(
                            name="processed_data",
                            parameters={"type": "tabular"},
                            description="Processed tabular data",
                        )
                    ],
                ),
                AgentGraphNode(
                    name="feature_extractor",
                    description="Extract features from processed data",
                    function=create_feature_extractor,
                    inputs=[
                        AgentData(
                            name="processed_data",
                            parameters={"type": "tabular"},
                            description="Processed tabular data",
                        )
                    ],
                    outputs=[
                        AgentData(
                            name="features",
                            parameters={"type": "numerical"},
                            description="Extracted numerical features",
                        )
                    ],
                ),
                AgentGraphNode(
                    name="model_trainer",
                    description="Train machine learning model",
                    function=create_model_trainer,
                    inputs=[
                        AgentData(
                            name="features",
                            parameters={"type": "numerical"},
                            description="Extracted numerical features",
                        )
                    ],
                    outputs=[
                        AgentData(
                            name="model",
                            parameters={"type": "ml_model"},
                            description="Trained machine learning model",
                        )
                    ],
                ),
                AgentGraphNode(
                    name="analyzer",
                    description="Analyze results",
                    function=create_analyzer,
                    inputs=[
                        AgentData(
                            name="features",
                            parameters={"type": "numerical"},
                            description="Extracted numerical features",
                        ),
                        AgentData(
                            name="model",
                            parameters={"type": "ml_model"},
                            description="Trained machine learning model",
                        ),
                    ],
                    outputs=[
                        AgentData(
                            name="analysis",
                            parameters={"type": "report"},
                            description="Analysis report",
                        )
                    ],
                ),
            ]
        )

        # Create TaiatBuilder and TaiatEngine
        mock_llm = MagicMock()
        builder = TaiatBuilder(mock_llm, verbose=False, add_metrics=False)

        # Build the graph
        graph = builder.build(
            node_set=node_set,
            inputs=[
                AgentData(
                    name="raw_data", parameters={}, description="Raw input data"
                )
            ],  # Provide a valid AgentData input matching data_processor
            terminal_nodes=["analyzer"],  # analyzer is the terminal node
        )

        taiat_engine = TaiatEngine(mock_llm, builder, None)

        # Create a custom QueryJobExecutor that creates jobs for each node
        class DependencyAwareQueryJobExecutor(JobExecutor):
            """Custom executor that creates jobs for each node in the dependency graph."""

            def __init__(self, taiat_engine, job_manager):
                self.taiat_engine = taiat_engine
                self.job_manager = job_manager
                self.jobs_created = []

            def can_execute(self, job) -> bool:
                return job.job_type == JobType.QUERY

            def execute(self, job) -> Dict[str, Any]:
                """Execute a query by creating jobs for each node in the dependency graph."""
                try:
                    # Test database connection
                    with self.job_manager.session_maker() as session:
                        result = session.execute(
                            text(
                                "SELECT name FROM sqlite_master WHERE type='table' AND name='taiat_jobs'"
                            )
                        )
                        tables = result.fetchall()

                    # Create jobs for each node in the dependency graph
                    node_jobs = {}

                    # Create jobs for each node
                    for node in self.taiat_engine.builder.node_set.nodes:
                        # Determine dependencies based on inputs
                        dependencies = []
                        for input_data in node.inputs:
                            # Find the node that provides this input
                            for (
                                provider_node
                            ) in self.taiat_engine.builder.node_set.nodes:
                                for output_data in provider_node.outputs:
                                    if (
                                        output_data.name == input_data.name
                                        and output_data.parameters
                                        == input_data.parameters
                                    ):
                                        if provider_node.name in node_jobs:
                                            dependencies.append(
                                                node_jobs[provider_node.name].id
                                            )
                                        break

                        # Create the job
                        node_job = self.job_manager.create_job(
                            job_type=JobType.AGENT,
                            name=node.name,
                            description=node.description,
                            payload={"agent_name": node.name, "parameters": {}},
                            dependencies=dependencies,
                        )
                        node_jobs[node.name] = node_job
                        self.jobs_created.append(node_job)

                    return {
                        "success": True,
                        "result": {"jobs_created": len(self.jobs_created)},
                        "query_id": job.id,
                    }
                except Exception as e:
                    import traceback

                    traceback.print_exc()
                    return {"success": False, "error": str(e), "traceback": str(e)}

        # Create agent registry
        agent_registry = {
            "data_processor": create_data_processor,
            "feature_extractor": create_feature_extractor,
            "model_trainer": create_model_trainer,
            "analyzer": create_analyzer,
        }

        # Setup worker with custom executor
        worker = TaiatWorker(
            session_maker=Session,
            taiat_engine=taiat_engine,
            agent_registry=agent_registry,
            poll_interval=0.1,
            max_concurrent_jobs=4,
        )

        # Replace the worker's job_manager with the one that has tables
        worker.job_manager = job_manager

        # Replace the default QueryJobExecutor with our custom one
        dependency_executor = DependencyAwareQueryJobExecutor(
            taiat_engine, job_manager
        )
        worker.executors = [
            dependency_executor,
            worker.executors[1],
        ]  # Keep AgentJobExecutor

        # Create and submit a query job
        query_job = job_manager.create_job(
            job_type=JobType.QUERY,
            name="ml_pipeline_query",
            description="Run machine learning pipeline",
            payload={
                "query": "Analyze the dataset and train a model",
                "inferred_goal_output": "analysis",
            },
        )

        # Process the query job once to create the dependency jobs
        await worker._process_jobs()

        # Wait a bit for job creation
        await asyncio.sleep(0.1)

        # Verify that jobs were created for each node
        assert len(dependency_executor.jobs_created) == 4

        # Get all jobs from the job manager
        all_jobs = job_manager.get_pending_jobs()
        assert len(all_jobs) == 4  # 4 node jobs (query job completed successfully)

        # Verify job names
        job_names = [job.name for job in all_jobs]
        assert "data_processor" in job_names
        assert "feature_extractor" in job_names
        assert "model_trainer" in job_names
        assert "analyzer" in job_names

        # Verify dependencies are correct
        data_processor_job = next(
            j for j in all_jobs if j.name == "data_processor"
        )
        feature_extractor_job = next(
            j for j in all_jobs if j.name == "feature_extractor"
        )
        model_trainer_job = next(
            j for j in all_jobs if j.name == "model_trainer"
        )
        analyzer_job = next(j for j in all_jobs if j.name == "analyzer")

        # data_processor has no dependencies
        assert data_processor_job.dependencies == []

        # feature_extractor depends on data_processor
        assert len(feature_extractor_job.dependencies) == 1
        assert feature_extractor_job.dependencies[0] == data_processor_job.id

        # model_trainer depends on feature_extractor
        assert len(model_trainer_job.dependencies) == 1
        assert model_trainer_job.dependencies[0] == feature_extractor_job.id

        # analyzer depends on both feature_extractor and model_trainer
        assert len(analyzer_job.dependencies) == 2
        assert feature_extractor_job.id in analyzer_job.dependencies
        assert model_trainer_job.id in analyzer_job.dependencies

        # Verify that ready jobs are correct (only data_processor should be ready initially)
        ready_jobs = job_manager.get_ready_jobs()
        ready_job_names = [job.name for job in ready_jobs]
        assert "data_processor" in ready_job_names
        assert "feature_extractor" not in ready_job_names
        assert "model_trainer" not in ready_job_names
        assert "analyzer" not in ready_job_names

    finally:
        # Clean up the temporary database file
        try:
            os.unlink(temp_db.name)
        except OSError:
            pass  # File might already be deleted
