% Unit tests for the Taiat Path Planner
% Run with: gplc path_planner_test.pl path_planner.pl -o test_path_planner && ./test_path_planner

test_write :- write('DEBUG: test_write works!'), nl.

% Test data
test_node_set(agent_graph_node_set([
    node('data_loader', 'Load data from source', 
         [], 
         [agent_data('raw_data', [], 'Raw data from source', null)]),
    node('preprocessor', 'Preprocess the data',
         [agent_data('raw_data', [], 'Raw data from source', null)],
         [agent_data('processed_data', [], 'Preprocessed data', null)]),
    node('analyzer', 'Analyze the data',
         [agent_data('processed_data', [], 'Preprocessed data', null)],
         [agent_data('analysis_results', [], 'Analysis results', null)]),
    node('visualizer', 'Create visualizations',
         [agent_data('analysis_results', [], 'Analysis results', null)],
         [agent_data('visualizations', [], 'Generated visualizations', null)])
])).

test_desired_outputs([
    agent_data('visualizations', [], 'Generated visualizations', null)
]).

% Test complex node set with multiple paths
test_complex_node_set(agent_graph_node_set([
    node('csv_loader', 'Load CSV data',
         [],
         [agent_data('csv_data', [], 'CSV data', null)]),
    node('json_loader', 'Load JSON data',
         [],
         [agent_data('json_data', [], 'JSON data', null)]),
    node('csv_processor', 'Process CSV data',
         [agent_data('csv_data', [], 'CSV data', null)],
         [agent_data('processed_csv', [], 'Processed CSV data', null)]),
    node('json_processor', 'Process JSON data',
         [agent_data('json_data', [], 'JSON data', null)],
         [agent_data('processed_json', [], 'Processed JSON data', null)]),
    node('data_merger', 'Merge processed data',
         [agent_data('processed_csv', [], 'Processed CSV data', null),
          agent_data('processed_json', [], 'Processed JSON data', null)],
         [agent_data('merged_data', [], 'Merged data', null)]),
    node('analyzer', 'Analyze merged data',
         [agent_data('merged_data', [], 'Merged data', null)],
         [agent_data('analysis_results', [], 'Analysis results', null)])
])).

test_complex_desired_outputs([
    agent_data('analysis_results', [], 'Analysis results', null)
]).

% Test predicates
test_agent_data_name :-
    agent_data_name(agent_data('test', [], 'test', null), 'test'),
    write('✓ agent_data_name test passed'), nl.

test_agent_data_match :-
    agent_data_match(
        agent_data('test', [], 'test', null),
        agent_data('test', [], 'test', null)
    ),
    write('✓ agent_data_match test passed'), nl.

test_agent_data_match_params_subset :-
    agent_data_match(
        agent_data('model', [type-logistic_regression], 'input', null),
        agent_data('model', [type-logistic_regression, version-v1], 'output', null)
    ),
    write('✓ agent_data_match params subset test passed'), nl.

test_agent_data_match_params_conflict :-
    (   agent_data_match(
            agent_data('model', [type-logistic_regression], 'input', null),
            agent_data('model', [type-neural_network, version-v1], 'output', null)
        )
    ->  (write('✗ agent_data_match params conflict test failed!'), nl, fail)
    ;   write('✓ agent_data_match params conflict test passed'), nl
    ).

test_agent_data_match_params_empty :-
    agent_data_match(
        agent_data('model', [], 'input', null),
        agent_data('model', [type-neural_network], 'output', null)
    ),
    write('✓ agent_data_match params empty test passed'), nl.

test_agent_data_match_name_mismatch :-
    (   agent_data_match(
            agent_data('modelA', [type-logistic_regression], 'input', null),
            agent_data('modelB', [type-logistic_regression], 'output', null)
        )
    ->  (write('✗ agent_data_match name mismatch test failed!'), nl, fail)
    ;   write('✓ agent_data_match name mismatch test passed'), nl
    ).

test_node_produces_output :-
    test_node_set(NodeSet),
    agent_graph_node_set_nodes(NodeSet, Nodes),
    member(Node, Nodes),
    node_name(Node, 'data_loader'),
    node_produces_output(Node, agent_data('raw_data', [], 'Raw data from source', null)),
    write('✓ node_produces_output test passed'), nl.

test_nodes_producing_output_name :-
    test_node_set(NodeSet),
    agent_graph_node_set_nodes(NodeSet, Nodes),
    nodes_producing_output_name(Nodes, 'raw_data', ProducingNodes),
    length(ProducingNodes, 1),
    member(Node, ProducingNodes),
    node_name(Node, 'data_loader'),
    write('✓ nodes_producing_output_name test passed'), nl.

test_node_dependencies :-
    test_node_set(NodeSet),
    agent_graph_node_set_nodes(NodeSet, Nodes),
    member(Node, Nodes),
    node_name(Node, 'preprocessor'),
    node_dependencies(Nodes, Node, Dependencies),
    length(Dependencies, 1),
    member(Dependency, Dependencies),
    node_name(Dependency, 'data_loader'),
    write('✓ node_dependencies test passed'), nl.

test_node_ready :-
    test_node_set(NodeSet),
    agent_graph_node_set_nodes(NodeSet, Nodes),
    member(Node, Nodes),
    node_name(Node, 'preprocessor'),
    node_ready(Nodes, Node, [node('data_loader', 'Load data from source', [], [agent_data('raw_data', [], 'Raw data from source', null)])]),
    write('✓ node_ready test passed'), nl.

test_required_nodes :-
    test_node_set(NodeSet),
    agent_graph_node_set_nodes(NodeSet, Nodes),
    test_desired_outputs(DesiredOutputs),
    required_nodes(Nodes, DesiredOutputs, RequiredNodes),
    length(RequiredNodes, 4),
    member(node('data_loader', _, _, _), RequiredNodes),
    member(node('preprocessor', _, _, _), RequiredNodes),
    member(node('analyzer', _, _, _), RequiredNodes),
    member(node('visualizer', _, _, _), RequiredNodes),
    write('✓ required_nodes test passed'), nl.

test_topological_sort :-
    test_node_set(NodeSet),
    agent_graph_node_set_nodes(NodeSet, Nodes),
    topological_sort(Nodes, SortedNodes),
    length(SortedNodes, 4),
    nth(1, SortedNodes, FirstNode),
    node_name(FirstNode, 'data_loader'),
    nth(2, SortedNodes, SecondNode),
    node_name(SecondNode, 'preprocessor'),
    nth(3, SortedNodes, ThirdNode),
    node_name(ThirdNode, 'analyzer'),
    nth(4, SortedNodes, FourthNode),
    node_name(FourthNode, 'visualizer'),
    write('✓ topological_sort test passed'), nl.

test_plan_execution_path :-
    test_node_set(NodeSet),
    test_desired_outputs(DesiredOutputs),
    plan_execution_path(NodeSet, DesiredOutputs, ExecutionPath),
    ExecutionPath = ['data_loader', 'preprocessor', 'analyzer', 'visualizer'],
    write('✓ plan_execution_path test passed'), nl.

test_validate_outputs :-
    test_node_set(NodeSet),
    validate_outputs(NodeSet, test_desired_outputs, Valid),
    Valid = true,
    write('✓ validate_outputs test passed'), nl.

test_available_outputs :-
    test_node_set(NodeSet),
    available_outputs(NodeSet, AvailableOutputs),
    member(agent_data('raw_data', _, _, _), AvailableOutputs),
    member(agent_data('processed_data', _, _, _), AvailableOutputs),
    member(agent_data('analysis_results', _, _, _), AvailableOutputs),
    member(agent_data('visualizations', _, _, _), AvailableOutputs),
    write('✓ available_outputs test passed'), nl.

test_complex_path_planning :-
    test_complex_node_set(NodeSet),
    test_complex_desired_outputs(DesiredOutputs),
    plan_execution_path(NodeSet, DesiredOutputs, ExecutionPath),
    member('csv_loader', ExecutionPath),
    member('json_loader', ExecutionPath),
    member('csv_processor', ExecutionPath),
    member('json_processor', ExecutionPath),
    member('data_merger', ExecutionPath),
    member('analyzer', ExecutionPath),
    write('✓ complex_path_planning test passed'), nl.

test_invalid_output :-
    test_node_set(NodeSet),
    validate_outputs(NodeSet, [agent_data('nonexistent', [], 'Nonexistent output', null)], Valid),
    Valid = false,
    write('✓ invalid_output test passed'), nl.

test_list_subtract :-
    list_subtract([1,2,3,4], [2,4], [1,3]),
    list_subtract([], [1,2], []),
    list_subtract([1,2,3], [], [1,2,3]),
    write('✓ list_subtract test passed'), nl.

test_list_delete :-
    list_delete([1,2,3,2,4], 2, [1,3,4]),
    list_delete([], 1, []),
    list_delete([1,2,3], 4, [1,2,3]),
    write('✓ list_delete test passed'), nl.

test_remove_duplicates :-
    remove_duplicates([1,2,2,3,1,4], [1,2,3,4]),
    remove_duplicates([], []),
    remove_duplicates([1], [1]),
    write('✓ remove_duplicates test passed'), nl.

% Run all tests
run_all_tests :-
    write('Running Taiat Path Planner Tests...'), nl, nl,
    run_test(test_agent_data_name),
    run_test(test_agent_data_match),
    run_test(test_agent_data_match_params_subset),
    run_test(test_agent_data_match_params_conflict),
    run_test(test_agent_data_match_params_empty),
    run_test(test_agent_data_match_name_mismatch),
    run_test(test_node_produces_output),
    run_test(test_nodes_producing_output_name),
    run_test(test_node_dependencies),
    run_test(test_node_ready),
    run_test(test_required_nodes),
    run_test(test_topological_sort),
    run_test(test_plan_execution_path),
    run_test(test_validate_outputs),
    run_test(test_available_outputs),
    run_test(test_complex_path_planning),
    run_test(test_invalid_output),
    run_test(test_list_subtract),
    run_test(test_list_delete),
    run_test(test_remove_duplicates),
    nl, write('All tests attempted!'), nl.

run_test(Test) :-
    (call(Test) -> true ; (write('✗ '), write(Test), write(' failed!'), nl)).

% Main entry point
main :-
    run_all_tests,
    halt.

% Automatically run main when program starts
:- initialization(main). 