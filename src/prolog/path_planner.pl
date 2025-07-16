% Taiat Path Planner Prolog Script
% This script takes an AgentGraphNodeSet and desired outputs to determine
% the optimal execution path for a TaiatQuery.

% Data structures:
% node(Name, Description, Inputs, Outputs)
% agent_data(Name, Parameters, Description, Data)
% agent_graph_node_set(Nodes)

% Input format for Python integration:
% Input: JSON-like structure representing AgentGraphNodeSet and desired outputs
% Output: List of node names in execution order

% Helper predicates for working with agent data
agent_data_name(agent_data(Name, _, _, _), Name).
agent_data_parameters(agent_data(_, Parameters, _, _), Parameters).
agent_data_description(agent_data(_, _, Description, _), Description).

% Helper predicates for working with nodes
node_name(node(Name, _, _, _), Name).
node_description(node(_, Description, _, _), Description).
node_inputs(node(_, _, Inputs, _), Inputs).
node_outputs(node(_, _, _, Outputs), Outputs).

% Check if two agent data items match (name and parameters)
agent_data_match(agent_data(Name1, Params1, _, _), agent_data(Name2, Params2, _, _)) :-
    Name1 = Name2,
    % For now, we'll do simple parameter matching
    % In a full implementation, this would check parameter subset relationships
    Params1 = Params2.

% Find a node that produces a specific output
node_produces_output(Node, Output) :-
    node_outputs(Node, Outputs),
    member(Output, Outputs).

% Find all nodes that produce a specific output name
nodes_producing_output_name(Nodes, OutputName, ProducingNodes) :-
    findall(Node, 
        (member(Node, Nodes), 
         node_outputs(Node, Outputs),
         member(agent_data(OutputName, _, _, _), Outputs)),
        ProducingNodes).

% Find all nodes that consume a specific input name
nodes_consuming_input_name(Nodes, InputName, ConsumingNodes) :-
    findall(Node,
        (member(Node, Nodes),
         node_inputs(Node, Inputs),
         member(agent_data(InputName, _, _, _), Inputs)),
        ConsumingNodes).

% Get all dependencies for a node (nodes that produce its inputs)
node_dependencies(Nodes, Node, Dependencies) :-
    node_inputs(Node, Inputs),
    findall(DependencyNode,
        (member(Input, Inputs),
         agent_data_name(Input, InputName),
         nodes_producing_output_name(Nodes, InputName, ProducingNodes),
         member(DependencyNode, ProducingNodes)),
        Dependencies).

% Check if a node is ready to execute (all dependencies satisfied)
node_ready(Nodes, Node, ExecutedNodes) :-
    node_dependencies(Nodes, Node, Dependencies),
    forall(member(Dependency, Dependencies),
           member(Dependency, ExecutedNodes)).

% Find all nodes needed to produce the desired outputs
required_nodes(Nodes, DesiredOutputs, RequiredNodes) :-
    required_nodes_from_outputs(Nodes, DesiredOutputs, [], RequiredNodes).

% Recursively collect required nodes from a list of agent_data outputs
required_nodes_from_outputs(_, [], Acc, Acc).
required_nodes_from_outputs(Nodes, [Output|Rest], Acc, RequiredNodes) :-
    agent_data_name(Output, OutputName),
    nodes_producing_output_name(Nodes, OutputName, [ProducingNode|_]),
    required_nodes_from_node(Nodes, ProducingNode, Acc, NewAcc),
    required_nodes_from_outputs(Nodes, Rest, NewAcc, RequiredNodes).
required_nodes_from_outputs(Nodes, [_|Rest], Acc, RequiredNodes) :-
    % If no node produces this output, skip
    required_nodes_from_outputs(Nodes, Rest, Acc, RequiredNodes).

% Recursively collect required nodes starting from a node
required_nodes_from_node(Nodes, Node, Acc, NewAcc) :-
    (member(Node, Acc) ->
        NewAcc = Acc
    ;
        node_inputs(Node, Inputs),
        required_nodes_from_outputs(Nodes, Inputs, Acc, AccWithDeps),
        append(AccWithDeps, [Node], NewAcc)
    ).

% Topological sort to determine execution order
topological_sort(Nodes, SortedNodes) :-
    topological_sort_recursive(Nodes, [], SortedNodes).

topological_sort_recursive([], SortedNodes, SortedNodes).
topological_sort_recursive(Nodes, CurrentSorted, FinalSorted) :-
    findall(Node,
        (member(Node, Nodes),
         node_ready(Nodes, Node, CurrentSorted)),
        ReadyNodes),
    (ReadyNodes = [] ->
        % No nodes ready - this might indicate a cycle or missing dependencies
        FinalSorted = CurrentSorted
    ;   % Add ready nodes to sorted list
        append(CurrentSorted, ReadyNodes, UpdatedSorted),
        list_subtract(Nodes, ReadyNodes, RemainingNodes),
        topological_sort_recursive(RemainingNodes, UpdatedSorted, FinalSorted)
    ).

% Helper predicate for list subtraction (GNU Prolog doesn't have subtract/3)
list_subtract([], _, []).
list_subtract([H|T], L, R) :-
    member(H, L),
    list_subtract(T, L, R).
list_subtract([H|T], L, [H|R]) :-
    \+ member(H, L),
    list_subtract(T, L, R).

% Helper predicate to extract node names from a list of nodes
extract_node_names([], []).
extract_node_names([node(Name, _, _, _)|T], [Name|NamesT]) :-
    extract_node_names(T, NamesT).

% Main predicate: plan execution path
plan_execution_path(NodeSet, DesiredOutputs, ExecutionPath) :-
    % Extract nodes from node set
    agent_graph_node_set_nodes(NodeSet, Nodes),
    
    % Find all required nodes
    required_nodes(Nodes, DesiredOutputs, RequiredNodes),
    
    % Remove duplicates while preserving order
    remove_duplicates(RequiredNodes, UniqueRequiredNodes),
    
    % Perform topological sort to get execution order
    topological_sort(UniqueRequiredNodes, SortedNodes),
    
    % Extract only the node names for output
    extract_node_names(SortedNodes, ExecutionPath).

% Helper predicate to extract nodes from node set
agent_graph_node_set_nodes(agent_graph_node_set(Nodes), Nodes).

% Helper predicate to remove duplicates while preserving order
remove_duplicates([], []).
remove_duplicates([H|T], [H|T1]) :-
    list_delete(T, H, T2),
    remove_duplicates(T2, T1).

% Helper predicate for list deletion (GNU Prolog doesn't have delete/3)
% This removes ALL occurrences of the element
list_delete([], _, []).
list_delete([H|T], H, Result) :-
    list_delete(T, H, Result).
list_delete([H|T], X, [H|T1]) :-
    H \= X,
    list_delete(T, X, T1).

% Predicate to validate that all desired outputs can be produced
validate_outputs(NodeSet, DesiredOutputs, Valid) :-
    agent_graph_node_set_nodes(NodeSet, Nodes),
    (forall(member(Output, DesiredOutputs),
            (agent_data_name(Output, OutputName),
             nodes_producing_output_name(Nodes, OutputName, ProducingNodes),
             ProducingNodes \= [])) ->
        Valid = true
    ;   Valid = false
    ).

% Predicate to find all outputs that can be produced by the node set
available_outputs(NodeSet, AvailableOutputs) :-
    agent_graph_node_set_nodes(NodeSet, Nodes),
    findall(Output,
        (member(Node, Nodes),
         node_outputs(Node, Outputs),
         member(Output, Outputs)),
        AllOutputs),
    remove_duplicates(AllOutputs, AvailableOutputs).

% Predicate to check for circular dependencies
has_circular_dependencies(NodeSet, HasCircular) :-
    agent_graph_node_set_nodes(NodeSet, Nodes),
    findall(Node,
        (member(Node, Nodes),
         node_dependencies(Nodes, Node, Dependencies),
         member(Node, Dependencies)),
        CircularNodes),
    (CircularNodes = [] ->
        HasCircular = false
    ;   HasCircular = true).

% Example usage predicates for testing
example_node_set(agent_graph_node_set([
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

example_desired_outputs([
    agent_data('visualizations', [], 'Generated visualizations', null)
]).

% Test predicate
test_path_planning :-
    example_node_set(NodeSet),
    example_desired_outputs(DesiredOutputs),
    plan_execution_path(NodeSet, DesiredOutputs, ExecutionPath),
    write('Execution Path: '), write(ExecutionPath), nl. 