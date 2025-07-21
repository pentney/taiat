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
% Input parameters must be a subset of output parameters (constraint matching)
agent_data_match(agent_data(Name1, Params1, _, _), agent_data(Name2, Params2, _, _)) :-
    Name1 = Name2,
    % Check if Params1 (input) is a subset of Params2 (output)
    % This implements constraint matching: input parameters are constraints
    % that must be satisfied by the output parameters
    parameters_subset(Params1, Params2).

% Check if parameters1 is a subset of parameters2. This means all key-value pairs in parameters1 must exist in parameters2
parameters_subset([], _).
parameters_subset([Key-Value|Rest], Params2) :-
    member(Key-Value, Params2),
    parameters_subset(Rest, Params2).

% Find a node that produces a specific output (with parameter matching)
node_produces_output(Node, Output) :-
    node_outputs(Node, Outputs),
    member(ProducedOutput, Outputs),
    agent_data_match(Output, ProducedOutput).

% Find all nodes that produce a specific output (with parameter matching)
% When multiple nodes can produce the same output, prefer the most specific match
nodes_producing_output(Nodes, Output, ProducingNodes) :-
    findall(Node, 
        (member(Node, Nodes), 
         node_outputs(Node, Outputs),
         member(ProducedOutput, Outputs),
         agent_data_match(Output, ProducedOutput)),
        AllProducingNodes),
    % If multiple nodes can produce this output, choose the most specific one
    (AllProducingNodes = [] ->
        ProducingNodes = []
    ;   AllProducingNodes = [SingleNode] ->
        ProducingNodes = [SingleNode]
    ;   % Multiple nodes - choose the most specific match
        find_most_specific_producer(Nodes, Output, AllProducingNodes, ProducingNodes)
    ).

% Find the most specific producer among multiple candidates
find_most_specific_producer(Nodes, DesiredOutput, Candidates, [BestProducer]) :-
    findall(Score-Node,
        (member(Node, Candidates),
         node_outputs(Node, Outputs),
         member(ProducedOutput, Outputs),
         agent_data_match(DesiredOutput, ProducedOutput),
         % Calculate specificity score
         agent_data_parameters(DesiredOutput, DesiredParams),
         agent_data_parameters(ProducedOutput, ProducedParams),
         calculate_specificity_score(DesiredParams, ProducedParams, Score)),
        ScoredNodes),
    % Sort by score (highest first) and take the first
    sort(ScoredNodes, SortedScoredNodes),
    reverse(SortedScoredNodes, [BestScore-BestProducer|_]).

% Calculate a specificity score for parameter matching
% Higher score means more specific match
calculate_specificity_score(DesiredParams, ProducedParams, Score) :-
    % Count how many desired parameters are matched exactly
    findall(1,
        (member(DesiredParam, DesiredParams),
         member(DesiredParam, ProducedParams)),
        ExactMatches),
    length(ExactMatches, Score).

% Find all nodes that produce a specific output name (legacy - for backward compatibility)
nodes_producing_output_name(Nodes, OutputName, ProducingNodes) :-
    findall(Node, 
        (member(Node, Nodes), 
         node_outputs(Node, Outputs),
         member(agent_data(OutputName, _, _, _), Outputs)),
        ProducingNodes).



% Get all dependencies for a node (nodes that produce its inputs with parameter matching)
% When multiple nodes can produce the same input, prefer the most specific match
node_dependencies(Nodes, Node, Dependencies) :-
    node_inputs(Node, Inputs),
    findall(DependencyNode,
        (member(Input, Inputs),
         nodes_producing_output(Nodes, Input, ProducingNodes),
         % If multiple nodes can produce this input, choose the most specific one
         (ProducingNodes = [DependencyNode|_] ->
            true
        ;   ProducingNodes = [DependencyNode])),
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
    nodes_producing_output(Nodes, Output, ProducingNodes),
    (ProducingNodes = [] ->
        % No node produces this output, fail
        fail
    ;
        % Try all possible producers, succeed if any can satisfy their inputs
        try_producers(Nodes, ProducingNodes, Output, Rest, Acc, RequiredNodes)
    ).

% Try all possible producers for an output, succeed if any can satisfy their inputs
try_producers(_, [], _, _, _, _) :- fail.
try_producers(Nodes, [Producer|RestProducers], Output, RestOutputs, Acc, RequiredNodes) :-
    (can_node_inputs_be_satisfied(Nodes, Producer) ->
        required_nodes_from_node(Nodes, Producer, Acc, NewAcc),
        required_nodes_from_outputs(Nodes, RestOutputs, NewAcc, RequiredNodes)
    ;
        try_producers(Nodes, RestProducers, Output, RestOutputs, Acc, RequiredNodes)
    ).

% Recursively collect required nodes starting from a node
required_nodes_from_node(Nodes, Node, Acc, NewAcc) :-
    (member(Node, Acc) ->
        NewAcc = Acc
    ;
        node_inputs(Node, Inputs),
        % Check if all inputs that are produced by nodes in the graph can be satisfied
        % If any input cannot be satisfied, this predicate should fail
        forall(member(Input, Inputs),
               (nodes_producing_output(Nodes, Input, ProducingNodes),
                (ProducingNodes = [] ->
                    % This input is not produced by any node, so it's an external input
                    % We don't need to validate it
                    true
                ;   % This input is produced by some nodes, so it must be satisfiable
                    % Check if any of the producing nodes can actually satisfy this input
                    findall(ProducerNode,
                        (member(ProducerNode, ProducingNodes),
                         node_outputs(ProducerNode, Outputs),
                         member(ProducedOutput, Outputs),
                         agent_data_match(Input, ProducedOutput)),
                        ValidProducers),
                    (ValidProducers = [] ->
                        % No valid producers for this input, fail
                        fail
                    ;   % At least one valid producer exists
                        true)
                ))),
        % For each input, find the most specific producer based on desired outputs
        findall(Input,
            (member(Input, Inputs),
             nodes_producing_output(Nodes, Input, ProducingNodes),
             ProducingNodes \= []),
            ValidInputs),
        required_nodes_from_outputs(Nodes, ValidInputs, Acc, AccWithDeps),
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
    
    % Filter to only include nodes that are actually needed
    filter_required_nodes(Nodes, DesiredOutputs, UniqueRequiredNodes, FilteredNodes),
    
    % Perform topological sort to get execution order
    topological_sort(FilteredNodes, SortedNodes),
    
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

 

% Check if a node is needed in the execution path
% A node is needed if it produces one of the desired outputs OR
% if it produces an output that is consumed by another required node
node_is_needed(Nodes, Node, DesiredOutputs, RequiredNodes, Needed) :-
    node_outputs(Node, NodeOutputs),
    (member(Output, NodeOutputs),
     member(DesiredOutput, DesiredOutputs),
     agent_data_match(DesiredOutput, Output) ->
        Needed = true
    ;   % Check if any of this node's outputs are consumed by other required nodes
        findall(ConsumerNode,
            (member(ConsumerNode, RequiredNodes),
             ConsumerNode \= Node,
             node_inputs(ConsumerNode, Inputs),
             member(Input, Inputs),
             member(ProducedOutput, NodeOutputs),
             agent_data_match(Input, ProducedOutput)),
            Consumers),
        (Consumers = [] ->
            Needed = false
        ;   % Check if this node produces parameterized outputs that aren't needed
            (member(ProducedOutput, NodeOutputs),
             agent_data_parameters(ProducedOutput, ProducedParams),
             ProducedParams \= [] ->
                % This node produces an output with specific parameters
                % Check if this specific parameter combination is actually needed
                (member(DesiredOutput, DesiredOutputs),
                 agent_data_name(DesiredOutput, OutputName),
                 agent_data_name(ProducedOutput, OutputName),
                 agent_data_parameters(DesiredOutput, DesiredParams),
                 ProducedParams = DesiredParams ->
                    % This exact parameter combination is requested
                    Needed = true
                ;   % Check if any consumer node needs this specific parameter combination
                    findall(ConsumerNode,
                        (member(ConsumerNode, RequiredNodes),
                         ConsumerNode \= Node,
                         node_inputs(ConsumerNode, Inputs),
                         member(Input, Inputs),
                         agent_data_name(Input, OutputName),
                         agent_data_name(ProducedOutput, OutputName),
                         agent_data_parameters(Input, InputParams),
                         parameters_subset(InputParams, ProducedParams)),
                        SpecificConsumers),
                    (SpecificConsumers = [] ->
                        Needed = false
                    ;   Needed = true)
                )
            ;   Needed = true)
        )
    ).

% Filter nodes to only include those that are actually needed
filter_required_nodes(Nodes, DesiredOutputs, AllRequiredNodes, FilteredNodes) :-
    findall(Node,
        (member(Node, AllRequiredNodes),
         node_is_needed(Nodes, Node, DesiredOutputs, AllRequiredNodes, true)),
        FilteredNodes).

% Plan alternative execution path excluding failed nodes
% This predicate uses backtracking to find different valid paths when some nodes have failed
plan_alternative_execution_path(NodeSet, DesiredOutputs, FailedNodeNames, ExecutionPath) :-
    agent_graph_node_set_nodes(NodeSet, Nodes),
    filter_out_failed_nodes(Nodes, FailedNodeNames, AvailableNodes),
    ( AvailableNodes = [] ->
        write('DEBUG: All nodes failed, returning empty path'), nl, flush_output,
        write('SUCCESS:[]'), nl, flush_output,
        ExecutionPath = []
    ;
        AvailableNodeSet = agent_graph_node_set(AvailableNodes),
        plan_execution_path(AvailableNodeSet, DesiredOutputs, ExecutionPath)
    ).

% Helper predicate to filter out failed nodes
filter_out_failed_nodes([], _, []).
filter_out_failed_nodes([Node|Rest], FailedNodeNames, [Node|FilteredRest]) :-
    node_name(Node, NodeName),
    \+ member(NodeName, FailedNodeNames),
    filter_out_failed_nodes(Rest, FailedNodeNames, FilteredRest).
filter_out_failed_nodes([Node|Rest], FailedNodeNames, FilteredRest) :-
    node_name(Node, NodeName),
    member(NodeName, FailedNodeNames),
    filter_out_failed_nodes(Rest, FailedNodeNames, FilteredRest).

% Plan multiple alternative paths with different strategies
% This predicate finds all possible alternative paths by trying different combinations
plan_multiple_alternative_paths(NodeSet, DesiredOutputs, FailedNodeNames, AlternativePaths) :-
    % Extract nodes from node set
    agent_graph_node_set_nodes(NodeSet, Nodes),
    
    % Filter out failed nodes
    filter_out_failed_nodes(Nodes, FailedNodeNames, AvailableNodes),
    
    % Find all possible paths by trying different node combinations
    findall(Path,
        (plan_alternative_execution_path(NodeSet, DesiredOutputs, FailedNodeNames, Path),
         Path \= []),
        AlternativePaths).

% Enhanced path planning that tries to find the best alternative when primary path fails
% This predicate implements a more sophisticated backtracking strategy
plan_execution_path_with_fallback(NodeSet, DesiredOutputs, FailedNodeNames, ExecutionPath) :-
    % First, try to plan without considering failed nodes
    plan_alternative_execution_path(NodeSet, DesiredOutputs, FailedNodeNames, ExecutionPath),
    ExecutionPath \= [].
    
% If no path found, try with different strategies
plan_execution_path_with_fallback(NodeSet, DesiredOutputs, FailedNodeNames, ExecutionPath) :-
    % Try to find any valid path by relaxing some constraints
    find_any_valid_path(NodeSet, DesiredOutputs, FailedNodeNames, ExecutionPath).

% Find any valid path by relaxing constraints
find_any_valid_path(NodeSet, DesiredOutputs, FailedNodeNames, ExecutionPath) :-
    % Extract nodes from node set
    agent_graph_node_set_nodes(NodeSet, Nodes),
    
    % Try different combinations of available nodes
    findall(Path,
        (subset(SelectedNodes, Nodes),
         filter_out_failed_nodes(SelectedNodes, FailedNodeNames, AvailableNodes),
         AvailableNodes \= [],
         AvailableNodeSet = agent_graph_node_set(AvailableNodes),
         plan_execution_path(AvailableNodeSet, DesiredOutputs, Path),
         Path \= []),
        AllPaths),
    
    % Take the shortest path as the best alternative
    (AllPaths = [] ->
        ExecutionPath = []
    ;   find_shortest_path(AllPaths, ExecutionPath)
    ).

% Helper predicate to find the shortest path from a list of paths
find_shortest_path([Path], Path).
find_shortest_path([Path1, Path2|Rest], ShortestPath) :-
    length(Path1, Len1),
    length(Path2, Len2),
    (Len1 =< Len2 ->
        find_shortest_path([Path1|Rest], ShortestPath)
    ;   find_shortest_path([Path2|Rest], ShortestPath)
    ).

% Helper predicate for subset generation (backtracking through all possible subsets)
subset([], []).
subset([H|T], [H|S]) :-
    subset(T, S).
subset([_|T], S) :-
    subset(T, S).

% Validate that all inputs for a specific node can be satisfied by the selected nodes
% Skip inputs that are not produced by any node in the graph (external inputs)
validate_node_inputs_from_selected(Nodes, SelectedNodes, Node) :-
    node_inputs(Node, Inputs),
    forall(member(Input, Inputs),
           (nodes_producing_output(Nodes, Input, ProducingNodes),
            (ProducingNodes = [] ->
                % This input is not produced by any node, so it's an external input
                % We don't need to validate it
                true
            ;   % This input is produced by some nodes, so validate it's satisfied
                % Check if any of the selected nodes can produce this input with matching parameters
                findall(ProducerNode,
                    (member(ProducerNode, SelectedNodes),
                     ProducerNode \= Node,
                     node_outputs(ProducerNode, Outputs),
                     member(ProducedOutput, Outputs),
                     agent_data_match(Input, ProducedOutput)),
                    ValidProducers),
                (ValidProducers = [] ->
                    % No selected node can produce this input with matching parameters
                    % Check if there are any nodes that could produce this input (even with wrong parameters)
                    findall(PotentialProducer,
                        (member(PotentialProducer, Nodes),
                         PotentialProducer \= Node,
                         node_outputs(PotentialProducer, Outputs),
                         member(ProducedOutput, Outputs),
                         agent_data_name(Input, InputName),
                         agent_data_name(ProducedOutput, InputName)),
                        PotentialProducers),
                    (PotentialProducers = [] ->
                        % No node at all can produce this input, so it's truly external
                        true
                    ;   % Some nodes can produce this input but with wrong parameters
                        % This should cause validation to fail
                        fail
                    )
                ;   % At least one selected node can produce this input
                    true
                )
            ))).



% Check if a node's inputs can be satisfied by any nodes in the graph
can_node_inputs_be_satisfied(Nodes, Node) :-
    node_inputs(Node, Inputs),
    forall(member(Input, Inputs),
           (nodes_producing_output(Nodes, Input, ProducingNodes),
            (ProducingNodes = [] ->
                % Check if this input is truly external (not produced by any node at all)
                % or if it's produced by nodes but with wrong parameters
                agent_data_name(Input, InputName),
                findall(PotentialProducer,
                    (member(PotentialProducer, Nodes),
                     PotentialProducer \= Node,
                     node_outputs(PotentialProducer, Outputs),
                     member(ProducedOutput, Outputs),
                     agent_data_name(ProducedOutput, InputName)),
                    PotentialProducers),
                (PotentialProducers = [] ->
                    % This input is truly external (not produced by any node)
                    true
                ;   % This input is produced by some nodes but with wrong parameters
                    % This should cause validation to fail
                    fail
                )
            ;   % This input is produced by some nodes, so it must be satisfiable
                % Check if any of the producing nodes can actually satisfy this input
                findall(ProducerNode,
                    (member(ProducerNode, ProducingNodes),
                     node_outputs(ProducerNode, Outputs),
                     member(ProducedOutput, Outputs),
                     agent_data_match(Input, ProducedOutput)),
                    ValidProducers),
                ValidProducers \= []
            ))).