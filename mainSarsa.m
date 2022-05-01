% Main program to run the GridWorld Q-learning algorithm

clc; clear;                         % clear the command window, the workspace and close the figures

% Initialize Environment variables
trainAgent    = true;
lengthGrid    = 8;                  % length/width of the grid
numEpisodes   = 20000;               % number of episodes to train the agent
maxIterations = 200;                % max number of iterations per episode

% Initialize Display variables
showEvery     = 1000;              % wait these many episodes to display

% Initialize QLearning object variables
epsilon       = 0.9;            % probability of exploration
epsilonDecay  = 0.99;             % rate at which exploration decreases (simulated annealing) 
alpha         = 0.005;                % update rate / weight of error term
gamma         = 0.9;                % discount of future rewards
lambda        = 0.8;                % eligibility temporal discount

% Instantiate reward objects
MOVE_PENALTY  = -1;   
CLIFF_PENALTY = -100;  
FOOD_REWARD   = 0;

% Object Locations
AGENT_LOCATION   = [8,1];
FOOD_LOCATION    = [8,8];

food  = Reward('food',FOOD_REWARD,FOOD_LOCATION);
enemy = Reward('enemy',CLIFF_PENALTY);
move  = Reward('move',MOVE_PENALTY);

% Instantiate sim objects
learningAlgorithm = SARSALearning(epsilon,epsilonDecay,alpha,gamma,trainAgent);
agent = Agent('player',learningAlgorithm,numEpisodes,AGENT_LOCATION);
gridWorld = Environment(lengthGrid,numEpisodes,maxIterations);
display = Display(showEvery);

theta = rand(20,1) * 0.01 - 0.005;  % init coefficients to random +/- 0.005

% Train the agent
if (learningAlgorithm.trainAgent)
    for episode = 1:gridWorld.numEpisodes                                                           

        % Add players to the grid world. Set new initial locations on grid.
        % Initialize other parameters.
        gridWorld = gridWorld.addPlayer(agent);                                                     
        gridWorld = gridWorld.addPlayer(food);
        gridWorld = gridWorld.addPlayer(enemy); 
        
        % Init eligibility trace. One trace vector for each possible action.
        eTrace = zeros(20,4);                 

        % Feature selection. 
        agent = agent.buildFeatures(food,lengthGrid);  
                   
        % Compute Q-value for all 4 possible actions
        Q = agent.phi' * theta; 
        oldPhi = agent.phi;
         
        % Select the best action based on static epsilon-greedy policy 
        [~,agent.action] = max(Q);
        [learningAlgorithm, agent] = learningAlgorithm.policy(agent);

        for iteration = 1:gridWorld.maxIterations                                                   
            
            % Take the action by moving the agent on the grid; observe the 
            % reward, and observe the new state, S_prime.            
            [agent, gridWorld] = takeAction(agent,gridWorld); 
            
            % Update (by deteriation) traces
            eTrace = gamma * lambda * eTrace;
            
            % Accumulate trace 
            eTrace(:,agent.action) = eTrace(:,agent.action) + ones(20,1);
            
            % Observe reward
            if (agent.x == 2 || agent.x == 3 || agent.x == 4 || agent.x == 5 || agent.x == 6 || agent.x == 7) && agent.y == 8  
                agent.iterationReward = enemy.value;                            % -100 fall off cliff
            elseif agent.x == food.x && agent.y == food.y
                agent.iterationReward = food.value;                             % +0 reach food goal
            else
                agent.iterationReward = move.value;                             % -1 move penalty
            end
            agent.episodeReward = agent.episodeReward + agent.iterationReward;  % Housekeeping
            
            % Select action a_prime using policy based on Q_new_action
            agent = agent.buildFeatures(food,lengthGrid);  
            Q_new = agent.phi' * theta;                                         % over all 4 possible actions
            
            % Select the best action based on static epsilon-greedy policy 
            [~,agent.newAction] = max(Q_new);                            
            [learningAlgorithm, agent] = learningAlgorithm.policy(agent);                
            
            % find delta 
            delta = agent.iterationReward + gamma * Q_new(agent.newAction) - Q(agent.action);  
                 
            % Update weights
            theta = theta + alpha * delta * oldPhi(:,agent.action) .* eTrace(:,agent.action);
%            theta = theta + alpha * delta * oldPhi(:,agent.action);
%            theta = theta + alpha * delta * agent.phi(:,agent.newAction);
            
            Q = Q_new;
            oldPhi = agent.phi;
            agent.action = agent.newAction;
            
            % Terminate if agent walks over cliff or obtains food
            if (agent.iterationReward == FOOD_REWARD) || (agent.iterationReward == CLIFF_PENALTY)
                break
            end            
            
        end % for iteration
        
        agent.rewardHistory(episode) = agent.rewardHistory(episode) + agent.episodeReward;
        learningAlgorithm.epsilon = learningAlgorithm.epsilon * learningAlgorithm.epsilonDecay;

        % Clean the environment
        [gridWorld,agent] = gridWorld.cleanWorld(agent); 

        if (mod(episode,1000)==0)
            disp(episode)
        end
        
    end % for episode
end % if (learningAlgorithm.trainAgent)
display.plotResults(gridWorld,agent,episode);

%% Functions that need to be added to classes

function [agent, environment] = takeAction(agent,environment)    
    previousX = agent.x;
    previousY = agent.y;
    [isValid, agent] = movePlayer(agent,environment.lengthGrid);
    choices_pool = {'up', 'down','left','right'}; % need to put this code into the called function. Too cumbersom here. 
    while(~isValid)
        agent.choice     = choices_pool{randi(4)};
        [isValid, agent] = movePlayer(agent,environment.lengthGrid);
    end
    environment.grid(previousY,previousX) = 0; % delete previous location
    environment.grid(agent.y,agent.x) = 3;   
end % takeAction

%%

function [isValid, agent] = movePlayer(agent,lengthGrid)
    % Move the player. If the player hits the edge of the grid,
    % then set the movement isValid = false. The calling function
    % then calls this function again until isValid = true. 
    isValid = true;
    if isequal(agent.choice,'right')
        if agent.x + 1 <= lengthGrid
            agent.x = agent.x + 1;
            agent.action = 4;
        else
            isValid = false;
        end
    elseif isequal(agent.choice,'left')
        if agent.x - 1 > 0
            agent.x = agent.x - 1;
            agent.action = 3;
        else
            isValid = false;
        end
    elseif isequal(agent.choice,'up')
        if agent.y - 1 > 0
            agent.y = agent.y - 1;
            agent.action = 1;
        else
            isValid = false;
        end
    elseif isequal(agent.choice,'down')
        if agent.y + 1 <= lengthGrid
            agent.y = agent.y + 1;
            agent.action = 2;
        else
            isValid = false;
        end
    end
end % move
