classdef SARSALearning
    
    % A SARSA-learning reinforcement learning algorithm
    
    properties
        
        type                % text string ('Q-learning,' etc)
        trainAgent          % BOOL true if training is on
        gamma               % discount of future rewards
        alpha               % weight of error term
        epsilon             % probability of exploration
        epsilonDecay        % rate at which exploration decreases
        qTable              % quality table Q(numStates, numActions) 
        Q                   % quality of state-action
        
    end % properties
    
    methods
        
        function obj = SARSALearning(epsilon,epsilonDecay,alpha,gamma,trainAgent) % constructor            
            obj.epsilon = epsilon;
            obj.epsilonDecay = epsilonDecay;
            obj.alpha = alpha;
            obj.gamma = gamma;           
            obj.type = 'SARSA-Lambda-lin-fcn';
            obj.trainAgent = trainAgent;
        end
        
        function obj = reset(obj,epsilon)
            obj.epsilon = epsilon;
        end
      
        function [obj, agent] = policy(obj, agent)   
            if rand(1) > obj.epsilon                 
                agent.choice = obj.act2choice(agent.action);
            else
                agent.action = randi(4,1);
                agent.choice = obj.act2choice(agent.action);
            end                    
        end % chooseAction
        
        function [choice] = act2choice(~,act)
            % Converts an action into a choice string.
            if act == 1
                choice = 'up';
            elseif act == 2
                choice = 'down';
            elseif act == 3
                choice = 'left';
            elseif act == 4
                choice = 'right';
            end
        end
            
    end % methods
        
end % class

