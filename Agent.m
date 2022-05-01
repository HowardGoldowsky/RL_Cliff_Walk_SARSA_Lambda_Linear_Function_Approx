classdef Agent
    
    properties
        
        x                       % INT x-coordinate on grid
        y                       % INT y-coordinate on grid
        ID                      % INT agent ID 
        type                    % STRING text string
        learningAlgorithm       % LEARNING OBJECT 
        observation
        action
        newAction
        rewardHistory
        iterationReward
        episodeReward           
        choice                  % choice of movement direction (left, right, down up)
        phi                     % feature vector
        
    end
    
    methods
        
        function obj = Agent(type,learningAlgorithm,numEpisodes,location)  % constructor    
            obj.type = type;
            obj.learningAlgorithm = learningAlgorithm;
            obj.iterationReward = 0;
            obj.episodeReward = 0;
            obj.rewardHistory = zeros(numEpisodes,1);
            obj.x = location(2);
            obj.y = location(1);
            obj.phi = zeros(20,4);
        end % constructor
        
        function obj = buildFeatures(obj,food,lengthGrid)
            % Agent observes how many cells it is away from the positive
            % and negative reward cells.
            
            switch(obj.x)
                case 1
                    cliffHorizDist = -1;
                case {2,3,4,5,6,7}
                    cliffHorizDist = 0;
                case 8
                    cliffHorizDist = 1;
            end
            
            foodHorizDist = obj.x - food.x;
            foodVertDist  = obj.y - food.y;                   
            cliffVertDist = obj.y - lengthGrid;
            foodDiff  = [foodHorizDist, foodVertDist];   
            cliffDiff = [cliffHorizDist, cliffVertDist];   
            obj.observation = [foodDiff, cliffDiff];% + lengthGrid; % This results in a positive number from 1 to lengthGrid-1.
            
            % Build feature vectors
            obj.phi(1,1) = 0;%foodHorizDist;       % up
            obj.phi(2,1) = sign(foodVertDist);
            obj.phi(3,1) = 0;%cliffHorizDist;
            obj.phi(4,1) = -sign(cliffVertDist);
            obj.phi(5,1) = 1;
            
            obj.phi(6,2) = 0;%foodHorizDist;       % down
            obj.phi(7,2) = -sign(foodVertDist);
            obj.phi(8,2) = 0;%cliffHorizDist;
            obj.phi(9,2) = sign(cliffVertDist);
            obj.phi(10,2) = 1;
            
            obj.phi(11,3) = sign(foodHorizDist);      % left
            obj.phi(12,3) = 0;%foodVertDist;
            obj.phi(13,3) = -sign(cliffHorizDist);
            obj.phi(14,3) = 0;%cliffVertDist;
            obj.phi(15,3) = 1;
            
            obj.phi(16,4) = -sign(foodHorizDist);      % right
            obj.phi(17,4) = 0;%foodVertDist;
            obj.phi(18,4) = sign(cliffHorizDist);
            obj.phi(19,4) = 0;%cliffVertDist;                      
            obj.phi(20,4) = 1;
        end 
        
        function obj = reset(obj)
            % Places agent back at the starting cell after each iteration.
            obj.x = 1;                      
            obj.y = 8;                    
            obj.observation = [];                    
            obj.iterationReward = 0;
            obj.episodeReward = 0;
        end    
    end
end

