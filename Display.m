classdef Display
    
    % Display class for GridWorld
    
    properties
        
        showEvery                   % display every n episodes
        hFig                        % figure handle
        
    end
    
    methods
   
        function obj = Display(showEvery) % constructor
              
            obj.showEvery = showEvery;
            obj.hFig = [];
                
        end
            
        function obj = plotResults(obj,~,agent,episode)
            figure;
            plot(1:episode,movmean(agent.rewardHistory,500),'-','linewidth',1);
            ylabel('Average Reward (over max 200 iterations)','Interpreter','latex','FontSize',13);
            xlabel('episode','Interpreter','latex','FontSize',13);
            tmpStr = sprintf('%s Cliff Walk Reward Convergence',agent.learningAlgorithm.type);
            title(tmpStr,'Interpreter','latex','FontSize',13);
        end
        
    end
end

