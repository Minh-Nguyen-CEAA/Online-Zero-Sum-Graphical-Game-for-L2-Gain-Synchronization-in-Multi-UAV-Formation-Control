%%% e_{i,1}
figure('Color','w')
ei1=out.Error_1;
plot(ei1.Time, ei1.Data(:,1),'r','LineWidth',1.5); hold on
plot(ei1.Time, ei1.Data(:,2),'b','LineWidth',1.5); hold on
plot(ei1.Time, ei1.Data(:,3),'y','LineWidth',1.5); hold on
plot(ei1.Time, ei1.Data(:,4),'g','LineWidth',1.5);

xlabel('Time (s)','Color','k')
ylabel('e_{i,1}(t)','Color','k')

lgd = legend('Agent 1','Agent 2','Agent 3', 'Agent 4');
lgd.Color = 'white';
lgd.TextColor = 'black';

grid on
set(gca,'Color','w','XColor','k','YColor','k')
xlim([0 max(ei1.Time)])  

%%% e_{i,2}
figure('Color','w')
ei2=out.Error_2;
plot(ei2.Time, ei2.Data(:,1),'r','LineWidth',1.5); hold on
plot(ei2.Time, ei2.Data(:,2),'b','LineWidth',1.5); hold on
plot(ei2.Time, ei2.Data(:,3),'y','LineWidth',1.5); hold on
plot(ei2.Time, ei2.Data(:,4),'g','LineWidth',1.5);

xlabel('Time (s)','Color','k')
ylabel('e_{i,2}(t)','Color','k')

lgd = legend('Agent 1','Agent 2','Agent 3', 'Agent 4');
lgd.Color = 'white';
lgd.TextColor = 'black';

grid on
set(gca,'Color','w','XColor','k','YColor','k')
xlim([0 max(ei2.Time)])  

%%% xi1
figure('Color','w')
xi1=out.State_1;
plot(xi1.Time, xi1.Data(:,1),'m','LineWidth',1.5); hold on
plot(xi1.Time, xi1.Data(:,2),'r','LineWidth',1.5); hold on
plot(xi1.Time, xi1.Data(:,3),'g','LineWidth',1.5); hold on
plot(xi1.Time, xi1.Data(:,4),'y','LineWidth',1.5); hold on
plot(xi1.Time, xi1.Data(:,5),'g','LineWidth',1.5); hold on

xlabel('Time (s)','Color','k')
ylabel('x_{i,1}(t)','Color','k')

lgd = legend('Leader','Agent 1','Agent 2', 'Agent 3', 'Agent 4');
lgd.Color = 'white';
lgd.TextColor = 'black';

grid on
set(gca,'Color','w','XColor','k','YColor','k')
xlim([0 max(xi1.Time)])  

%%% xi2
figure('Color','w')
xi2=out.State_2;
plot(xi2.Time, xi2.Data(:,1),'m','LineWidth',1.5); hold on
plot(xi2.Time, xi2.Data(:,2),'r','LineWidth',1.5); hold on
plot(xi2.Time, xi2.Data(:,3),'b','LineWidth',1.5); hold on
plot(xi2.Time, xi2.Data(:,4),'y','LineWidth',1.5); hold on
plot(xi2.Time, xi2.Data(:,5),'g','LineWidth',1.5); 

xlabel('Time (s)','Color','k')
ylabel('x_{i,2}(t)','Color','k')

lgd = legend('Leader','Agent 1','Agent 2', 'Agent 3', 'Agent 4');
lgd.Color = 'white';
lgd.TextColor = 'black';

grid on
set(gca,'Color','w','XColor','k','YColor','k')
xlim([0 max(xi2.Time)])  

%%% Hình thứ 5: biểu đồ pha (x_{i,1} vs x_{i,2})
figure('Color','w')
plot(xi1.Data(:,2), xi2.Data(:,2),'r--','LineWidth',1.5); hold on   % Agent 1
plot(xi1.Data(:,3), xi2.Data(:,3),'b--','LineWidth',1.5); hold on   % Agent 2
plot(xi1.Data(:,2), xi2.Data(:,4),'y--','LineWidth',1.5); hold on   % Agent 3
plot(xi1.Data(:,3), xi2.Data(:,5),'g--','LineWidth',1.5); hold on   % Agent 4
plot(xi1.Data(:,1), xi2.Data(:,1),'m','LineWidth',1.5);           % Leader

xlabel('x_{i,1}(t)','Color','k')
ylabel('x_{i,2}(t)','Color','k')

lgd = legend('Agent 1','Agent 2','Agent 3','Agent 4','Leader');
lgd.Color = 'white';
lgd.TextColor = 'black';

grid on
set(gca,'Color','w','XColor','k','YColor','k')
%%% Hình 6: biểu đồ pha 3D
figure('Color','w')

% Leader
plot3(xi1.Data(:,1), xi1.Time, xi2.Data(:,1), 'm','LineWidth',1.5); hold on
% Agent 1
plot3(xi1.Data(:,2), xi1.Time, xi2.Data(:,2), 'r--','LineWidth',1.5); hold on
% Agent 2
plot3(xi1.Data(:,3), xi1.Time, xi2.Data(:,3), 'b-.','LineWidth',1.5); hold on
% Agent 3
plot3(xi1.Data(:,2), xi1.Time, xi2.Data(:,4), 'y--','LineWidth',1.5); hold on
% Agent 4
plot3(xi1.Data(:,3), xi1.Time, xi2.Data(:,5), 'g-.','LineWidth',1.5);


xlabel('x_{i,1}(t)','Color','k')        % trục X
ylabel('Time (s)','Color','k')   % trục Y
zlabel('x_{i,2}(t)','Color','k')        % trục Z

lgd = legend('Leader','Agent 1','Agent 2','Agent 3','Agent 4');
lgd.Color = 'white';
lgd.TextColor = 'black';

grid on
set(gca,'Color','w','XColor','k','YColor','k','ZColor','k')
view(45,90) % góc nhìn nghiêng để thấy rõ quỹ đạo

%Weight NN
for agent = 1:4
    % Lấy dữ liệu timeseries (9 x 1 x N)
    raw = out.(['Weight_NN_' num2str(agent)]).Data;   % 9x1xN
    t   = out.(['Weight_NN_' num2str(agent)]).Time;   % Nx1
    
    % Chuyển về ma trận N×9 (mỗi hàng là một thời điểm, mỗi cột là một trọng số)
    W = permute(raw,[3 1 2]);   % từ 9x1xN -> N×9
    
    % Critic: 3 phần tử đầu
    Wc = W(:,1:3);
    % Actor: 3 phần tử tiếp theo
    Wa = W(:,4:6);
    
    figure('Color','w')
    
    % Critic
    subplot(1,2,1)
    plot(t, Wc(:,1),'r','LineWidth',1.5); hold on
    plot(t, Wc(:,2),'b','LineWidth',1.5); hold on
    plot(t, Wc(:,3),'g','LineWidth',1.5);
    xlabel('Time (s)','Color','k')
    ylabel('W_{c}(t)','Color','k')
    legend('W_{c1}','W_{c2}','W_{c3}','Location','best','Color','w','TextColor','k')
    title(['Parameters of the NN Critic of Agent ' num2str(agent)])
    grid on
    set(gca,'Color','w','XColor','k','YColor','k')
    xlim([0 max(t)])
    
    % Actor
    subplot(1,2,2)
    plot(t, Wa(:,1),'r','LineWidth',1.5); hold on
    plot(t, Wa(:,2),'b','LineWidth',1.5); hold on
    plot(t, Wa(:,3),'g','LineWidth',1.5);
    xlabel('Time (s)','Color','k')
    ylabel('W_{a}(t)','Color','k')
    legend('W_{a1}','W_{a2}','W_{a3}','Location','best','Color','w','TextColor','k')
    title(['Parameters of the NN Actor of Agent ' num2str(agent)])
    grid on
    set(gca,'Color','w','XColor','k','YColor','k')
    xlim([0 max(t)])
end




