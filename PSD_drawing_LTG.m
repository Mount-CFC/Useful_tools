% 更改图片的大小等，用于PSD
% 控制colorbar，只需要右键colorbar-属性检查器即可。

%% PSD = 1.0
figure(1)

% 坐标范围
xlim([-3.3,-2.2])
ylim([-5,-3])
set(gca,'LineWidth',2,'FontSize',20)
ylabel('Noise Power /\itG \rm(log \itG\rm_0)', 'Interpreter','tex','FontSize',30)
xlabel('Conductance / log (\itG/\itG\rm_0)', 'Interpreter', 'tex','FontSize',30)

% 控制图片大小
set(gcf,'unit','centimeters','position',[2 2 25 20])
% 单位为厘米，大小为25cm×20cm

set(gca,'OuterPosition',[.05 .05 .95 .95]);
% 图形在figure中所占的比例

%% PSD归一化的图
%填入图号
figure(2)

% 坐标范围
xlim([-3.4,-2])
ylim([-4.8,-1.5])
set(gca,'LineWidth',2,'FontSize',20)
xlabel('Conductance / log (\itG/\itG\rm_0)', 'Interpreter', 'tex','FontSize',30)

% y轴，G的几次方
ylabel('Noise Power /\itG \rm(log \itG\rm_0^{1.1})', 'Interpreter','tex','FontSize',30)
set(gcf,'unit','centimeters','position',[2 2 25 20])
set(gca,'OuterPosition',[.05 .05 .95 .95]);

