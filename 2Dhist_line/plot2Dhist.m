hist = importdata("2dhist.txt");
fit = importdata('fitting.txt');
x= [0.01 : 0.01 : 0.6];
y = [130:-5:5];
figure(1)
imagesc(x,y, hist)
set(gca,'YDir','normal')
colormap(PYCM().Blues())
colorbar
caxis([0, 25])
set(gca,'LineWidth',2,'FontSize',20)
ylabel('Time / min', 'Interpreter','tex','FontSize',30)
xlabel('Length distribution / nm', 'Interpreter', 'tex','FontSize',30)

% 控制图片大小
set(gcf,'unit','centimeters','position',[2 2 27 20])
% 单位为厘米，大小为25cm×20cm

set(gca,'OuterPosition',[.05 .05 .95 .95]);

hold on
% plot(fit(1,:), fit(2,:),'color',[0/255 176/255 80/255],'LineWidth',3)
plot(fit(1,:), fit(2,:),'color','#EDB120','LineWidth',4,'LineStyle','--')
% xlim([-3.3,-2.2])
% ylim([5,130])