%% Plot linear test data
load("linear_test.mat");
fig1 = figure;fig2 = figure;fig3 = figure;
close(fig1);close(fig2);close(fig3)

fig1 = figure;
plot(TimeSequence', squeeze(StateSequence)');
fig2 = figure;
plot(TimeSequence', squeeze(InputSequence)');
fig3 = figure;
plot(xcorr(squeeze(StateSequence(:,1,:)), squeeze(InputSequence)))

%% Plot Van Der Pol Oscillator test data
load("Korda2018.mat");
fig1 = figure;fig2 = figure;fig3 = figure;
close(fig1);close(fig2);close(fig3)

fig1 =figure;
plot(squeeze(StateSequence(:,1,:))', squeeze(StateSequence(:,2,:))')
fig2 = figure;
plot(TimeSequence', squeeze(StateSequence(1,2,:)));
hold on
plot(TimeSequence', squeeze(InputSequence(1,:,:)));
hold off


