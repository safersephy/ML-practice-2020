function plotData(x, y)
%PLOTDATA Plots the data points x and y into a new figure 

figure; % open a new figure window

plot(x, y, 'rx', 'MarkerSize', 10); % Plot the data 
ylabel('label y'); % Set the y−axis label
xlabel('label x'); % Set the x−axis label

end
