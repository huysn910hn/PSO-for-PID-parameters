% F = x^2 + y^2 - 2*x -2*y -6 
syms x y
w = 0.5;
r1 = 0.5;
r2 = 0.5;
c1 = 2;
c2 = 2;
d = 2;
N = 10;
t = 100;
X = [-3 2; 2 2; 3 -3; -2 2; -1 1; -0.5 1; -4 3; -1.5 3; -2 0.5; -1 2.5];
V = [0 0; 0 0; 0 0; 0 0; 0 0; 0 0; 0 0; 0 0; 0 0; 0 0];
for a = 1:N
    F(a) = X(a,1)^2 + X(a,2)^2 - 2*X(a,1) - 2*X(a,2) - 6;
end
Pbest = X;
F_Pbest = F;
[gbest, idx] = min(F);
X_best = X(idx,:);
Gbest = repmat(X_best, N, 1);

% vẽ chuyển động
figure; hold on; grid on;
colors = lines(N);
traject = cell(N,1);
best_history = zeros(t,1); 

for i = 1:t
    for j = 1:N
        F(j) = X(j,1)^2 + X(j,2)^2 - 2*X(j,1) - 2*X(j,2) - 6;
        if F(j) < F_Pbest(j)
            Pbest(j,:) = X(j,:);
            F_Pbest(j) = F(j);
        end
    end
    [gbest, idx] = min(F_Pbest);
    X_best = X(idx,:);
    Gbest = repmat(X_best, N, 1);
    V = w*V + c1*r1*(Pbest - X) + c2*r2*(Gbest- X);
    X = X + V;
    best_history(i) = gbest;
    fprintf("Vòng %d: gbest = %.4f, Gbest = [%.4f %.4f]\n", i, gbest, Gbest(1,1), Gbest(1,2));
    for j = 1:N
        traject{j}(end+1,:) = X(j,:);
    end

    % ---- Vẽ từng vòng ----
    cla;
    for j = 1:N
        plot(traject{j}(:,1), traject{j}(:,2), '-', 'Color', colors(j,:), 'LineWidth',1);
        scatter(X(j,1), X(j,2), 40, colors(j,:), 'filled');
    end
    scatter(Gbest(1,1), Gbest(1,2), 100, 'r', 'p', 'filled'); % gbest
    title(sprintf("PSO Iteration %d", i));
    xlabel("x"); ylabel("y");
    axis([-6 6 -6 6]); % giới hạn cho đẹp
    pause(0.1); % để nhìn thấy chuyển động
end
figure;
plot(1:t, best_history, 'b-', 'LineWidth', 2);
xlabel('Iteration');
ylabel('Best Fitness (gbest)');
title('PSO Convergence Curve');
grid on;



