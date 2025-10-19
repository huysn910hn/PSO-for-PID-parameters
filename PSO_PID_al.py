from transfer_function import *
import numpy as np
import matplotlib.pyplot as plt

w = 0.5
r1 = 0.5
r2 = 0.5
c1 = 2
c2 = 2
N = 20 # population size
T = 100 # iteration times
dim = 3
Kp_range = [0, 50]
Kd_range = [0, 10]
Ki_range = [0, (12+2*max(Kd_range))*(20.02+2*max(Kp_range))/2]
X = np.zeros((N, T, dim))
V = np.zeros((N, T, dim))

X = np.random.uniform(low=[Kp_range[0], Ki_range[0], Kd_range[0]], high=[Kp_range[1], Ki_range[1], Kd_range[1]], size=(N, dim))
V = np.zeros_like(X)

def fitness_function(params, mode):
    Kp, Ki, Kd = params
    t, y, error = simulate_step_response(Kp, Ki, Kd)
    if mode == 'IAE':
        return f_IAE(error, t)
    elif mode == 'ITAE':
        return f_ITAE(error, t)
    elif mode == 'MSE':
        return f_MSE(error, t)
    else:
        raise ValueError("Invalid mode. Choose 'IAE', 'ITAE', or 'MSE'.")
    
fitness_history_all = {}
best_params_all = {}
for metrics in ['IAE', 'ITAE', 'MSE']:
    print(f"\n Running PSO for {metrics}")
    fitness_history = []
    pbest = X.copy()
    pbest_value = np.array([fitness_function(ind, mode = metrics) for ind in pbest])
    gbest = pbest[np.argmin(pbest_value)]
    gbest_value = np.min(pbest_value)
    fitness_history.append(gbest_value)
    for t in range(T):
        for i in range(N):
            V[i] = (w * V[i] + 
                    c1 * r1 * (pbest[i] - X[i]) + 
                    c2 * r2 * (gbest - X[i]))
            X[i] = X[i] + V[i]
            X[i] = np.clip(X[i], [Kp_range[0], Ki_range[0], Kd_range[0]], [Kp_range[1], Ki_range[1], Kd_range[1]])
            current_value = fitness_function(X[i], mode=metrics)
            if current_value < pbest_value[i]:
                pbest[i] = X[i]
                pbest_value[i] = current_value
                if current_value < gbest_value:
                    gbest = X[i]
                    gbest_value = current_value
        fitness_history.append(gbest_value)
    print(f"Iteration {t+1}/{T}, Best Fitness: {gbest_value}")   
    fitness_history_all[metrics] = fitness_history.copy()
    best_params_all[metrics] = gbest.copy()
    print('Best Fitness:')
    print(f" {gbest_value}")
    print('Best PID parameters:')
    print(f" Kp: {gbest[0]}, Ki: {gbest[1]}, Kd: {gbest[2]}")
    print(best_params_all)


plt.plot(fitness_history_all['IAE'], label='IAE')
plt.plot(fitness_history_all['ITAE'], label='ITAE')
plt.plot(fitness_history_all['MSE'], label='MSE')
plt.xlabel("Iteration")
plt.ylabel("Best Fitness")
plt.title("PSO Optimization")
plt.legend()
plt.show()

plt.figure(figsize=(8,5))
for metrics in ['IAE', 'ITAE', 'MSE']:
    best_pid = best_params_all[metrics]
    t, y, _ = simulate_step_response(best_pid[0], best_pid[1], best_pid[2])
    plt.plot(t, y, label=f"{metrics} (Kp={best_pid[0]:.2f}, Ki={best_pid[1]:.2f}, Kd={best_pid[2]:.2f})")

plt.title("Step Response Comparison of Optimized PID Controllers")
plt.xlabel("Time (s)")
plt.ylabel("Output")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()