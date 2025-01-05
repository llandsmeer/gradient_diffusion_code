import os
os.environ['JAX_PLATFORM'] = 'cpu'
import tqdm
import numpy as np
import pickle
import matplotlib.pyplot as plt

save = True

x = []
y = []
t = []
i = 0
while True:
    fn = f"./OPT_adam_['1e-2', '1000', 'adam']_opt_{i:03d}.pkl"
    if not os.path.exists(fn):
        break
    tt = os.path.getmtime(fn)
    t.append(tt)
    with open(fn, 'rb') as f:
        data = pickle.load(f)
        mse = ((data['v'][:,1] - data['vtgt'])**2).mean()
        x.append(i)
        y.append(mse)
        if i in [0, 300]:
            plt.plot(data['v'][:,1], color='blue')
            plt.plot(data['vtgt'], color='black')
            plt.xlabel('Time (ms)')
            plt.ylabel('V (ms)')
            if save:
                plt.savefig(f'out/grads_trace_f-{i:03d}.svg')
            plt.clf()
    i += 1
    if i > 300:
        break
print(y)

# x = []
# y = []
# t = []
# i = 0
# while True:
#     fn = f"./OPT_adam_['1e-3', '1000', 'adam']_opt_{i:03d}.pkl"
#     if not os.path.exists(fn):
#         break
#     tt = os.path.getmtime(fn)
#     t.append(tt)
#     with open(fn, 'rb') as f:
#         data = pickle.load(f)
#         mse = ((data['v'][:,1] - data['vtgt'])**2).mean()
#         x.append(i)
#         y.append(mse)
#         # if i in [0, 74]:
#         #     plt.plot(data['v'][:,1], color='blue')
#         #     plt.plot(data['vtgt'], color='black')
#         #     plt.xlabel('Time (ms)')
#         #     plt.ylabel('V (ms)')
#         #     if save:
#         #         plt.savefig(f'out/grads_trace_f-{i:03d}.svg')
#         #     plt.clf()
#     i += 1
# print(y)
# plt.plot(x, y, '--', color='red')

xevo = []
yevo = []
tevo = []
i = 0
while True:
    fn = f"./OPT_adam_['1', '1000', 'evo']_opt_{i:03d}.pkl"
    if not os.path.exists(fn):
        break
    tt = os.path.getmtime(fn)
    with open(fn, 'rb') as f:
        data = pickle.load(f)
        # mses = []
        # for j in range(20):
        #     v = data[f'v{j}'][:,1]
        #     mse = ((v - data['vtgt'])**2).mean()
        #     mses.append(mse)
        xevo.append(i)
        yevo.append(2*min(data['L']))
        tevo.append(tt)
        #for j in range(20):
            #plt.plot(data[f'v{j}'][:,1], color='blue')
        #plt.plot(data['vtgt'], color='black')
        #plt.xlabel('Time (ms)')
        #plt.ylabel('V (ms)')
        #plt.show()
    i += 1

# xevo = []
# yevo = []
# tevo = []
# i = 0
# while True:
#     fn = f"./OPT_adam_['1', '1000', 'evo', '50']_opt_{i:03d}.pkl"
#     if not os.path.exists(fn):
#         break
#     tt = os.path.getmtime(fn)
#     with open(fn, 'rb') as f:
#         data = pickle.load(f)
#         # mses = []
#         # for j in range(20):
#         #     v = data[f'v{j}'][:,1]
#         #     mse = ((v - data['vtgt'])**2).mean()
#         #     mses.append(mse)
#         xevo.append(i)
#         yevo.append(2*min(data['L']))
#         tevo.append(tt)
#         #for j in range(20):
#             #plt.plot(data[f'v{j}'][:,1], color='blue')
#         #plt.plot(data['vtgt'], color='black')
#         #plt.xlabel('Time (ms)')
#         #plt.ylabel('V (ms)')
#         #plt.show()
#     i += 1

print(f'{np.mean(np.diff(t))}+{np.std(np.diff(t))}')
plt.title(f'{np.mean(np.diff(t))}+{np.std(np.diff(t))}')
plt.plot(x, y, '-', color='black')
plt.plot(xevo, yevo, '-')

if save:
    plt.savefig('out/grads_trace_decay.svg')
else:
    plt.show()
