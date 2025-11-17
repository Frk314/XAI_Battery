import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
#%%
SMALL_SIZE = 6
MEDIUM_SIZE = 10
BIGGER_SIZE = 12
try:
    plt.rc('text', usetex = False)
except:
    print('No Tex!')
plt.rc('font', size=SMALL_SIZE,family='serif')          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE,linewidth=1)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def rmse(data):
    return ((data['ref'] - data['pred'])**2).mean()**0.5

def error(data):
    return (np.abs(data['ref'] - data['pred'])/data['ref'] * 100)[0]


dataset_name = 'matr_1'
# cnn
mdir = './transformer_results/'

model_name = 'transformers_1'
data_cnn_best = np.load(mdir+f'{dataset_name}/res_{model_name}.npz')

model_name = 'transformers_8'
data_cnn_worst = np.load(mdir+f'{dataset_name}/res_{model_name}.npz')

rmse_cnn_best = rmse(data_cnn_best)
rmse_cnn_worst = rmse(data_cnn_worst)

error_cnn_best = error(data_cnn_best)
error_cnn_worst = error(data_cnn_worst)


def plot_best_worst(data_best, data_worst, error_best, error_worst, index, name='qm_heatmaps', save=False):
    qd = np.flip(data_best['q_matrices'][index].squeeze(), 0)
    hmb = np.abs(np.flip(data_best['heatmaps'][index], 0))
    hmw = np.abs(np.flip(data_worst['heatmaps'][index], 0))
    eb = error_best[index]
    ew = error_worst[index]

    fig, ax = plt.subplots(1, 3, figsize=(6, 1.5), sharey=True)
    
    v = np.linspace(2.0, 3.6, num=100, endpoint=True)
    cycle = np.linspace(1.0, 100.0, num=100, endpoint=True)
    xx, yy = np.meshgrid(cycle, v)
    plt.set_cmap('jet')
    
    v = np.linspace(2.0, 3.6, num=hmb.shape[0], endpoint=True)
    cycle = np.linspace(1.0, 100.0, num=hmb.shape[1], endpoint=True)
    xxh, yyh = np.meshgrid(cycle, v)
    
    
    c0 = ax[0].pcolormesh(xx, yy, qd)
    cb0 = fig.colorbar(c0, ax=ax[0], format='%.2f')
    cb0.ax.yaxis.set_major_locator(plt.LinearLocator(numticks=5))
    # cb0.ax.set_title('$Q_n$ (Ah)')
    ax[0].set_yticks([2.0, 2.4, 2.8, 3.2, 3.6])
    ax[0].set_ylabel('Voltage (V)')
    ax[0].set_title('Capacity matrix')
    
    # plt.set_cmap('plasma')
    c1 = ax[1].pcolormesh(xxh, yyh, hmb, vmin=0, vmax=1)
    cb1 = fig.colorbar(c1, ax=ax[1], format='%.2f')
    cb1.ax.yaxis.set_major_locator(plt.LinearLocator(numticks=5))
    # cb1.ax.set_title('$Q_n$ (Ah)')
    
    v = np.linspace(2.0, 3.6, num=hmw.shape[0], endpoint=True)
    cycle = np.linspace(1.0, 100.0, num=hmw.shape[1], endpoint=True)
    xxh, yyh = np.meshgrid(cycle, v)
    
    c2 = ax[2].pcolormesh(xxh, yyh, hmw, vmin=0, vmax=1)
    cb2 = fig.colorbar(c2, ax=ax[2], format='%.2f')
    cb2.ax.yaxis.set_major_locator(plt.LinearLocator(numticks=5))
    # cb1.ax.set_title('$Q_n$ (Ah)')
    
    ax[1].set_title(f'({round(eb)} \%)')
    ax[2].set_title(f'({round(ew)} \%)')
    
    for axx in ax:
        axx.set_xticks([1, 25, 50, 75, 100])
        axx.set_xlim(1, 100)
        axx.set_ylim(2.0, 3.6)
        axx.set_xlabel('Cycle number')
        axx.tick_params(direction="in",top=True,right=True,which='both')
        axx.yaxis.set_minor_locator(tck.AutoMinorLocator())
        axx.xaxis.set_minor_locator(tck.AutoMinorLocator())
    

    if save:
        plt.savefig(f'./figs/{name}.png', dpi=300, bbox_inches = 'tight')
        # plt.savefig(f'{name}.pdf', dpi=300, bbox_inches = 'tight')
    
n_samples = data_cnn_best['q_matrices'].shape[0]
for index in range(n_samples):
    plot_best_worst(data_cnn_best, data_cnn_worst, error_cnn_best, error_cnn_worst, index, name=f'{dataset_name}_{index}', save=True)
    
#%% RUL
def plot_rul(ref, pred, std, cname, save=False):
    fig, ax = plt.subplots(figsize=(1.5, 1.5))
    ax.grid()
    c = ax.scatter(ref, pred, s=5, c=std, cmap='cool', zorder=3)
    ax.plot(ref, ref, c='tab:pink', lw=0.5, alpha=1, zorder=2)
    ax.set_xlabel('Reference')
    ax.set_ylabel('Prediction')
    ax.set_xlim(ref.min()//500 * 500, ref.max()//500 * 500 + 500)
    ax.set_ylim(ref.min()//500 * 500, ref.max()//500 * 500 + 500)
    ax.tick_params(direction="in",top=True,right=True,which='both')
    # ax.yaxis.set_minor_locator(tck.AutoMinorLocator())
    # ax.xaxis.set_minor_locator(tck.AutoMinorLocator())
    ax.set_aspect('equal', adjustable='box')    
    plt.locator_params(axis='y', nbins=4)
    plt.locator_params(axis='x', nbins=4)
    cb = fig.colorbar(c, ax=ax, format='%d', shrink=0.8)
    cb.ax.yaxis.set_major_locator(plt.LinearLocator(numticks=5))
    cb.ax.set_title('$\sigma_{\\varepsilon}(\%)$')
    if save:
        plt.savefig(f'rul_{cname}.pdf', dpi=300, bbox_inches = 'tight')
        


mdir = './transformer_results/'
preds = []
ers = []
rmses = []
for i in [1, 8]:
    model_name = f'transformers_{i}'
    data = np.load(mdir+f'{dataset_name}/res_{model_name}.npz')
    preds.append(data['pred'][0])
    ers.append(error(data))
    rmses.append(rmse(data))

ref = data['ref']
preds = np.array(preds)
ers = np.array(ers)
pred = preds.mean(0)
std = ers.std(0)

rmses = np.array(rmses)
print(round(rmses.mean()), round(rmses.std()))


plot_rul(ref, pred, std, f'transformer_{dataset_name}', save=True)
