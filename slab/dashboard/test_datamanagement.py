from dashboard import SlabFileRemote
import time
import numpy as np

def test():
    f = SlabFileRemote(filename=r"S:\Phil\some_file.h5", autosave=True)
    x = np.linspace(0, 3, 50)
    #f.manager.clear_all_data()
    f.create_dataset('do not plot', plot=False)
    f.create_dataset('do not save', save=False)
    f['do not plot'] = (x, np.sin(x))
    f['do not save'] = (x, np.cos(x))
    f['test group']['test ds'] = np.tan(np.arange(50))
    f['test group']['test img'] = np.outer(np.sin(x), np.cos(x))
    f['very loooooooooooooooooooong group name']['ds'] = np.sin(x)
    f['test group']['test img'].set_range(x0=1, xscale=.02, y0=2, yscale=.02)
    f['test group']['test img'].set_labels(xlabel='XLABEL?', ylabel='ylabel...', zlabel='Something?')
    f['parametric'] = (x*np.sin(x), x*np.cos(x))

    for i in range(50):
        time.sleep(.05)
        f['appending']['parametric'].set_labels(xlabel='sin', ylabel='cos')
        f['appending']['line'].append(np.random.normal())
        f['appending']['parametric'].append((i*np.sin(i/20.), i*np.cos(i/20.)))
        f['appending']['img'].append(np.random.normal(size=100), show_most_recent=False)

    #assert np.array(f['test_ds']).shape == (2, 50)

if __name__ == "__main__":
    test()
    print 'done'