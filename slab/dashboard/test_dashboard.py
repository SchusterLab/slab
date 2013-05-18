from interface import SlabFileRemote, DataClient
import time
import numpy as np

def test_file_interface():
    f = SlabFileRemote(filename=r"S:\Phil\some_file.h5", autosave=True)
    x = np.linspace(0, 3, 50)
    f['do not plot'].set_data((x, np.sin(x)), plot=False)
    f['do not save'].set_data((x, np.cos(x)), save=False)
    f['test group']['test ds'] = np.tan(np.arange(50))
    f['test group']['test img'] = np.outer(np.sin(x), np.cos(x))
    f['very loooooooooooooooooooong group name']['ds'] = np.sin(x)
    f['test group']['test img'].set_range(x0=1, xscale=.02, y0=2, yscale=.02)
    f['test group']['test img'].set_labels(xlabel='XLABEL?', ylabel='ylabel...', zlabel='Something?')
    f['parametric (implicit)'] = (x*np.sin(x), x*np.cos(x))
    f['parametric (explicit)'].set_data(np.array([x*np.cos(x), x*np.sin(x)]), parametric=True)

    for i in range(50):
        time.sleep(.05)
        f['appending']['parametric'].set_labels(xlabel='sin', ylabel='cos')
        f['appending']['line'].append_data(np.random.normal())
        f['appending']['parametric'].append_data((i*np.sin(i/20.), i*np.cos(i/20.)))
        f['appending']['img'].append_data(np.random.normal(size=100), show_most_recent=True)

    #assert np.array(f['test_ds']).shape == (2, 50)


def test_proxy_interface():
    c = DataClient()
    c.set_data('no hierarchy', [1,2,3])
    c.set_data(('some hierarchy', 'a'), [3,2,1])
    c.set_data(('some hierarchy', 'b'), [[1,2],[3,4]])
    c.save_with_file('some hierarchy', r'S:\Phil\test_programmatic_save.h5')

if __name__ == "__main__":
    test_file_interface()
    #test_proxy_interface()
    #c = DataClient()
    #c.load_h5file(r'S:\Phil\some_file.h5')
    print 'done'