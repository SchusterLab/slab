if __name__ == "__main__":
    from coverage import coverage
    cov = coverage(data_suffix='client')
    cov.start()

from interface import SlabFile, DataClient
import time
import numpy as np
import multiprocessing

def test_file_interface(remote=True):
    f = SlabFile(filename=r"S:\Phil\some_file.h5", remote=remote)
    x = np.linspace(0, 3, 50)
    f['do not plot'].set_data((x, np.sin(x)), plot=False)
    f['do not plot'].attrs['function'] = 'sine'
    f['do not save'].set_data((x, np.cos(x)), save=False, color='b')
    f['test group']['test ds'] = np.tan(np.arange(50))
    f['test group']['test img'] = np.outer(np.sin(x), np.cos(x))
    f['very loooooooooooooooooooong group name']['ds'] = np.sin(x)
    f['test group']['test img'].set_range(x0=1, xscale=.02, y0=2, yscale=.02)
    f['test group']['test img'].set_labels(xlabel='XLABEL?', ylabel='ylabel...', zlabel='Something?')
    #f['parametric (implicit)'] = (x*np.sin(x), x*np.cos(x))
    #f['parametric (explicit)'].set_data(np.array([x*np.cos(x), x*np.sin(x)]), parametric=True)
    f['test group']['pulse'] = np.zeros(50)
    f['test group']['pulse'][5:25] = np.ones(20)
    f['test group']['test img'][:,5:25] = np.ones(20)

    for i in range(3):
        time.sleep(.5)
        x = i
        f['par x'].append_data(x*np.sin(x))
        f['par y'].append_data(x*np.cos(x))
        f['appending']['parametric'].set_labels(xlabel='sin', ylabel='cos')
        f['appending']['line'].append_data(np.random.normal())
        f['appending']['parametric'].append_data((i*np.sin(i/20.), i*np.cos(i/20.)))
        f['appending']['img'].append_data(np.random.normal(size=100), show_most_recent=True)

    print f['test group']['test ds'][1:3]
    time.sleep(5) # Check for timeout errors
    assert np.array(f['test group']['test ds']).shape == (50,)
    print f['test group']['test img'][1, :].shape
    print np.array(f['test group']['test img']).shape


def test_gui_elements(interactive=True):
    c = DataClient()
    x = np.linspace(0, 3, 50)
    c.set_data(('A',), x*np.sin(x))
    c.set_data(('B',), x*np.cos(x))
    c.set_data(('C', 'D'), np.outer(x, x))
    c.gui_method('_test_edit_widget', ('A',))
    c.gui_method('_test_edit_widget', ('B',))
    c.gui_method('_test_show_hide', ('C', 'D'))
    c.gui_method('_test_multiplot', [('A',), ('B',)])
    c.gui_method('_test_multiplot', [('A',), ('B',)], parametric=True)
    c.set_data(('A',), x*np.cos(x)) # Now test update of multiplots
    c.set_data(('B',), x*np.sin(x))
    if interactive:
        c.gui_method('_test_save_selection', [('A',), ('B',)])
        c.gui_method('load')
    c.gui_method('remove_multiplot', ('A', 'B'))
    c.gui_method('msg', 'test string')


def test_proxy_interface():
    c = DataClient()
    c.set_data('no hierarchy', [1,2,3])
    c.set_data(('some hierarchy', 'a'), [3,2,1])
    c.set_data(('some hierarchy', 'b'), [[1,2],[3,4]])
    c.save_with_file('some hierarchy', r'S:\Phil\test_programmatic_save.h5')

def close_window():
    c = DataClient()
    c.gui_method('close')

def target():
    from coverage import coverage
    cov = coverage(data_suffix='window')
    cov.start()
    from dashboard import main
    main()
    cov.stop()
    cov.save()

if __name__ == "__main__":
    #test_file_interface(remote=False)
    import os

    interactive = True
    if os.path.exists(r'S:\Phil\some_file.h5'):
        os.remove(r'S:\Phil\some_file.h5')
    p = multiprocessing.Process(target=target)
    p.start()
    time.sleep(2)

    test_file_interface()
    test_gui_elements(interactive=interactive)
    time.sleep(1)
    if not interactive:
        close_window()
    cov.stop()
    cov.save()

    p.join()

    cov.combine()
    cov.html_report()
    print 'done'