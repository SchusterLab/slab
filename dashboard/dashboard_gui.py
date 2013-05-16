"""
A greatly minimized version of gui.py

The main benefit gui.py achieved was to create a simple method
of backgrounding processes. However, there was plenty of redundancy
as well as obscure, and unneeded features. gui2 provides far less,
but hopefully is hopefully much more usable.

As in gui, gui2 requests that you subclass two entities, now named
:py:class:`BackgroundObject` and :py:class:`SlabWindow`. The main
functionality this provides is two proxy type objects.

1.  SlabWindow.background is a proxy which allows methods of the
    BackgroundObject to be called. i.e. ``self.background.take_data()``
2.  BackgroundObject.gui is a proxy which allows methods of properties
    stored in the SlabWindow to be called. i.e.
    ``self.gui.message_box.write('hello')``
3.  BackgroundObject.

Boilerplate has been reduced as well in the process.

"""

import types
from PyQt4 import Qt


class GenericProxy(object):
    """
    Turns proxy.a['b'].c[1](d, e='f') into
    emitter.emit(sig_name, [('prop','a'),'b','c',1], d, {'e':'f'})

    Warning, not capable of returning values!
    """
    def __init__(self, sig_name,  emitter, context=[]):
        self._emitter = emitter
        self._sig_name = sig_name
        self._context = context

    def __getattr__(self, attr):
        return GenericProxy(self._sig_name, self._emitter, self._context + [('attr', attr)])

    def __getitem__(self, item):
        return GenericProxy(self._sig_name, self._emitter, self._context + [('item', item)])

    def __call__(self, *args, **kwargs):
        self._emitter.emit(Qt.SIGNAL(self._sig_name), self._context, args, kwargs)



def proxify(obj, connector, sig_name):
    emitter_obj = Qt.QObject()
    proxy = GenericProxy(sig_name, emitter_obj)
    def run_from_proxy_fn(self, context, args, kwargs):
        current = self
        for operation, arg in context:
            if operation == 'attr':
                current = getattr(current, arg)
            elif operation == 'item':
                try:
                    current = current[arg]
                except:
                    print context, current
                    raise
            else:
                raise ValueError('Unidentified operation ' + str(operation))
        current(*args, **kwargs)
    obj._run_from_proxy = types.MethodType(run_from_proxy_fn, obj, type(obj))
    def conn_fn():
        connector.connect(proxy._emitter, Qt.SIGNAL(sig_name),
                                        obj._run_from_proxy)

    return proxy, conn_fn


class BackgroundObject(Qt.QObject):
    def __init__(self):
        Qt.QObject.__init__(self)
        self.gui = None
        self.params = {}

    def set_param(self, name, value):
        self.params[name] = value
        
class SlabWindow(Qt.QMainWindow):
    def __init__(self, background_obj):
        Qt.QMainWindow.__init__(self)
        self.background_obj = background_obj
        self.background_obj.gui, connect_gui = proxify(self, self, 'gui')
        connect_gui()

        if hasattr(self, 'setupUi'):
            self.setupUi(self)

        self.background_thread = Qt.QThread()
        self.background, connect_background = proxify(self.background_obj, self, 'background')
        self.background_obj.moveToThread(self.background_thread)
        connect_background()
        self.connect(self, Qt.SIGNAL('lastWindowClosed()'), self.background_thread.exit)

    def register_param(self, name, widget):
        for klass, actions in widget_tools.items():
            if isinstance(widget, klass):
                if isinstance(actions["read"], str):
                    read_action = actions["read"]
                    read_fn = lambda: \
                        self.background_obj.set_param(name, getattr(widget, read_action)())
                else:
                    read_fn = actions["read"]
                getattr(widget, actions["change_signal"]).connect(read_fn)
                read_fn() # Call it to initialize
        
    def start_thread(self):
        self.background_thread.start()

widget_tools = {
    Qt.QSpinBox : {
        "read" : "value",
        "change_signal" : "valueChanged"},
    Qt.QDoubleSpinBox : {
        "read" : "value",
        "change_signal" : "valueChanged"},
    Qt.QLineEdit : {
        "read" : "text",
        "change_signal" : "textEdited"},
    Qt.QButtonGroup : {
        "read" : "checkedId",
        "change_signal" : "buttonClicked"},
    Qt.QCheckBox : {
        "read" : "isChecked",
        "change_signal" : "stateChanged"},
    Qt.QSlider : {
        "read" : "value",
        "change_signal" : "valueChanged"},
#        Qt.QTextEdit : {
#            "read" : "toPlainText",
#            "change_signal" :
#                lambda w, f: w.textChanged.connect(lambda: f(w.document().toPlainText()))},
#        Qt.QComboBox : {
#            "read" : "currentText",
#            "change_signal":
#               lambda w, f: w.currentIndexChanged[str].connect(f)},
#        Qt.QAction : {
#            "read" : None,
#            "change_signal":
#                lambda w, f: w.triggered.connect(lambda: f(True))},
#        Qt.QTreeWidget : {
#            "read" : "currentItem",
#            "change_signal":
#                lambda w, f:
#                w.currentItemChanged.connect(lambda cur, prev: f(cur))},
}
