from dataserver_helpers import run_dataserver, dataserver_client, DATA_DIRECTORY
import time
from PyQt4 import Qt
import sys

app = Qt.QApplication([])
win = Qt.QMainWindow()
contents = Qt.QWidget()
contents.setLayout(Qt.QVBoxLayout())
status_label = Qt.QLabel('Not Running')
start_button = Qt.QPushButton('Start')
exit_button = Qt.QPushButton('Quit')
exit_button.setEnabled(False)
contents.layout().addWidget(status_label)
contents.layout().addWidget(start_button)
contents.layout().addWidget(exit_button)
win.setCentralWidget(contents)
client = None
alive_poll = None
alive_time = 500

def start():
    run_dataserver(qt=True)
    time.sleep(1)
    global client, alive_poll, start_button, exit_button
    client = dataserver_client()
    alive_poll = Qt.QTimer()
    alive_poll.timeout.connect(check_alive)
    alive_poll.start(alive_time)
    start_button.setEnabled(False)
    exit_button.setEnabled(True)

# This is somewhat useless now. Keep for multiprocessing
def check_alive():
    global client, alive_poll
    if client.hello() != "hello":
        status_label.setText("Not Running")
        client = None
        alive_poll.stop()
    else:
        status_label.setText('Server running at ' + DATA_DIRECTORY)

def stop():
    global client
    if client is not None:
        client.quit()
    start_button.setEnabled(True)
    exit_button.setEnabled(False)

start_button.clicked.connect(start)
exit_button.clicked.connect(stop)
app.lastWindowClosed.connect(stop)
win.show()
sys.exit(app.exec_())
