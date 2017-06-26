import sys
import os
from PyQt5.QtWidgets import (QWidget, QApplication, QPushButton, QLineEdit,
                             QHBoxLayout, QVBoxLayout, QTextEdit, QProgressBar,
                             QFileDialog)
from PyQt5.QtCore import pyqtSlot, Qt
from filelist import FileList


class MainWidget(QWidget):
    dirname = ''
    step = 0

    def __init__(self, parent=None):
        super(MainWidget, self).__init__(parent)
        self.initUI()
        self.fileList = FileList()
        self.fileList.sig_file.connect(self.update_status)
        self.fileList.finished.connect(self.finish_process)

    def initUI(self):
        self.resize(480, 360)

        self.txtFolder = QLineEdit()
        self.txtFolder.setReadOnly(True)
        self.btnFolder = QPushButton('参照...')
        self.btnFolder.clicked.connect(self.show_folder_dialog)
        hb1 = QHBoxLayout()
        hb1.addWidget(self.txtFolder)
        hb1.addWidget(self.btnFolder)

        self.btnExec = QPushButton('実行')
        self.btnExec.clicked.connect(self.exec_process)
        self.btnExec.setEnabled(False)
        self.btnExec.setVisible(True)

        self.btnExit = QPushButton('終了')
        self.btnExit.setVisible(False)
        self.btnExit.setEnabled(False)
        self.btnExit.clicked.connect(self.close)

        hb2 = QHBoxLayout()
        hb2.addWidget(self.btnExec)
        hb2.addWidget(self.btnExit)

        self.txtLog = QTextEdit()
        self.txtLog.setReadOnly(True)

        self.pbar = QProgressBar()
        self.pbar.setTextVisible(False)

        layout = QVBoxLayout()
        layout.addLayout(hb1)
        layout.addLayout(hb2)
        layout.addWidget(self.txtLog)
        layout.addWidget(self.pbar)
        self.setLayout(layout)

        self.setWindowTitle('PyQt5 Sample')

    def show_folder_dialog(self):
        ''' open dialog and set to foldername '''
        dirname = QFileDialog.getExistingDirectory(self,
                                                   'open folder',
                                                   os.path.expanduser('.'),
                                                   QFileDialog.ShowDirsOnly)
        if dirname:
            self.dirname = dirname.replace('/', os.sep)
            self.txtFolder.setText(self.dirname)
            self.btnExec.setEnabled(True)
            self.step = 0

    def print_log(self, logstr):
        self.txtLog.append(logstr)

    @pyqtSlot()
    def exec_process(self):
        if os.path.exists(self.dirname):
            try:
                QApplication.setOverrideCursor(Qt.WaitCursor)
                self.fileList.setup(self.dirname, '.py')
                maxCnt = self.fileList.length
                self.pbar.setValue(0)
                self.pbar.setMinimum(0)
                self.pbar.setMaximum(maxCnt)
                self.fileList.start()
            except Exception as e:
                self.print_log(str(e))
            finally:
                QApplication.restoreOverrideCursor()
        else:
            self.print_log('{0} is not exists'.format(self.dirname))

    @pyqtSlot(str)
    def update_status(self, filename):
        self.txtLog.append(filename)
        self.step += 1
        self.pbar.setValue(self.step)

    @pyqtSlot()
    def finish_process(self):
        self.fileList.wait()
        self.btnExec.setEnabled(False)
        self.btnExec.setVisible(False)
        self.btnExit.setEnabled(True)
        self.btnExit.setVisible(True)

def main(args):
    app = QApplication(args)
    dialog = MainWidget()
    dialog.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main(sys.argv)
