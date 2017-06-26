import os
import sys
from PyQt5.QtCore import pyqtSignal, QMutexLocker, QMutex, QThread


class FileList(QThread):
    ''' store file list'''

    sig_file = pyqtSignal(str)

    def __init__(self, parent=None):
        super(FileList, self).__init__(parent)
        self.stopped = False
        self.mutex = QMutex()

    def setup(self, root_dir, ext):
        self.root_dir = root_dir
        self.ext = ext
        self.retrieve()
        self.stopped = False

    def stop(self):
        with QMutexLocker(self.mutex):
            self.stopped = True

    def run(self):
        for f in self.files:
            fname = f
            self.process_file(fname)
            self.sig_file.emit(fname)
        self.stop()
        self.finished.emit()

    def retrieve(self):
        self.files = []
        for rd, _, fl in os.walk(self.root_dir):
            for f in fl:
                _, fext = os.path.splitext(f)
                if fext == self.ext:
                    self.files.append(os.path.join(rd, f))
        self.length = len(self.files)

    def process_file(self, path):
        ''' ひとまず何もしない '''
        cnt = 0
        if os.path.exists(path):
            cnt += 1
        else:
            cnt = 0

    def print(self):
        for f in self.files:
            print(f)


def main(args):
    root_dir = '.'
    ext = '.py'
    if len(args) == 3:
        root_dir = args[1]
        ext = args[2]
    fileList = FileList()
    fileList.setup(root_dir, ext)
    fileList.print()

if __name__ == '__main__':
    main(sys.argv)
