import sys
from filelist import FileList

if __name__ == '__main__':
    fl = FileList()
    args = sys.argv
    if len(args) == 3:
        root_dir = args[1]
        ext = args[2]
    else:
        root_dir = '.'
        ext = '.py'

    fl.setup(root_dir, ext)
    fl.print()
