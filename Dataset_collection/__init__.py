import os
import sys

from preprocessing import label
from preprocessing import slice
from preprocessing import utils
from binaryninja import BinaryView


class WrapperView(BinaryView):
    name = "Dummy Wrapper"
    """ Dummy view to wrap the initialization of a BinaryView """
    def __init__(self, data):
        BinaryView.__init__(self, parent_view = data, file_metadata = data.file)
        self.raw = data

    @classmethod
    def is_valid_for_data(self, data):
        path = os.path.dirname(data.file.filename) + "/bninja_scripts"
        if path not in sys.path:
            print("LocalScriptLoader: Adding {} to path".format(
                    path
                )
            )
            sys.path.append(path)

        # Just a wrapper. Always fail
        return False

WrapperView.register()