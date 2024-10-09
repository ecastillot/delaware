import os
from SeisMonitor.monitor.locator.nlloc.nlloc import NLLoc
from SeisMonitor.monitor.locator import utils as lut

import platform
from SeisMonitor.monitor.locator.nlloc.utils import download_nlloc


nlloc_path = "/home/emmanuel/NLLoc"
download_nlloc(nlloc_path)
print("NLLoc now is working properly" )