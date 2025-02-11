import os 
import shutil
from concurrent.futures import ThreadPoolExecutor

old_path = "/home/emmanuel/ESG/ECP/Analytics/eguzmanv/client_requests/20250211_event/archive/sds"
new_path = "/home/emmanuel/ecp_archive/SEISCOMP"

# for dirpath, dirnames, filenames in os.walk(old_path):
#     for filename in filenames:
#         full_path = os.path.join(dirpath, filename)
#         new_full_path = full_path.replace(old_path,new_path)
#         if not os.path.isdir(os.path.dirname(new_full_path )):
#             os.makedirs(os.path.dirname(new_full_path ))
        
#         shutil.copy(full_path,new_full_path)
#         print(full_path,new_full_path)

def do_copy(args):
    # shutil.copy(full_path,new_full_path)
    full_path,new_full_path = args
    if not os.path.isdir(os.path.dirname(new_full_path )):
        os.makedirs(os.path.dirname(new_full_path ))
    shutil.copy(full_path,new_full_path)
    print(full_path,new_full_path)

paths = []
for dirpath, dirnames, filenames in os.walk(old_path):
    for filename in filenames:
        full_path = os.path.join(dirpath, filename)
        new_full_path = full_path.replace(old_path,new_path)
        paths.append((full_path,new_full_path))
        
with ThreadPoolExecutor() as executor:
    executor.map(do_copy,paths) 
        