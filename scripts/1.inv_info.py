# /**
#  * @author Emmanuel Castillo
#  * @email [castillo.280997@gmail.com]
#  * @create date 2024-08-16 17:12:21
#  * @modify date 2024-08-16 17:12:21
#  * @desc [description]
#  */
# The idea of this script is to find the stations information using an IRIS client.

import os
import utils as ut
from obspy.clients.fdsn import Client 


provider = "TEXNET"
output_path = "/home/emmanuel/ecastillo/dev/delaware/data/metadata/delaware_stations_160824.csv"

client =  Client(provider)

# TX 2T 4T Texnet is the owner. # 4O from operators
inventory = client.get_stations(network="TX,2T,4T,4O",station="*",level="channel")
print(inventory)
inv_info = ut.get_inventory_info(inventory)

if not os.path.isdir(os.path.dirname(output_path)):
    os.makedirs(os.path.dirname(output_path))
    
print(inv_info)
# inv_info.to_csv(output_path,index=False)

