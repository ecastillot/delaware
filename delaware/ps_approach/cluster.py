# /**
#  * @author Emmanuel Castillo
#  * @email [castillo.280997@gmail.com]
#  * @create date 2024-08-24 21:47:52
#  * @modify date 2024-08-24 21:47:52
#  * @desc [description]
#  */


def get_clusters(picks,radius):
    
    picks["s-p"] = picks["time_s"] - picks["time_p"]
    
    