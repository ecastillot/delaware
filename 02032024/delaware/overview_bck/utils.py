
"""
 * @author Emmanuel David Castillo Taborda
 * @email ecastillot@unal.edu.co
 * @create date 2021-03-03 23:51:21
 * @modify date 2025-01-16 16:32:00 - Emmanuel Guzman
 * @desc [description]
"""

import os
import ast
import glob
import pandas as pd
from obspy import read_inventory

def get_csv_events_picks(catalog, with_magnitude=True, 
					csv_folder_path = None, request_event_type = None):
	"""
	parameters
	----------
    catalog: 
        Obspy catalog (events + picks)
	with_magnitude: Bolean
		If True return the magnitude and magnitude information
	picker: None
		In 'eqt' or 'phasenet' doesn't matter the channel where is located the pick.
		While in others matters the channel. None is select to  have importance in the channel
	csv_folder_path: str (deault : None)
		Path to export in the csv files. None don't create any file, then only returns.
	returns
	-------
	appended_events : list
		It's a list of Pandas DataFrames where each dataframe contains all information about the picks from the respective event
	"""
	event_list = catalog.events 


	event_colname= ["n_event","n_origin","event_id","event_time","latitude","latitude_uncertainty",
					"longitude","longitude_uncertainty","depth","depth_uncertainty",
					"rms","region","method","earth_model","event_type","magnitude",
					"magnitude_type","n_P_phases","n_S_phases", "preferred_origin"]
	pick_colname = ["n_event", "n_origin", "event_id","pick_id","phasehint","arrival_time",
					"probability","snr","detection_probability",
					"network","station","location","channel","picker", "evaluation_mode",
					"preferred_origin"]
	events_info_list = []
	picks_info_list = []
	for n_ev,event in enumerate(event_list):
		# ---------------- Event information ----------------
		loc_id = os.path.basename(str(event.resource_id))
		ev_type = event.event_type
		if request_event_type is not None:
			if ev_type != request_event_type: continue
			
		ev_type = event.event_type

		if event.event_descriptions:
			region = event.event_descriptions[0].text
		else:
			region = None

		# ---------------- Origin information ----------------
		origin_list = event.origins

		for n_or, origin in enumerate(origin_list):
			## Preferred Origin
			pref_origin = str(origin.resource_id == event.preferred_origin_id) # True or False
			time = origin.time
			latitude = origin.latitude
			latitude_error = origin.latitude_errors.uncertainty
			longitude = origin.longitude
			longitude_error = origin.longitude_errors.uncertainty

			depth = origin.depth
			depth_error = origin.depth_errors
			rms = origin.quality.standard_error
			method = os.path.basename(str(origin.method_id))
			earth_model = os.path.basename(str(origin.earth_model_id))
			evaluation_mode = origin.evaluation_mode

			if depth != None:
				depth = float(depth) #in km
			if depth_error != None:
				if depth_error.uncertainty != None:
					depth_error = depth_error.uncertainty #in km
				else:
					depth_error = None
			else:
				depth_error = None
			## Preferred Magnitude
			if with_magnitude:
				pref_magnitude = event.preferred_magnitude()
				if pref_magnitude != None:
					magnitude = pref_magnitude.mag
					magnitude_type = pref_magnitude.magnitude_type
					if magnitude != None:
						magnitude = round(magnitude,2)
				else:
					magnitude = None
					magnitude_type = None

			else:
				magnitude = None
				magnitude_type = None
		
		
			## Dictionary with the picks information
			picks = {}
			for pick in event.picks:
			# for pick in origin.arrivals:
				if pick.resource_id.id not in picks.keys():
					# print(loc_id, pick.resource_id)
					
					if pick.creation_info == None:
						_author = None
					else:
						_author = pick.creation_info.author
					
					if pick.comments:
						comment = ast.literal_eval(pick.comments[0].text)
						prob = comment["probability"]
						if _author == "EQTransformer":
							snr = comment["snr"]
							ev_prob = comment["detection_probability"]
						else:
							snr = None
							ev_prob = None
					else:
						prob = None
						snr = None
						ev_prob = None

					picks[pick.resource_id.id] = {
												"id":os.path.basename(str(pick.resource_id)),
												"network_code":pick.waveform_id.network_code,
												"station_code":pick.waveform_id.station_code,
												"location_code":pick.waveform_id.location_code,
												"channel_code":pick.waveform_id.channel_code,
												"phase_hint":pick.phase_hint,
												"time":pick.time,
												"author":_author,
												"probability": prob,
												"snr":snr,
												"detection_probability":ev_prob,
												"time_errors":pick.time_errors,
												"filter_id":pick.filter_id,
												"method_id":pick.method_id,
												"polarity":pick.polarity,
												"evaluation_mode":pick.evaluation_mode,
												"evaluation_status":pick.evaluation_status } 
			
			p_count = 0
			s_count = 0
			for i,arrival in enumerate(origin.arrivals):

				pick = picks[arrival.pick_id.id]
				pick_row = [n_ev,n_or,loc_id,pick["id"],pick["phase_hint"],
							pick["time"].datetime,pick["probability"],
							pick["snr"],pick["detection_probability"],
							pick["network_code"],pick["station_code"],
							pick["location_code"],pick["channel_code"],
							pick["author"],pick["evaluation_mode"],pref_origin
							]
				
				picks_info_list.append(pick_row)

				if pick["phase_hint"].upper() == "P":
					p_count += 1
				elif pick["phase_hint"].upper() == "S":
					s_count += 1

			
			event_row = [n_ev,n_or,loc_id,time.datetime,latitude,latitude_error,\
						longitude,longitude_error,depth,depth_error,rms,region,\
						method, earth_model,ev_type,  magnitude, magnitude_type,
						p_count,s_count,pref_origin]

			events_info_list.append(event_row)

	# Create and Fill dataframes
	
	events_df = pd.DataFrame()
	
	for event_row in events_info_list:
		events_df_tmp = pd.DataFrame([event_row],columns=event_colname)
		events_df = pd.concat([events_df, events_df_tmp], ignore_index=True)
	
	picks_df = pd.DataFrame()

	for picks_row in picks_info_list:
		picks_df_tmp = pd.DataFrame([picks_row],columns=pick_colname)
		picks_df = pd.concat([picks_df, picks_df_tmp], ignore_index=True)


	events_df = events_df.sort_values(by=["event_id", "n_origin"],
										ascending=[True, True],
										ignore_index=True)
	picks_df = picks_df.sort_values(by=["arrival_time", "n_origin"],
										ascending=[True, True],
										ignore_index=True)

	if csv_folder_path != None:
		if os.path.isdir(csv_folder_path) == False: os.makedirs(csv_folder_path)

		events_csv_fpath = os.path.join(csv_folder_path, "catalog.csv")
		picks_csv_fpath = os.path.join(csv_folder_path, "picks.csv")

		events_df.to_csv(events_csv_fpath, index=False)
		picks_df.to_csv(picks_csv_fpath, index=False)
		
		print(f"Events_csv_file: {events_csv_fpath}")
		print(f"Picks_csv_file: {picks_csv_fpath}")
	return events_df,picks_df

def get_csv_stations(inv_folder_path, csv_fpath = None):
	inv_fpath_list = glob.glob(os.path.join(inv_folder_path, "*.DATALESS"))
	
	info_dict = {"network": [], "station": [], "longitude": [], "latitude": []}
	print(f"Reading inventories ...")
	for inv_fpath in inv_fpath_list:
		print(f"{inv_fpath} [read]")
		# Read inventory
		inv = read_inventory(inv_fpath)
		# Get network and station
		network = inv[0]
		station = network[0]
		# Extract net, sta, lon, lat, depth, elevation
		info_dict["network"].append(network.code)
		info_dict["station"].append(station.code)

		info_dict["longitude"].append(station.longitude)
		info_dict["latitude"].append(station.latitude)

	sta_df = pd.DataFrame.from_dict(info_dict)
	if os.path.isdir(os.path.dirname(csv_fpath)) == False: os.makedirs(os.path.dirname(csv_fpath))
	sta_df.to_csv(csv_fpath, index = False)
	print(f"{csv_fpath} [saved]")
	return sta_df