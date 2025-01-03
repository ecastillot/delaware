import pandas as pd
import glob
import os
import lasio
import numpy as np
import matplotlib.pyplot as plt
from delaware.vel.vel import VelModel
from matplotlib import cm

import time
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def get_web_preferences(download_folder, hide=False):
    """
    Web preferences for Firefox Web Driver.

    Parameters:
    -----------
        download_folder: str
            Path to the folder where downloaded files will be saved.
        hide: bool
            If true, the web driver will run in headless mode.

    Returns:
    --------
        options: Options
            Configured options for the Firefox WebDriver.
    """
    options = Options()

    # Set Firefox preferences
    options.set_preference('browser.download.folderList', 2)  # Use a custom location for downloads
    options.set_preference('browser.download.manager.showWhenStarting', False)
    options.set_preference('browser.download.dir', download_folder)
    options.set_preference('browser.helperApps.neverAsk.saveToDisk', 
                           "application/force-download,application/vnd.google-earth.kml+xml")

    # Run in headless mode if specified
    if hide:
        options.add_argument('--headless')

    # Automatically set the binary location
    firefox_binary = find_firefox_binary()
    if firefox_binary:
        options.binary_location = firefox_binary
    else:
        raise FileNotFoundError("Firefox binary could not be found!")

    print(options.binary_location)
    return options

def mv_downloaded_files(download_folder):
    name = os.path.basename(download_folder)
    filepaths = []
    for downloaded_file in glob.glob(os.path.join(download_folder,"*")):
        basename = os.path.basename(downloaded_file)
        dirname = os.path.dirname(downloaded_file)
        filepath = os.path.join(dirname,".".join((name,basename.split(".")[-1])))

        msg = f"mv {downloaded_file} {filepath}"
        os.system(msg)

        filepaths.append(filepath)
    return filepaths

def find_firefox_binary():
    """
    Automatically find the Firefox binary path.

    Returns:
    --------
        str: Path to the Firefox binary, or None if not found.
    """
    # Use 'which' on Linux/macOS and 'where' on Windows
    command = "which firefox" if os.name != "nt" else "where firefox"
    path = os.popen(command).read().strip()
    return path if path else None

def find_geckodriver():
    """
    Automatically find the Geckodriver executable path.

    Returns:
    --------
        str: Path to the Geckodriver executable, or None if not found.
    """
    # Use 'which' on Linux/macOS and 'where' on Windows
    command = "which geckodriver" if os.name != "nt" else "where geckodriver"
    path = os.popen(command).read().strip()
    return path if path else None

class Client():
    def __init__(self,user,pss,
                 link=None,hide_driver=False):
        self.user = user
        self.pss = pss
        self.hide_driver = hide_driver
        
        if link is None:
            link = rf'https://login.auth.enverus.com/'
            
        self.link = link
    
    def query(self,project="DW",download_folder=None):
        # options = get_web_preferences(download_folder=download_folder,hide=self.hide_driver)
        # # driver = webdriver.Firefox(options=options)
        
        # # Automatically find the Geckodriver path
        # gecko_path = find_geckodriver()
        # if not gecko_path:
        #     raise FileNotFoundError("Geckodriver could not be found!")
        #     # exit()    
        # # service = Service(executable_path=geckodriver_path)
        # service = Service(executable_path=gecko_path)
        # # print(service)
        # driver = webdriver.Firefox(service=service, options=options)
        # print(driver)
        # driver.set_window_size(1800, 1200) #For a correct size for the screenshot
        # driver.get(self.link)
        
        service = webdriver.ChromeService()
        driver = webdriver.Chrome(service=service)
        driver.set_window_size(1800, 1200) #For a correct size for the screenshot
        driver.get(self.link)
        
        def send_key_to_container_main(driver,xpath,value):
            if value != None:
                e = driver.find_element(By.XPATH,xpath)
                print(e)
                # driver.find_element(By.XPATH,xpath).clear()
                # driver.find_element(By.XPATH,xpath).send_keys(str(value))
        
        # c = '//*[@id="auth0"]/div/div/form/div/div/div[3]/span/div/div/div/div/div/div/div/div/div/div[1]/div'
        c = '//*[@id="auth0"]/div/div/form/div/div/div[3]/span/div/div/div/div/div/div/div/div/div/div[1]/div/input'
        input_field = WebDriverWait(driver, 10).until(
                    EC.visibility_of_element_located((By.XPATH, c))
                )
        input_field.send_keys(self.user)
        
        c = '//*[@id="auth0"]/div/div/form/div/div/div[3]/span/div/div/div/div/div/div/div/div/div/div[2]/div[1]/div/input'
        input_field = WebDriverWait(driver, 10).until(
                    EC.visibility_of_element_located((By.XPATH, c))
                )
        input_field.send_keys(self.pss)
        
        c = '//*[@id="auth0"]/div/div/form/div/div/button'
        driver.find_element(By.XPATH,c).click()
        
        c = '//*[@id="app-tray-section"]/di-carousel/div/div[1]'
        element = WebDriverWait(driver, 10).until(
                        EC.element_to_be_clickable((By.XPATH, c))
                    )
        element.click()
        
        # Wait for the new tab to open
        WebDriverWait(driver, 10).until(lambda d: len(d.window_handles) > 1)

        # Switch to the new tab
        new_tab = driver.window_handles[1]  # The second tab (index starts at 0)
        driver.switch_to.window(new_tab)
        
        # for handle in new_tab:
        #     # driver.switch_to.window(handle)
        #     # Check if the page contains the target element with "DW"
            
        #     try:
        #         dw_label = WebDriverWait(driver, 10).until(
        #                         EC.element_to_be_clickable((By.XPATH, f'//label[@title="{project}"]'))
        #                     )
        #         dw_label.click()
        #         break
        #     except Exception as e:
        #         print(f"Element with {project} not found in this tab:", driver.title)
        
        # # Wait for the table container to be visible
        # # table_container = WebDriverWait(driver, 10).until(
        # #     # EC.visibility_of_element_located((By.XPATH, './/div[contains(@class, "style_dataViewerTable-3lfj_B1v")]'))
        # #     EC.visibility_of_element_located((By.XPATH, './/div[contains(@class, "rsTableVirtualCell row0 column_WellApi")]'))
        # # )
        # # for row in table_container.find_elements(By.XPATH,'.//div[contains(@class, "rsTableVirtualCell row0 column_WellApi")]'):
        
        # elements = []
        # time.sleep(15)
        # a_elements = driver.find_elements(By.XPATH, "//div[contains(@class, 'column_WellApi')]//span//a")
        # len_elements = len(a_elements)
        
        # print(len_elements)
        # for i in a_elements:
        #     print(i.text)
        # # for row in driver.find_elements(By.XPATH, "//div//span//a"):
        # for index, row in enumerate(a_elements):
        #     time.sleep(10)
            
        #     # Re-locate the elements to avoid stale references
        #     # print(len_elements)
        #     # print(f"//div[contains(@class, 'row{len_elements} column_WellApi')]//span//a")
        #     # WebDriverWait(driver, 10).until(
        #     #                     EC.element_to_be_clickable((By.XPATH, f"//div[contains(@class, 'row{len_elements} column_WellApi')]//span//a"))
        #     #                 )
            
        #     # a_elements = driver.find_elements(By.XPATH, "//div[contains(@class, 'column_WellApi')]//span//a")
        #     # print(index,len(a_elements))
        #     row = a_elements[index]
            
        #     # Interact with the element
        #     if row.text in elements:
        #         continue
        #     else:
        #         elements.append(row.text)
            
        #     print(row.text)
            # row.click()
            
            
            
            # c = "/html/body/div[4]/div/div[2]/div/div[2]/div/div[1]/div[3]/div[2]/div/ul/li[1]/a"
            # log = WebDriverWait(driver, 10).until(
            #                     EC.element_to_be_clickable((By.XPATH, c))
            #                 )
            # log.click()
            # # print(log)
            
            # # Wait until the button is clickable
            # close_button = WebDriverWait(driver, 10).until(
            #     EC.element_to_be_clickable((By.XPATH, "//button[@aria-label='Close' and contains(@class, 'ant-modal-close')]"))
            # )

            # # Click the button
            # close_button.click()
            # time.sleep(10)
            
            
            # <button type="button" aria-label="Close" class="ant-modal-close"><span class="ant-modal-close-x"><span role="img" aria-label="close" class="anticon anticon-close ant-modal-close-icon"><svg viewBox="64 64 896 896" focusable="false" data-icon="close" width="1em" height="1em" fill="currentColor" aria-hidden="true"><path d="M563.8 512l262.5-312.9c4.4-5.2.7-13.1-6.1-13.1h-79.8c-4.7 0-9.2 2.1-12.3 5.7L511.6 449.8 295.1 191.7c-3-3.6-7.5-5.7-12.3-5.7H203c-6.8 0-10.5 7.9-6.1 13.1L459.4 512 196.9 824.9A7.95 7.95 0 00203 838h79.8c4.7 0 9.2-2.1 12.3-5.7l216.5-258.1 216.5 258.1c3 3.6 7.5 5.7 12.3 5.7h79.8c6.8 0 10.5-7.9 6.1-13.1L563.8 512z"></path></svg></span></span></button>
            # logs = WebDriverWait(n_driver, 10).until(
            #     EC.visibility_of_element_located((By.XPATH, "//div//span//a"))
            # )
            
            # # Get all window handles
            # window_handles = driver.window_handles
            # # Switch to the new window
            # for window in window_handles:
            #     if window != original_window:
            #         driver.switch_to.window(window)
            #         break
            
            # logs.click()
            # print(logs.text)
            
            
            
        # <li class="metrics-well-card-geology-logs "><a>Logs</a></li>
        # print(table_container.find_element())
        # <div class="rsTableVirtualCell row0 column_WellApi" role="cell" style="position: absolute; left: 32px; top: 0px; height: 31px; width: 170px;"><span><a>42-389-10508</a></span></div>
        # for row in rows:
        #     print(row)
            # columns = row.find_elements(By.XPATH, './/div[contains(@class, "column_WellApi")]')
            # print(columns)
            # row_data = [col.text for col in columns]
            # print(row_data)
        # //*[@id="rsTableContainer-4c75660b-3d1d-4007-972b-45e35a7c5000"]/div[5]/div
        # <div class="rsTableVirtualCell row0 column_WellApi" role="cell" style="position: absolute; left: 32px; top: 0px; height: 31px; width: 170px;"><span><a>42-389-10508</a></span></div>
        # <div class="rsTableVirtualCell row0 column_WellApi12" role="cell" style="position: absolute; left: 170px; top: 0px; height: 31px; width: 170px;"><span>42-389-10508-00</span></div>
        # c = '//*[@id="app-tray-section"]/di-carousel/div/div[1]'
        # element = WebDriverWait(driver, 10).until(
        #                 EC.element_to_be_clickable((By.XPATH, c))
        #             )
        # element.click()        
        # //*[@id="widget-52e3a118-41d4-471e-b6c8-67ddba2f6db1"]
        # table_page= driver.find_element('/html/body/table[3]')
        # //*[@id="rsTableContainer-4c75660b-3d1d-4007-972b-45e35a7c5000"]/div[5]/div/div[1]
        
        
        # # c = '//*[@id="rc-tabs-0-panel-welcome"]/div[2]/div[2]/div[2]/div[2]'
        # c = '//*[@id="rc-tabs-0-panel-welcome"]/div[2]/div[2]/div[2]/div[2]/div[1]/label'
        # element = WebDriverWait(driver, 10).until(
        #                 EC.element_to_be_clickable((By.XPATH, c))
        #             )
        # element.click()
        
        # # Iterate through all open tabs/windows
        # for handle in driver.window_handles:
        #     driver.switch_to.window(handle)
        #     # Check if the title contains "DW"
        #     if "DW" in driver.title:
        #         print("Switched to tab with title containing 'DW':", driver.title)
        #         break
        # else:
        #     print("No tab with title containing 'DW' found.")
        
        # # Wait for the element in the "DW" tab to be clickable
        # target_element = WebDriverWait(driver, 10).until(
        #     EC.element_to_be_clickable((By.XPATH, 'YOUR_TARGET_ELEMENT_XPATH'))
        # )
        # c = '//*[@id="rsTableContainer-ab61d05a-36e6-44ac-bffe-385c230200e8"]/div[5]/div/div[1]/span/a'
        # element = WebDriverWait(driver, 10).until(
        #                 EC.element_to_be_clickable((By.XPATH, c))
        #             )
        # element.click()
        
        
        # c = driver.find_element(By.XPATH, c).click()
        # print(c)
        
        # //*[@id="auth0"]/div/div/form/div/div/div[3]/span/div/div/div/div/div/div/div/div/div/div[1]/div/input
        
        # content = driver.find_element(By.CLASS_NAME,'auth0-lock-input-block.auth0-lock-input-username')
        # content = driver.find_element(By.XPATH, '//*[@id="auth0"]/div/div/form/div/div/div[3]/span/div/div/div/div/div/div/div/div/div')
        # content = driver.find_element(By.XPATH, '//*[@id="auth0"]/')
        # xpath_expression = '//*[@id="auth0"]/div/div/form/div/div/div[3]/span/div/div/div/div/div/div/div/div/div/div[1]'
        # xpath_expression = '//input[@placeholder="username/email"]'
        # xpath_expression = '//input[contains(@placeholder, "username/email")]'
        # content = driver.find_element(By.XPATH, xpath_expression)
        # xpath_expression = "//*[@id="auth0"]/div/div/form/div/div/div[3]/span/div/div/div/div/div/div/div/div/div/div[1]//div[contains(@class, 'auth0-lock-icon auth0-lock-icon-box')]"
        # print(content)
        
                # <input type="text" name="username" class="auth0-lock-input" placeholder="username/email" autocomplete="off" autocapitalize="off" spellcheck="off" autocorrect="off" aria-label="User name" aria-invalid="false" aria-describedby="auth0-lock-error-msg-username" value="emmanuel.castillotaborda@utdallas.edu">
        # <div class="auth0-lock-input-block auth0-lock-input-username"><div class="auth0-lock-input-wrap auth0-lock-input-wrap-with-icon"><span aria-hidden="true"><svg aria-hidden="true" focusable="false" width="13px" height="14px" viewBox="0 0 15 16" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" class="auth0-lock-icon auth0-lock-icon-box"><g stroke="none" stroke-width="1" fill="none" fill-rule="evenodd"><g transform="translate(-11.000000, -1471.000000)" fill="#888888"><path d="M25.552,1486.998 L11.449,1486.998 C10.667,1485.799 10.984,1483.399 11.766,1482.6 C12.139,1482.219 14.931,1481.5 16.267,1481.172 C14.856,1480.076 13.995,1478.042 13.995,1476.103 C13.995,1473.284 14.813,1470.999 18.498,1470.999 C22.182,1470.999 23,1473.284 23,1476.103 C23,1478.037 22.145,1480.065 20.74,1481.163 C22.046,1481.489 24.88,1482.228 25.241,1482.601 C26.019,1483.399 26.328,1485.799 25.552,1486.998 L25.552,1486.998 Z M24.6,1483.443 C24.087,1483.169 21.881,1482.548 20,1482.097 L20,1480.513 C21.254,1479.659 21.997,1477.806 21.997,1476.12 C21.997,1473.841 21.414,1471.993 18.499,1471.993 C15.583,1471.993 15,1473.841 15,1476.12 C15,1477.807 15.744,1479.662 17,1480.515 L17,1482.112 C15.109,1482.556 12.914,1483.166 12.409,1483.442 C12.082,1483.854 11.797,1485.173 12,1486 L25,1486 C25.201,1485.174 24.922,1483.858 24.6,1483.443 L24.6,1483.443 Z"></path></g></g></svg></span><input type="text" name="username" class="auth0-lock-input" placeholder="username/email" autocomplete="off" autocapitalize="off" spellcheck="off" autocorrect="off" aria-label="User name" aria-invalid="false" aria-describedby="auth0-lock-error-msg-username" value="emmanuel.castillotaborda@utdallas.edu"></div></div>
        # user_xpath =  f'//*[@id="auth0"]/div/div/form/div/div/div[3]/span/div/div/div/div/div/div/div/div/div/div[1]/div/input'   
        # pss_xpath = f'//*[@id="auth0"]/div/div/form/div/div/div[3]/span/div/div/div/div/div/div/div/div/div/div[2]/div[1]/div/input' 
        # print(user_xpath)
        
        # send_key_to_container_main(driver,user_xpath,self.user)
        # send_key_to_container_main(driver,pss_xpath,self.pss)
        
        # Keep the browser open until manually closed by you
        input("Press Enter to close the browser...")  # This will keep the browser open until you press Enter

        # Close the driver
        driver.quit()
        
        
        
        
        

def get_well_names(folder):
    """
    Extract unique well names from LAS files in a given folder.

    Parameters:
        folder (str): Path to the directory containing LAS files.

    Returns:
        list: A list of unique well names extracted from the file names.
    """
    # Get all LAS file paths in the specified folder
    paths = glob.glob(os.path.join(folder, "*.LAS"))
    
    # Initialize an empty list to store well names
    well_names = []
    
    # Iterate through each LAS file path
    for path in paths:
        # Get the file name without extension
        basename = os.path.splitext(os.path.basename(path))[0]
        
        # Extract the well name (assumes it's the first part of the file name before an underscore)
        well_name = basename.split("_")[0]
        
        # Append the well name to the list
        well_names.append(well_name)
    
    # Remove duplicates by converting the list to a set and back to a list
    well_names = list(set(well_names))
    
    return well_names

def get_uwi12(las_path):
    basename = os.path.basename(las_path)
    well_name = basename.split("_")[0]
    uwi12 = well_name[:-3]
    return uwi12
    

def get_sonic_data(las_path,formation):
    las = lasio.read(las_path)
    df = las.df()
    df = df.reset_index()
    # print(df.columns)
    
    uwi12 = get_uwi12(las_path)
    
    formation = formation[formation["API_UWI_12"] == uwi12]
    formation = formation.sort_values("MD_Top")
    
    md_values = None
    for x in ["DEPTH","DEPT","DEP"]:
        # print(df.columns.to_list())
        if x in df.columns.to_list():
            md_values = df[x]
            # print(x,df.columns.to_list())
            continue
            
    # print(md_values)
    if md_values is None:
        raise Exception("No Depth")
    
    dt_columns = []    
    if "DT" in df.columns.to_list():
        dt_columns.append("DT")
    if "DT_S" in df.columns.to_list():
        dt_columns.append("DT_S")
    if not dt_columns:
        raise Exception(f"No DT or DTS in {uwi12}")
    
    md_real_values = [md_values.min()] + formation["MD_Top"].to_list() + [md_values.max()]
    tvd_real_values = [md_values.min()] + formation["TVD_Top"].to_list() + [md_values.max()]
    tvd_values = np.interp(md_values, 
                           md_real_values, 
                           tvd_real_values)
    
    result = pd.DataFrame({"MD": md_values, "TVD": tvd_values})
    result = pd.concat([result,df[dt_columns]],axis=1)
    result.dropna(subset=dt_columns,inplace=True)

    # Convert MD and TVD from ft to km
    result["MD[km]"] = result["MD"] * 0.0003048
    result["TVD[km]"] = result["TVD"] * 0.0003048

    # result["Vp[km/s]"] = np.nan
    # result["vs[km/s]"] = np.nan
    # Convert DT (µs/ft) to Velocity (km/s)
    for dt in dt_columns:
        if dt == "DT":
            result["Vp[km/s]"] = 1 / (result["DT"] * 1e-6 * 3280.84)
        elif dt == "DT_S":
            result["Vs[km/s]"] = 1 / (result["DT"] * 1e-6 * 3280.84)
            
    result.dropna(subset=dt_columns,inplace=True)
    
    if result.empty:
        raise Exception("No Vp or Vs data")
    
    # print(result)
    # # Plot a specific log
    # plt.figure(figsize=(6, 10))
    # plt.plot(result['V[km/s]'], result["TVD[km]"], label="Vel", color='green')
    # plt.gca().invert_yaxis()
    # plt.xlabel("Vel [km/s]")
    # plt.ylabel("Depth (m)")
    # plt.title("Vel Log")
    # plt.legend()
    # plt.grid()
    # plt.show()
    # print(result)
    return result
    
def get_sonics_data(las_folder,formation,well):
    
    # Get all LAS file paths in the specified folder
    paths = glob.glob(os.path.join(las_folder, "*.LAS"))
    # print(get_well_names(las_folder))
    
    wells =  {}
    # p_well_names = []
    for las_path in paths:
        
        # Get the file name without extension
        basename = os.path.splitext(os.path.basename(las_path))[0]
        
        # Extract the well name (assumes it's the first part of the file name before an underscore)
        well_name = basename.split("_")[0]
        
               
        uwi12 = get_uwi12(las_path)
        single_well = well[well["API_UWI_12"] == uwi12]
        # print(single_well)
        # exit()
        single_well = single_well.iloc[0]
        latitude = single_well["Latitude"]
        longitude = single_well["Longitude"]
        elevation = single_well["ENVElevationGL_FT"]*0.0003048
        # print(lat,lon)
        
        # exit()
        
        
        # print(las_path)
        try:
            df = get_sonic_data(las_path,formation)
            # print(well_name,"OK",latitude,
            #       longitude,elevation)
        except Exception as e:
            continue
    
        df.insert(0,"well_name",well_name)
        df.insert(1,"latitude",latitude)
        df.insert(2,"longitude",longitude)
        df.insert(3,"elevation[km]",elevation)
    
        if well_name not in list(wells.keys()):
            print(well_name,"OK\t",latitude,
                  longitude,elevation)
        #     prev_max_depth = wells[well_name]["TVD[km]"].max()
        #     curr_max_depth = df["TVD[km]"].max()
        #     print(prev_max_depth,curr_max_depth)
        #     continue
    
        # p_well_names.append(well_name)
        wells[well_name] = df
        
    wells = pd.concat(list(wells.values()))
    # print(wells.describe())
    
    return wells

def get_dw_models():
    dw_path = "/home/emmanuel/ecastillo/dev/delaware/10102024/data_git/vel"
    vel_path = os.path.join(dw_path,"DW_model.csv")
    data = pd.read_csv(vel_path)
    sheng_velmodel = VelModel(data=data,dtm=-0.7,name="Sheng")
    
    vel_path = os.path.join(dw_path,"DB_model.csv")
    data = pd.read_csv(vel_path)
    db1d_velmodel = VelModel(data=data,dtm=-0.7,name="db1d")
    return db1d_velmodel, sheng_velmodel

def plot_all_velocity_logs(data, 
                       formations=None,
                       wells=None, 
                       depth="Depth[km]", 
                       vel="Vp[km/s]", 
                       smooth="moving_average",
                       xlims= None,
                       ylims=None):
    grouped_data = data.groupby("well_name").__iter__()
    
    if wells is None:
        wells = list(set(data["well_name"].tolist()))
    
    if smooth is not None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 10))  # Adjusted figure size for better clarity
    
    db1d_velmodel, sheng_velmodel = get_dw_models()
    
    for well_name, single_data in grouped_data:
        single_data = single_data.sort_values("TVD[km]")
        
        if well_name not in wells:
            continue
        
        api12 = well_name[0:-3]
        # print(api12)
        single_formation = formations[formations["API_UWI_12"]==api12]
        single_formation = single_formation.drop_duplicates("FormationName",
                                                            ignore_index=True)
        # print(single_formation)
        
        if depth == "Depth[km]":
            single_data[depth] = single_data["TVD[km]"] - single_data["elevation[km]"]
        
        if smooth == "moving_average":
            single_data['Vp_smoothed'] = single_data['Vp[km/s]'].rolling(window=1000, 
                                                                         center=True).mean()
        
        for i in range(len(axes)):
            if i == 0:
                axes[i].step(single_data[vel], 
                        single_data[depth], 
                        label=well_name)
            elif i == 1:
                axes[i].step(single_data['Vp_smoothed'], 
                        single_data[depth], 
                        label=well_name)

    if not single_formation.empty:
        # print(single_formation)
        print(single_data["elevation[km]"].iloc[0])
        for row,f in single_formation.iterrows():
            depth_f = (f["TVD_Top"] * 0.0003048) - single_data["elevation[km]"].iloc[0]
            label_f = f["FormationName"]
            axes[i].axhline(y=depth_f, color='k', linestyle='--', linewidth=0.5)

            axes[i].text(1.02, depth_f, label_f, color='black', fontsize=8,
                    ha='left', va='top', 
                    transform=axes[i].get_yaxis_transform())  # Label
    # x_min, x_max = data['Vp_smoothed'].min(), data['Vp_smoothed'].max()
    # y_min, y_max = data[depth].min(), data[depth].max()
    
    # Set the same x and y limits for both subplots
    for ax in axes:
        ax.step(db1d_velmodel.data["VP (km/s)"], 
                db1d_velmodel.data["Depth (km)"], 'blue', 
                linewidth=2.5, linestyle='-', 
                label='DB1D')
        ax.step(sheng_velmodel.data["VP (km/s)"], 
                sheng_velmodel.data["Depth (km)"], 
                'orange', linewidth=2.5, linestyle='-', 
                label='Sheng (2022)')
        if xlims is not None:
            ax.set_xlim(xlims[0],xlims[1])
        if ylims is not None:
            ax.set_ylim(ylims[0],ylims[1])
        ax.invert_yaxis()  # Invert y-axis
        
        ax.set_xlabel("Vp (km/s)")
        ax.set_ylabel(depth)
        ax.legend()
        ax.grid()

    plt.tight_layout()  # Adjust layout for clarity
    plt.show()
    
    return

# Assign colors to formations globally
def assign_global_colors(formations):
    """Assign consistent colors to formations globally."""
    unique_formations = sorted(formations.unique())  # Sort to ensure consistency
    color_map = plt.cm.get_cmap("tab20", len(unique_formations))  # Use tab20 for distinct colors
    return {formation: color_map(i) for i, formation in enumerate(unique_formations)}
        
def plot_velocity_logs(data, 
                       formations=None,
                       wells=None, 
                       depth="Depth[km]", 
                       vel="Vp[km/s]", 
                       smooth_interval=None,
                       xlims= None,
                       ylims=None,
                       grid = True,
                       minor_ticks = True):
    
    sortby = wells
    if wells is None:
        wells = list(set(data["well_name"].tolist()))
    
    data = data[data["well_name"].isin(wells)]
    
    if sortby is None:
        data = data.sort_values(["longitude"])
        order = data["well_name"].unique()
    else:
        data["well_name"] = pd.Categorical(data["well_name"], categories=wells, ordered=True)
        data = data.sort_values("well_name")
        order = data["well_name"].unique()
        
    grouped_data = data.groupby("well_name")
    
    fig, axes = plt.subplots(1, len(wells), 
                             sharey=True,
                             figsize=(12, 14)# Adjusted figure size for better clarity
                                )  
    
    db1d_velmodel, sheng_velmodel = get_dw_models()
    
    
    vel_model_style = {'DB1D':{"linewidth":1.5,
                               "color":'black',
                               "linestyle":'-',
                               "vel_model":db1d_velmodel
                               },
                       'Sheng (2022)':{"linewidth":1.5,
                               "color":'black',
                               "linestyle":'--',
                               "vel_model":sheng_velmodel
                               }
                       }
    vel_legend_handles = []
    for key in vel_model_style.keys():
        legend_line = plt.Line2D([0], [0], 
                                   color=vel_model_style[key]["color"], 
                                   lw=vel_model_style[key]["linewidth"], 
                                   linestyle = vel_model_style[key]["linestyle"],
                                   label=key)
        vel_legend_handles.append(legend_line)
    
    for i,well_name in enumerate(order):
        single_data = grouped_data.get_group(well_name)
        single_data = single_data.sort_values("TVD[km]")
        
        
        # print(single_formation)
        
        #elevation
        elevation_km = - round(single_data["elevation[km]"].iloc[0],1)
        axes[i].axhline(y= elevation_km, 
                        color='black', linestyle='-', linewidth=1)
        
        # vel models
        for key in vel_model_style.keys():
            vel_model = vel_model_style[key]["vel_model"]
            
            data2plot = vel_model.data.copy()
            data2plot["Depth (km)"] = data2plot["Depth (km)"]+elevation_km
            data2plot = data2plot[data2plot["Depth (km)"]>=elevation_km]
            
            axes[i].step(data2plot["VP (km/s)"], 
                    data2plot["Depth (km)"], 
                    color=vel_model_style[key]["color"], 
                    linewidth=vel_model_style[key]["linewidth"], 
                    linestyle=vel_model_style[key]["linestyle"], 
                    label=key)
            
        
        if depth == "Depth[km]":
            single_data[depth] = single_data["TVD[km]"] + elevation_km
        
        if smooth_interval is not None:
            single_data['Depth_interval'] = (single_data['Depth[km]'] // smooth_interval) * smooth_interval
            result = single_data.groupby('Depth_interval')['Vp[km/s]'].median().reset_index()
            result.columns = ['Depth[km]', 'Vp[km/s]']
            
            if result.empty:
                print(f"{well_name} empty using smooth_interval = {smooth_interval} km")
                continue
            else:
                single_data = result
                
            
            print(single_data)
            # single_data['Vp_smoothed'] = single_data['Vp[km/s]'].rolling(window=100, 
            #                                                              center=True).mean()
        
        
        axes[i].step(single_data[vel], 
                        single_data[depth], 
                        "red",
                        linewidth=2.5,
                        label=well_name)

        
        
        ##formations
        api12 = well_name[0:-3]
        # print(api12)
        # print(formations)
        single_formation = formations[formations["API_UWI_12"]==api12]
        single_formation = single_formation.sort_values("TVD_Top",
                                                            ignore_index=True)
        single_formation = single_formation.drop_duplicates("FormationName",
                                                            ignore_index=True)
            
        # Inside your main function
        # Assign global colors based on all formations
        all_formations = formations["FormationName"].drop_duplicates()
        global_formation_colors = assign_global_colors(all_formations)
        
        # Create a list for global legend handles
        global_legend_handles = []
        

        # Modify the plotting code
        if not single_formation.empty:
            
            for _, f in single_formation.iterrows():
                depth_f = (f["TVD_Top"] * 0.0003048) + elevation_km
                formation_name = f["FormationName"]
                color = global_formation_colors.get(formation_name, "gray")
                
                # Plot hlines with consistent global colors
                axes[i].axhline(y=depth_f, color=color, linestyle="--", linewidth=1.5)
                
                # Add to global legend handles if not already added
                if not any(handle.get_label() == formation_name for handle in global_legend_handles):
                    global_legend_handles.append(
                        plt.Line2D([0], [0], color=color, lw=2,linestyle="--", label=formation_name)
                    )
        # ylim = axes[i].get_ylim()  # Get y-axis limits    
        
            
        if xlims is not None:
            axes[i].set_xlim(xlims[0],xlims[1])
        if ylims is not None:
            axes[i].set_ylim(ylims[0],ylims[1])
        
        axes[i].set_title(well_name)
        
        ## grid
        if grid:
            if minor_ticks:
                xlim = axes[i].get_xlim()  # Get x-axis limits
                ylim = axes[i].get_ylim()  # Get y-axis limits
                axes[i].set_xticks(np.arange(xlim[0], xlim[1], 0.5), minor=True)
                axes[i].set_yticks(np.arange(ylim[0], ylim[1], 0.5), minor=True)
                axes[i].grid(color='gray', linewidth=0.5,linestyle=":",which='minor')
                
            axes[i].grid(color='black', linewidth=0.5,linestyle=":")
        
        
        axes[i].invert_yaxis()  # Invert y-axis
    
    bottom_adjust = 0.2
    
    # # Add legend for formations
    # axes[i].legend(handles=legend_handles, loc="lower left", fontsize=8)
    # Add a single, consolidated legend after plotting all axes
    fig.legend(
        handles=global_legend_handles,
        loc="upper left",
        fontsize=10,
        ncol=4,  # Adjust the number of columns as needed
        bbox_to_anchor=(0.01, bottom_adjust - 0.03)  # Position the legend below the last axis
    )
    # Add another legend for velocity models
    fig.legend(
        handles=vel_legend_handles,
        loc="upper left",
        fontsize=10,
        ncol=1,  # Adjust the number of columns as needed
        bbox_to_anchor=(0.75, bottom_adjust - 0.03)  # Position this legend slightly to the right
    )
    
    plt.tight_layout()  # Adjust layout for clarity
    plt.subplots_adjust(bottom=bottom_adjust)  # Add space at the bottom for the legend
    plt.show()
    
    return        
        
    
    
    
# 1 
# folder = "/home/emmanuel/ecastillo/dev/delaware/10102024/data_git/enverus/EnverusData_AOI/data"
# well_pad_names = get_well_pad_names(folder)

# 2
# well_path = "/home/emmanuel/ecastillo/dev/delaware/10102024/data_git/enverus/EnverusData_AOI/env_csv-Wells-6e346_2024-12-23.csv"
# formation_path = "/home/emmanuel/ecastillo/dev/delaware/10102024/data_git/enverus/EnverusData_AOI/env_csv-FormationTops-332ba_2024-12-23.csv"
# formation = pd.read_csv(formation_path)
# well = pd.read_csv(well_path)
# folder = "/home/emmanuel/ecastillo/dev/delaware/10102024/data_git/enverus/EnverusData_AOI/data"
# data = get_sonics_data(folder,formation,well)
# output = "/home/emmanuel/ecastillo/dev/delaware/10102024/data_git/enverus/EnverusData_AOI/wells_aoi.csv"
# data.to_csv(output,index=False)


# 3
# output = "/home/emmanuel/ecastillo/dev/delaware/10102024/data_git/enverus/EnverusData_AOI/wells_aoi.csv"
# data = pd.read_csv(output)
# formations = "/home/emmanuel/ecastillo/dev/delaware/10102024/data_git/enverus/EnverusData_AOI/env_csv-FormationTops-332ba_2024-12-23.csv"
# formations = pd.read_csv(formations)
# plot_velocity_logs(data,depth="Depth[km]",
#                    ylims=(-2,6),
#                    xlims=(1.5,6.5),
#                    smooth_interval=0.1,
#                    formations=formations)
# plot_velocity_logs(data,depth="TVD[km]",ylims=(-2,6))


## enverus sheng

# 2
# well_path = "/home/emmanuel/ecastillo/dev/delaware/10102024/data_git/enverus/EnverusData_AOISheng/env_csv-Wells-7d7bb_2024-12-31.csv"
# formation_path = "/home/emmanuel/ecastillo/dev/delaware/10102024/data_git/enverus/EnverusData_AOISheng/env_csv-FormationTops-e90fb_2024-12-31.csv"
# formation = pd.read_csv(formation_path)
# well = pd.read_csv(well_path)
# folder = "/home/emmanuel/ecastillo/dev/delaware/10102024/data_git/enverus/EnverusData_AOISheng/data"
# data = get_sonics_data(folder,formation,well)
# output = "/home/emmanuel/ecastillo/dev/delaware/10102024/data_git/enverus/EnverusData_AOISheng/wells_aoiSheng.csv"
# data.to_csv(output,index=False)

# output = "/home/emmanuel/ecastillo/dev/delaware/10102024/data_git/enverus/EnverusData_AOISheng/wells_aoiSheng.csv"
# data = pd.read_csv(output)
# formations = "/home/emmanuel/ecastillo/dev/delaware/10102024/data_git/enverus/EnverusData_AOISheng/env_csv-FormationTops-e90fb_2024-12-31.csv"
# formations = pd.read_csv(formations)
# plot_velocity_logs(data,depth="Depth[km]",
#                    ylims=(-2,6),
#                    xlims=(1.5,6.5),
#                    smooth_interval=0.1,
#                    formations=formations
#                 )


## eneverus get data
user = "emmanuel.castillotaborda@utdallas.edu"
pss = "Sismologia#1804"
client = Client(user,pss)
client.query()