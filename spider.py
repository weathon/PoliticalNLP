from bs4 import BeautifulSoup
import pandas as pd
import requests

url_set = {'Democrat': 'https://twitter.com/TheDemocrats/lists/house-democrats/members', 'Republican': 'https://twitter.com/HouseGOP/lists/house-republicans/members'}
output_data = []; # Make a list of lists for the final dataframe
tw_party= []
tw_names = []
tw_handles = []
tw_avatar = []
for party, tw_url in url_set.items():    
    r  = requests.get(tw_url)
    data = r.text
    soup = BeautifulSoup(data,'html.parser')
    profile = soup.findAll("a", { "class" : "js-user-profile-link" })
    for pf in profile:
        fullnames = pf.find_all("strong", { "class" : "fullname" })
        for fn in fullnames:
            tw_names.append(fn.getText())
        avatar = pf.find_all("img", { "class" : "avatar" })
        for av in avatar:
            tw_avatar.append(av['src'])
        handles = pf.find_all("b")
        for hdl in handles:
            tw_handles.append(hdl.getText())
        tw_party.append(party)
output_data = pd.DataFrame([tw_party,tw_names,tw_handles,tw_avatar]).transpose();
output_data.columns = ['Party','Name','Handle','AvatarURL']
output_data.to_csv('TwitterHandles.csv',index=False)