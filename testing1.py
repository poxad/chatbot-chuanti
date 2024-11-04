import joblib
import pprint

try:
    past_chats = joblib.load('data/past_chats_list')
except:
    past_chats = {}
    
pprint.pprint(past_chats)