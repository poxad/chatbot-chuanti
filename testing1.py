import joblib
import pprint

try:
    past_chats = joblib.load('data/graph_data')
except:
    past_chats = {}
    
pprint.pprint(past_chats)