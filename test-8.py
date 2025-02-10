import neuroLSH, data_extractor, sys

log_file_name = "a-8.txt"
sys.stdout = open("logs/" + log_file_name, "a")

infos=[
{
    'data_loader': data_extractor.get_data_movielens,
    'graph_file': '-0-knn.graph',
    'dataset': 'movielens',
    'k_graph': 100
},
{
    'data_loader': data_extractor.get_data_netflix,
    'graph_file': '-0-knn.graph',
    'dataset': 'netflix',
    'k_graph': 100
},
{
    'data_loader': data_extractor.get_data_sun,
    'graph_file': '-0-knn.graph',
    'dataset': 'sun',
    'k_graph': 100
},
{
    'data_loader': data_extractor.get_data_Yelp,
    'graph_file': '-0-knn.graph',
    'dataset': 'Yelp',
    'k_graph': 100
},
{
    'data_loader': data_extractor.get_data_yahoomusic,
    'graph_file': '-0-knn.graph',
    'dataset': 'yahoomusic',
    'k_graph': 100
}
]
for info in infos:
    neuroLSH.run(info)