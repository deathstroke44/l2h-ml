import neuroLSH, data_extractor, sys

log_file_name = "a-9.txt"
sys.stdout = open("logs/" + log_file_name, "a")

infos=[
{
    'data_loader': data_extractor.get_data_vcseis,
    'graph_file': '-0-knn.graph',
    'dataset': 'vcseis',
    'k_graph': 100
},
{
    'data_loader': data_extractor.get_data_txed,
    'graph_file': '-0-knn.graph',
    'dataset': 'txed',
    'k_graph': 100
},
{
    'data_loader': data_extractor.get_data_geofon,
    'graph_file': '-0-knn.graph',
    'dataset': 'geofon',
    'k_graph': 100
},
{
    'data_loader': data_extractor.get_data_MNIST,
    'graph_file': '-0-knn.graph',
    'dataset': 'MNIST',
    'k_graph': 100
}
]
for info in infos:
    neuroLSH.run(info)