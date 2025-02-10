import neuroLSH, data_extractor, sys

log_file_name = "a-7.txt"
sys.stdout = open("logs/" + log_file_name, "a")

infos=[
{
    'data_loader': data_extractor.get_data_notre,
    'graph_file': '-0-knn.graph',
    'dataset': 'notre',
    'k_graph': 100
},
{
    'data_loader': data_extractor.get_data_nuswide,
    'graph_file': '-0-knn.graph',
    'dataset': 'nuswide',
    'k_graph': 100
},
{
    'data_loader': data_extractor.get_data_nytimes,
    'graph_file': '-0-knn.graph',
    'dataset': 'nytimes',
    'k_graph': 100
}
]
for info in infos:
    neuroLSH.run(info)