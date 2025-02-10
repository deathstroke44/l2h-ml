import neuroLSH, data_extractor, sys

log_file_name = "a-2.txt"
sys.stdout = open("logs/" + log_file_name, "a")

infos=[
{
    'data_loader': data_extractor.get_data_enron,
    'graph_file': '-0-knn.graph',
    'dataset': 'enron',
    'k_graph': 100
},
{
    'data_loader': data_extractor.get_data_audio,
    'graph_file': '-0-knn.graph',
    'dataset': 'audio',
    'k_graph': 100
},
{
    'data_loader': data_extractor.get_data_cifar,
    'graph_file': '-0-knn.graph',
    'dataset': 'cifar',
    'k_graph': 100
},
{
    'data_loader': data_extractor.get_data_ethz,
    'graph_file': '-0-knn.graph',
    'dataset': 'ethz',
    'k_graph': 100
}
]
for info in infos:
    neuroLSH.run(info)