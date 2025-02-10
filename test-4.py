import neuroLSH, data_extractor, sys

log_file_name = "a-4.txt"
sys.stdout = open("logs/" + log_file_name, "a")

infos=[
{
    'data_loader': data_extractor.get_data_trevi,
    'graph_file': '-0-knn.graph',
    'dataset': 'trevi',
    'k_graph': 100
}
]
for info in infos:
    neuroLSH.run(info)