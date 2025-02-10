import neuroLSH, data_extractor, sys

log_file_name = "a-6.txt"
sys.stdout = open("logs/" + log_file_name, "a")
sys.stderr = open("logs/" + 'err-'+ log_file_name, "a")

infos=[
{
    'data_loader': data_extractor.get_data_imageNet,
    'graph_file': '-0-knn.graph',
    'dataset': 'imagenet',
    'k_graph': 100
}
]
for info in infos:
    neuroLSH.run(info)