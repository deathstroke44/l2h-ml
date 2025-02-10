import neuroLSH, data_extractor, sys

log_file_name = "a-1.txt"
sys.stdout = open("logs/" + log_file_name, "a")
sys.stderr = open("logs/" + 'err-'+ log_file_name, "a")

infos=[
{
    'data_loader': data_extractor.get_data_sift1M,
    'graph_file': '-0-knn.graph',
    'dataset': 'sift1m',
    'k_graph': 100
}
]
for info in infos:
    neuroLSH.run(info)