import neuroLSH, data_extractor
infos=[
{
    'data_loader': data_extractor.get_data_sift,
    'graph_file': '-0-knn.graph',
    'dataset': 'sift-small',
    'k_graph': 100
}
]
for info in infos:
    neuroLSH.run(info)