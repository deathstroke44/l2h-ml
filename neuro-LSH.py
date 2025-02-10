
'''
Demo for running training or linear models.
'''

import utils
from kahip.kmkahip import run_kmkahip
from kahip import create_graph
import data_extractor

    
def run(data_loader,graph_file):
    opt = utils.parse_args()

    #adjust the number of parts and the height of the hierarchy
    n_cluster_l = [2]
    height_l = [1]
    
    
    # load dataset 
    dataset = utils.load_data(data_loader, 'train').to(utils.device)
    queryset = utils.load_data(data_loader, 'query').to(utils.device)    
    neighbors = utils.load_data(data_loader, 'answers').to(utils.device)

    #specify which action to take at each level, actions can be km, kahip, train, or svm. Lower keys indicate closer to leaf.
    #Note that if 'kahip' is included, evaluation must be on training rather than test set, since partitioning was performed on training, but not test, set.
    #e.g.: opt.level2action = {0:'km', 1:'train', 3:'train'}
    opt.level2action = {0:'train'}
    opt.data_dir='/data/kabir/similarity-search/models/lth-data'
    opt.graph_file=graph_file
    data_x=utils.get_data_sift1M()[0]
    grp=create_graph.create_knn_graph(data_x,100)
    create_graph.write_knn_graph(grp,opt.data_dir+'/'+opt.graph_file)
    print(opt)
    for n_cluster in n_cluster_l:
        print('n_cluster {}'.format(n_cluster))
        opt.n_clusters = n_cluster
        opt.n_class = n_cluster
        for height in height_l:
            run_kmkahip(height, opt, dataset, queryset, neighbors)

run(data_extractor.get_data_sift,'sift-0-knn.graph')