import numpy as np
import os
from collections import OrderedDict
from math import hypot
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
    
def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / N
    
def calc_running_mean_data(data, window_len):
    # build new datastructure here containing all data params (columns) as deltas
    data_running_mean_delta = []
    for i in range(data.shape[1]):
        running_mean_i = running_mean(data[:,i], window_len)
        #substract 'next' element from previous running mean
        # e.g. calculate running mean of 3 elements -> substract it from element 4
        delta_to_running_mean_i = abs(data[window_len:,i] - running_mean_i[:-1])
        # align (no delta for first N elements, as there is no running_mean for them)
        delta_to_running_mean_i = np.insert(delta_to_running_mean_i, 0, np.zeros(window_len)) 
        data_running_mean_delta.append(delta_to_running_mean_i)

    # convert params list to matrix by stacking params horizontally (as columns)
    data_running_mean_delta = np.stack(data_running_mean_delta, axis=1) 
    return data_running_mean_delta

'''
    Split data into training and validation set based on map coordinates.
    
    args:
        raw_data (list): List of raw map data containing map coordinates in first columns
        data (list): List of map data (features)
        axis (int): (Map) axis to split data along. E.g. 0 uses first column of raw_data to calculate split position.
        val_train_ratio (float): Ratio of validation to training data. E.g. 0.25 takes 1/4 as validation and 3/4 as training data.
        
    returns:
        Tuple with 4 elements: (Raw train data, Raw validation data, train data, validation data)
'''
def train_val_split(raw_data, data, labels, axis, val_train_ratio):
    # calculate coordinate to split data at according to val_train_ratio
    axis_min_coord = np.min(raw_data[:,axis])
    split_coord = ((np.max(raw_data[:,axis]) - axis_min_coord) * val_train_ratio ) + axis_min_coord

    # now split the data using the row ids we calculated above
    train_ids = np.where(raw_data[:,axis] > split_coord)
    val_ids = np.where(raw_data[:,axis] <= split_coord)

    raw_data_train = raw_data[train_ids]
    raw_data_val = raw_data[val_ids]       
    data_train = data[train_ids]
    data_val = data[val_ids]
    labels_train = labels[train_ids]
    labels_val = labels[val_ids]

    return raw_data_train, raw_data_val, data_train, data_val, labels_train, labels_val

def plot_equal(ax, legend_linewidth=10):
    legend = ax.legend(frameon = 1, fontsize=16)
    for i in range(len(legend.legendHandles)):
        legend.legendHandles[i].set_linewidth(legend_linewidth)
    frame = legend.get_frame()
    frame.set_color('white')
    frame.set_edgecolor('black')
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
    
'''
    Find continuos cluster segments along the trajectory.
    This makes it possible to plot a multicolored lineplot instead of a scatter plot
    which is a lot more performant when rendered as a pdf (because line simplification is then possible).
'''
def find_cluster_segments(raw_data, labels, min_seg_distance=.14):
    segs = []
    seg2cluster = []
    curr_seg = []
    prev_label = ''
    prev_raw = ''
    for raw, l in zip(raw_data, labels):
        if (l == prev_label and hypot(raw[0]-prev_raw[0], raw[1]-prev_raw[1]) < min_seg_distance) or prev_label == '':
            curr_seg.append((raw[0], raw[1]))
        else:
            segs.append(curr_seg)
            seg2cluster.append(prev_label)
                        
            curr_seg = []
            # start next segment with endpoint of prev segment if distance is less than threshold
            # (don't join disconnected trajectories)
            if hypot(raw[0]-prev_raw[0], raw[1]-prev_raw[1]) < .14:
                curr_seg.append((prev_raw[0], prev_raw[1]))
            curr_seg.append((raw[0], raw[1]))
        prev_label = l
        prev_raw = raw
        
    segs.append(curr_seg) # the final one
    seg2cluster.append(l)
    return segs, seg2cluster

def plot_clusters_lineplot(ax, map_, labels, clusters_n, cluster_type, clusters_first_label=0, raw_data=None, filter_clusters=None, min_seg_distance=.14, linewidth=.5, legend_fontsize=16, legend_linewidth=1.5, colors=['#001277', '#FF5A03', '#9E0008', '#773E9E', '#156E22']):
    
    ax.set_title(cluster_type)
    map_.plot_map(ax)
    
    segments, seg2cluster = find_cluster_segments(map_.raw_data if raw_data is None else raw_data, labels, min_seg_distance=min_seg_distance)
    
    legend = {}
    for seg, cluster in zip(segments, seg2cluster):
        if filter_clusters is not None and cluster not in filter_clusters:
            continue
        lc = LineCollection([seg], linewidth=linewidth, color=colors[cluster] if len(colors)>cluster else None)
        ax.add_collection(lc)
        legend["cluster-{}".format(cluster)] = lc
        
    ax.autoscale()
    sorted_legend = OrderedDict(sorted(legend.items()))
    legend = ax.legend(sorted_legend.values(), sorted_legend.keys(), frameon=1, fontsize=legend_fontsize)
    for i in range(len(legend.legendHandles)):
        legend.legendHandles[i].set_linewidth(legend_linewidth)
    frame = legend.get_frame()
    frame.set_color('white')
    frame.set_edgecolor('black')
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
    
def plot_errors_lineplot(ax, map_, predictions, labels, raw_data=None, linewidth=.5, legend_fontsize=16, legend_linewidth=1.5):
    #ax.set_title('{} clusters on {}'.format(cluster_type, map_.name))
    map_.plot_map(ax)
    
    if raw_data is None:
        error_data = map_.raw_data[np.where(predictions != labels)]
    else:
        error_data = raw_data[np.where(predictions != labels)]
    
    # abuse find_cluster_segments to get path segments (of only one faked cluster 0)
    segments, seg2cluster = find_cluster_segments(error_data, np.zeros(len(error_data)))
    
    legend = {}
    for seg, cluster in zip(segments, seg2cluster):
        lc = LineCollection([seg], linewidth=linewidth, color='red')
        ax.add_collection(lc)
        legend["Cluster/prediction differences".format(cluster)] = lc
        
    ax.autoscale()
    sorted_legend = OrderedDict(sorted(legend.items()))
    legend = ax.legend(sorted_legend.values(), sorted_legend.keys(), frameon=1, fontsize=legend_fontsize)
    for i in range(len(legend.legendHandles)):
        legend.legendHandles[i].set_linewidth(legend_linewidth)
    frame = legend.get_frame()
    frame.set_color('white')
    frame.set_edgecolor('black')
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
    
def plot_errors(map_, predictions, labels):    
    for modelfile, pred in predictions.items():
        fig, ax = plt.subplots(figsize=(20,20))
        ax.set_title('Prediction on {} using {} model'.format(map_.name, modelfile))
        map_.plot_map(ax)
        error_data = map_.raw_data[np.where(pred != labels)]
        ax.scatter(error_data[:,0], error_data[:,1], label='prediction/clustering mismatches', s=1, lw=0, color='red')

        plot_equal(ax)
    
def saveplot(fig, filename='figure', folder='./plots_tmp', format='pdf', dpi=600, rasterized=False):
    try:
        os.makedirs(folder)
    except:
        pass
    
    target = os.path.join(folder,'{}.{}'.format(filename,format))
    print('Saving plot to {} ...'.format(target))
    fig.savefig(target, format=format, dpi=dpi, bbox_inches='tight', rasterized=rasterized)
    print('Done')
    
'''
    Write logfile for tensorboard projector visualizing data points and labels.
    Note: Resets default tensorflow graph.
'''
def write_tf_projector(maps_labels_headers, limit=None, log_dir='./logs'):
    try:
        os.makedirs(log_dir)
    except:
        pass
    
    tf.reset_default_graph()
    
    embedding_vars = []
    metadata_filenames = []
    for (map_, label_matrix, header) in maps_labels_headers:
        
        data = map_.data if limit is None else map_.data[:limit]
        
        #write data points to file for online tf projector
        data.tofile(os.path.join(log_dir, '{}.bytes'.format(map_.name.replace(" ", "_"))))
        
        metadata_filename = 'metadata_{}.tsv'.format(map_.name)
        with open(os.path.join(log_dir, metadata_filename), 'w') as metadata_file:
            if(label_matrix.ndim>1): #tf requires to write header row only if there is more than a single column of labels
                metadata_file.write('{}\n'.format('\t'.join(header)))
                
            if limit is not None:
                label_matrix = label_matrix[:limit]
            for label_row in label_matrix:
                labels_arrstr = np.char.mod('%d', label_row)
                metadata_file.write('{}\n'.format('\t'.join(labels_arrstr)))
        metadata_filenames.append(metadata_filename)

        embedding_var = tf.Variable(data, name=map_.name.replace(" ", "_"))
        embedding_vars.append(embedding_var)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        saver.save(sess, os.path.join(log_dir, "model.ckpt"))

        config = projector.ProjectorConfig()

        for embedding_var, metadata_filename in zip(embedding_vars, metadata_filenames):
            embedding = config.embeddings.add()
            embedding.tensor_name = embedding_var.name
            embedding.metadata_path = metadata_filename

        summary_writer = tf.summary.FileWriter(log_dir)
        projector.visualize_embeddings(summary_writer, config)