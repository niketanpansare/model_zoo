import lmdb, caffe, os, cv2
import numpy as np
from StringIO import StringIO
# pip install opencv-python

def get_pretty_str(key, value):
    return '\t"' + key + '": ' + str(value) + ',\n'
	
def write_mtd(file_path, rows, cols, nnz):
    with open(file_path, 'w') as file:
        file.write('{\n\t"data_type": "matrix",\n\t"value_type": "double",\n')
	file.write(get_pretty_str('rows', rows))
	file.write(get_pretty_str('cols', cols))
	file.write(get_pretty_str('nnz', nnz))
	file.write('\t"format": "csv",\n\t"description": {\n\t\t"author": "SystemML"\n\t}\n}\n')
	file.write(str(cols))

def convert_caffemodel(network_file, caffemodel_file, output_dir, format='csv'):
    """
    Saves the weights and bias in the caffemodel file to output_dir. This method requires caffe to be installed.

    Parameters
    ----------
    network_file: string
        Path to the input network file
        
    caffemodel_file: string
        Path to the input caffemodel file
        
    output_dir: string
        Output directory for weights/biases (local filesystem)
    
    format: string
        Output format for weights/biases (currently supported formats: csv)
    """
    if format != 'csv':
        raise ValueError('Only csv format supported in this version')
    net = caffe.Net(network_file, caffemodel_file, caffe.TEST)
    for layerName in net.params.keys():
	num_parameters = len(net.params[layerName])
        if num_parameters == 0:
            continue
        elif num_parameters == 2:
            # Weights and Biases
            w = net.params[layerName][0].data
            w = w.reshape(w.shape[0], -1)
            b = net.params[layerName][1].data
            b = b.reshape(b.shape[0], -1)
	    layerType = net.layers[list(net._layer_names).index(layerName)].type
            if layerType == 'InnerProduct':
                b = b.T
                w = w.T
            np.savetxt(os.path.join(output_dir, layerName + '_weight.mtx'), w, delimiter=',')
            np.savetxt(os.path.join(output_dir, layerName + '_bias.mtx'), b, delimiter=',')
            write_mtd(os.path.join(output_dir, layerName + '_weight.mtx.mtd'), w.shape[0], w.shape[1], np.count_nonzero(w))
            write_mtd(os.path.join(output_dir, layerName + '_bias.mtx.mtd'), b.shape[0], b.shape[1], np.count_nonzero(b))
        else:
            raise ValueError('Unsupported number of parameters:' + str(num_parameters))

def save_lmdb(lmdb_file, output_dir, labels_file):
    """
    Saves the images in the lmdb file to output_dir. This method requires caffe to be installed.

    Parameters
    ----------
    lmdb_file: string
        Path to the input lmdb file
        
    output_dir: string
        Output directory for images (local filesystem)
    
    labels_file: string
        Output file (local filesystem) which saves the file names and associated labels. Format: 'output_file_path label'
    """
    lmdb_cursor = lmdb.open(lmdb_file, readonly=True).begin().cursor()
    datum = caffe.proto.caffe_pb2.Datum()
    i = 1
    with open(labels_file, 'w') as file:
        for _, value in lmdb_cursor:
            datum.ParseFromString(value)
            label = datum.label
            data = caffe.io.datum_to_array(datum)
            #data = np.transpose(data, (1,2,0))
            #data = np.rollaxis(data, 0,3)
            #s = StringIO()
            #s.write(datum.data)
            #s.seek(0)
            #PIL.Image.open(s).save(output_file_path, 'JPEG')
            #PIL.Image.open(PIL.Image.io.BytesIO(datum.data)).save(output_file_path, 'JPEG')
            output_file_path = os.path.join(output_dir, 'file_' + str(i) + '.jpg')
            file.write(output_file_path + ' ' + str(datum.label) + '\n')
            image = np.transpose(data, (1,2,0)) # CxHxW to HxWxC in cv2
            cv2.imwrite(output_file_path, image)
            i = i + 1

