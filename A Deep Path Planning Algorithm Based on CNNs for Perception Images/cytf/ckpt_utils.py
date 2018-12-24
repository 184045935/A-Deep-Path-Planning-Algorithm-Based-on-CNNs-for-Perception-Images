# coding: utf-8

# TODO: 规范化注释

def freeze_graph(model_dir, output_node_names):
    '''
    Args:
        model_dir: ckpt 系列文件所在目录
        output_node_names: list, 包含要导出的 node 的名字(不带后面的:0)
    '''

    checkpoint = tf.train.get_checkpoint_state(model_dir)
    input_checkpoint = checkpoint.model_checkpoint_path
    output_graph = os.path.join(model_dir, 'frozen_model.pb')
  
    with tf.Session(graph=tf.Graph()) as sess:
        # 导入 MetaGraphDef 到当前 Graph
        saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)

        # 恢复变量值到当前 Session
        saver.restore(sess, input_checkpoint)

        # 转换为包含常量形式的变量值的 GraphDef
        output_graph_def = \
        tf.graph_util.convert_variables_to_constants(
            sess, 
            tf.get_default_graph().as_graph_def(), 
            output_node_names=output_node_names)

        # 储存
        with tf.gfile.GFile(output_graph, 'wb') as f:
            f.write(output_graph_def.SerializeToString()) 

def load_graph(frozen_graph_filename):

    # 从磁盘加载 frozen_model.pb 文件
    with tf.gfile.GFile(frozen_graph_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # 返回的时候默认给每个 node 名字前加 'prefix'，
    # 避免对当前 graph 中的变量产生影响.
    # 注意：如果不设定 name, tf 自动加前缀 'import'.
    # 令 name = '' 就可以不加前缀
    # 返回一个 graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='prefix')
    return graph

def replace_var_names_of_ckpt(old_name_list, new_name_list, all_vars_list):
    '''对旧的 ckpt 中的权重变量等进行更名
    old_name_list: 旧图中需要更改的变量名
    new_name_list: 旧图中需要更改为的对应新图中的变量名
    all_vars_list: 为 str 时是 ckpt 文件路径；为 dict 时是 reader.get_variable_to_shape_map() 返回值;
                   为 list 时是旧图中的所有变量名列表
    '''

    if isinstance(all_vars_list, str):
        all_vars_list = tf.train.NewCheckpointReader(all_vars_list).get_variable_to_shape_map().keys()
    elif isinstance(all_vars_list, dict):
        all_vars_list = all_vars_list.keys()

    # 保留未更改的名字
    for name in old_name_list:
        all_vars_list.remove(name)
    variables_to_restore = tf.contrib.framework.get_variables_to_restore(all_vars_list)
    # 组成dict
    load_var_dict = dict(zip(all_vars_list, variables_to_restore))

    new_name_var_list = [tf.get_collection('variables', new_name)[0] for new_name in new_name_list]

    load_var_dict.update(dict(zip(old_name_list, new_name_var_list)))

    return load_var_dict

