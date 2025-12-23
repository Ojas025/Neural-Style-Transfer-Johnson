def gram_matrix(feature_map):
    (batch_size, channels, height, width) = feature_map.size()
    
    # flatten for matrix multiplication
    features = feature_map.view(batch_size, channels, height * width)
    transposed_features = features.transpose(1, 2)
    
    # batch matrix multiplication
    gram = features.bmm(transposed_features)
    
    # normalize gram matrix
    gram /= channels * height * width
    
    return gram